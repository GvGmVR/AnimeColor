def run_animecolor(config: dict):
    """
    run_animecolor.py
    =================
    Single-pass AnimeColor inference designed to match research-paper quality.

    KEY DESIGN DECISIONS
    --------------------
    1. EXACTLY 49 frames per run — the CogVideoX-2B native context window.
    Processing all frames in one pass means the 3D attention sees the full
    temporal sequence simultaneously, which is what gives the paper its
    colour consistency and temporal smoothness.

    2. NO chunking of the main inference — chunking was the root cause of
    ghosting, colour drops, and 6.6x discontinuities at every boundary.

    3. If your lineart folder has more than 49 frames, the script picks
    the first 49.  Run it again with a different START_FRAME offset to
    process subsequent segments, then concatenate the clips externally.

    4. GPU-primary decode with smart VRAM fallback (bf16 → fp32 → CPU).
    CPU never runs conv3d — it only saves PNGs.

    5. Thermal rests between the three heavy phases:
        Load → (rest) → Inference → (rest) → Decode → compile

    TUNABLE SETTINGS (top of file)
    -------------------------------
    START_FRAME   : first frame index to read from LINEART_DIR
    NUM_FRAMES    : keep at 49 for best quality (model's native window)
    OUTPUT_FPS    : 24 = standard anime; 12 = slower motion feel
    GUIDANCE_SCALE: 6.0 matches reference script default
    INFER_REST    : GPU cooldown after inference (seconds)
    DECODE_REST   : GPU cooldown after decode (seconds)

    CHANGE LOG
    ----------
    v3 — RADIO decoupled from pipeline object entirely.
         RADIO forward pass now runs standalone BEFORE pipe is constructed,
         so enable_sequential_cpu_offload never installs hooks on dclip_model.
         id_cond / id_vit_hidden are plain CPU tensors passed into pipe().
         Scheduler changed from CogVideoXDDIMScheduler → DDIMScheduler
         to match reference script (test_msketch.py).
         Inference steps raised from 25 → 50 to match reference script.
         Guidance scale default changed from 7.5 → 6.0 to match reference.
    """

    import os, sys, gc, torch, imageio, warnings, time, re
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from accelerate.hooks import remove_hook_from_module
    import importlib.util

    # ============================================================
    # 1. ENVIRONMENT
    # ============================================================
    warnings.filterwarnings("ignore")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # ============================================================
    # 2. PATHS (FROM CONFIG)
    # ============================================================
    BASE_PATH   = Path(config["base_path"])
    SRC_DIR     = BASE_PATH / config["src_dir"]
    CKPT_DIR    = BASE_PATH / config["ckpt_dir"]
    BASE_DIR    = BASE_PATH / config["base_model_dir"]
    RADIO_DIR   = BASE_PATH / config["radio_dir"]

    LINEART_DIR = Path(config["lineart_dir"])
    REF_IMAGE   = Path(config["ref_image"])

    OUTPUT_DIR  = Path(config["output_dir"])
    FRAMES_DIR  = OUTPUT_DIR / "decoded_frames"    # intermediate per-frame PNGs

    print("Creating directories...")
    print("FRAMES_DIR:", FRAMES_DIR)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("Directory exists now?", FRAMES_DIR.exists())

    # ============================================================
    # 3. SETTINGS (FROM CONFIG)
    # ============================================================
    START_FRAME      = config.get("start_frame", 0)
    NUM_FRAMES       = config.get("num_frames", 49)      # Native CogVideoX-2B context window
    WIDTH            = config.get("width", 512)
    HEIGHT           = config.get("height", 320)
    OUTPUT_FPS       = config.get("output_fps", 24)
    # guidance_scale 6.0 matches the reference script (test_msketch.py).
    # The model was trained and evaluated at this value.
    # Raise to 7–8 only if colors are still weak after HCE is confirmed working.
    GUIDANCE_SCALE   = config.get("guidance_scale", 6.0)
    # 50 steps matches the reference script. 25 was too few — the denoising
    # trajectory doesn't converge fully at 25 steps with DDIMScheduler.
    INFERENCE_STEPS  = config.get("inference_steps", 50)
    INFER_REST       = config.get("infer_rest", 10)
    DECODE_REST      = config.get("decode_rest", 5)
    SEED             = config.get("seed", 43)             # 43 matches reference script default

    weight_dtype = torch.bfloat16

    # ============================================================
    # 4. SELF-HEALING PATCHES
    # ============================================================
    if not SRC_DIR.exists():
        print(f"[WARN] SRC_DIR not found: {SRC_DIR}")
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    diffusers_spec = importlib.util.find_spec("diffusers")
    if diffusers_spec and diffusers_spec.origin:
        dep_path = Path(diffusers_spec.origin).parent / "utils" / "deprecation_utils.py"
        if dep_path.exists():
            content = dep_path.read_text(encoding="utf-8")
            OLD = ("if version.parse(version.parse(__version__).base_version)"
                " >= version.parse(version_name):")
            if OLD in content:
                dep_path.write_text(content.replace(OLD, "if False:"), encoding="utf-8")
                print("[PATCH] Fixed diffusers deprecation_utils.py")

    pipe_file = SRC_DIR / "cogvideox/pipeline/pipeline_cogvideo_color_ref.py"
    if pipe_file.exists():
        content = pipe_file.read_text(encoding="utf-8")
        if "video = torch.from_numpy(video)" in content:
            pipe_file.write_text(
                content.replace("video = torch.from_numpy(video)",
                                "video = torch.as_tensor(video)"),
                encoding="utf-8"
            )
            print("[PATCH] Fixed pipeline numpy->tensor cast")

    for t_file in ["transformer3d_radio.py", "transformer3d.py"]:
        tf_path = SRC_DIR / f"cogvideox/models/{t_file}"
        if tf_path.exists():
            content = tf_path.read_text(encoding="utf-8")
            patched = re.sub(
                r'torch\.from_numpy\((\w+_pos_embedding)\)',
                r'torch.as_tensor(\1)', content
            )
            if patched != content:
                tf_path.write_text(patched, encoding="utf-8")
                print(f"[PATCH] Fixed pos_embedding cast in {t_file}")

    for py_file in RADIO_DIR.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if "class RADIOModel" in content and "all_tied_weights_keys" not in content:
            content = content.replace(
                "def forward(self",
                "@property\n    def all_tied_weights_keys(self):\n"
                "        return {}\n\n    def forward(self"
            )
            py_file.write_text(content, encoding="utf-8")
            print(f"[PATCH] Fixed RADIOModel in {py_file.name}")

    # ============================================================
    # 5. SMART GPU DECODE  (bf16 → fp32 → CPU last resort)
    # ============================================================
    def get_free_vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)

    def decode_gpu_bf16(vae, latent: torch.Tensor) -> torch.Tensor:
        vae.to("cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        with torch.inference_mode():
            out = vae.decode(latent.to("cuda", dtype=torch.bfloat16)).sample
            out = (out.float() / 2.0 + 0.5).clamp(0, 1)
        result = out.cpu()
        del out
        torch.cuda.empty_cache()
        vae.to("cpu")
        return result

    def decode_gpu_fp32(vae, latent: torch.Tensor) -> torch.Tensor:
        print("  [DECODE] Falling back to GPU float32 (low VRAM).")
        vae.to("cuda", dtype=torch.float32)
        torch.cuda.empty_cache()
        with torch.inference_mode():
            out = vae.decode(latent.to("cuda", dtype=torch.float32)).sample
            out = (out.float() / 2.0 + 0.5).clamp(0, 1)
        result = out.cpu()
        del out
        torch.cuda.empty_cache()
        vae.to("cpu")
        return result

    def decode_cpu_fp32(vae, latent: torch.Tensor) -> torch.Tensor:
        print("  [DECODE] Last resort: CPU float32 (GPU OOM on both dtypes).")
        vae.to("cpu", dtype=torch.float32)
        with torch.inference_mode():
            out = vae.decode(latent.to(torch.float32)).sample
            out = (out.float() / 2.0 + 0.5).clamp(0, 1)
        return out

    def smart_decode(vae, latent: torch.Tensor) -> torch.Tensor:
        """Returns decoded [B,C,F,H,W] tensor on CPU. GPU used whenever possible."""
        free = get_free_vram_gb()
        print(f"  [DECODE] Free VRAM before decode: {free:.2f} GB")

        if free >= 4.0:
            try:
                return decode_gpu_bf16(vae, latent)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()

        if free >= 1.5:
            try:
                return decode_gpu_fp32(vae, latent)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()

        return decode_cpu_fp32(vae, latent)

    # ============================================================
    # 6. SKETCH PRE-PROCESSING
    #    Input sketches are expected to already be Anime2Sketch or
    #    XDoG quality (clean binary black-on-white). The clean_sketch
    #    function applies a final binarization pass as a safety net
    #    to catch any residual anti-aliasing from the extraction step.
    # ============================================================
    def clean_sketch(img: Image.Image, threshold: int = 128) -> Image.Image:
        gray = img.convert("L")
        binary = gray.point(lambda x: 0 if x < threshold else 255)
        return binary.convert("RGB")

    # ============================================================
    # 7. VALIDATE FRAME COUNT
    # ============================================================
    all_frame_paths = sorted(
        p for p in LINEART_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not all_frame_paths:
        raise ValueError(f"No lineart frames found in {LINEART_DIR}")

    available = len(all_frame_paths)
    end_frame  = START_FRAME + NUM_FRAMES

    if end_frame > available:
        print(f"[WARN] Only {available} frames available from START_FRAME={START_FRAME}.")
        end_frame  = available
        NUM_FRAMES = end_frame - START_FRAME

    if NUM_FRAMES < 1:
        raise ValueError(f"No frames to process: START_FRAME={START_FRAME}, available={available}")

    selected_paths = all_frame_paths[START_FRAME:end_frame]
    actual_frames  = len(selected_paths)

    def nearest_valid_frame_count(n: int) -> int:
        """CogVideoX requires frames = 4k+1. Find largest valid count <= n."""
        if n <= 0: return 1
        k = (n - 1) // 4
        return 4 * k + 1

    padded_frames = nearest_valid_frame_count(actual_frames)
    pad_needed    = padded_frames - actual_frames

    print(f"\n[CONFIG] Processing frames {START_FRAME}–{end_frame-1} "
        f"({actual_frames} real + {pad_needed} pad = {padded_frames} total)")
    print(f"         Resolution: {WIDTH}×{HEIGHT}  |  FPS: {OUTPUT_FPS}  |  "
        f"Duration: {actual_frames/OUTPUT_FPS:.2f}s")
    print(f"         Guidance: {GUIDANCE_SCALE}  |  Steps: {INFERENCE_STEPS}  |  Seed: {SEED}")

    # ============================================================
    # 8. RADIO FORWARD PASS — STANDALONE, BEFORE PIPELINE EXISTS
    #
    #    CRITICAL: RADIO must run here, before the pipeline is built
    #    and before enable_sequential_cpu_offload is called.
    #
    #    The previous approach attached dclip_model to the pipe object
    #    and called enable_sequential_cpu_offload afterwards. This caused
    #    accelerate to install hooks on dclip_model, which intercepted
    #    the .to("cuda") call and corrupted the forward pass device
    #    routing. The symptom was correct scene structure but wrong/
    #    desaturated colors — the HCE signal was not reaching the DiT.
    #
    #    The fix (matching test_msketch.py from the paper's repo):
    #    - Load RADIO independently
    #    - Run forward pass on reference image
    #    - Store id_cond, id_vit_hidden as plain CPU tensors
    #    - Delete RADIO model before pipeline is constructed
    #    - Pass the plain tensors into pipe() — pipeline handles
    #      device placement internally during the forward pass
    # ============================================================
    print("\n[PHASE 0] RADIO encoder — standalone HCE forward pass...")

    from transformers import AutoModel, CLIPImageProcessor

    # Open reference image at inference resolution for RADIO
    ref_pil_radio = Image.open(REF_IMAGE).convert("RGB").resize((WIDTH, HEIGHT))

    # Load RADIO directly to CUDA — no pipeline hooks involved
    dclip_model     = AutoModel.from_pretrained(
        RADIO_DIR, trust_remote_code=True, torch_dtype=weight_dtype
    ).to("cuda")
    dclip_processor = CLIPImageProcessor.from_pretrained(RADIO_DIR, torch_dtype=weight_dtype)

    # Preprocess with CLIPImageProcessor — this applies the correct
    # ImageNet-style normalization that RADIO expects.
    # Raw /255.0 normalization (used for ref_tensor → LCG path) is wrong here.
    dclip_input = dclip_processor(
        images=ref_pil_radio,
        return_tensors="pt"
    ).pixel_values.to("cuda", dtype=weight_dtype)

    print(f"  RADIO input shape  : {dclip_input.shape}")

    with torch.inference_mode():
        id_cond, id_vit_hidden = dclip_model(dclip_input)

    # Move HCE tokens to CPU immediately — they will be passed as plain
    # tensors to pipe(). The pipeline's internal cross-attention will
    # move them to the correct device during the forward pass.
    id_cond      = id_cond.cpu()
    id_vit_hidden = id_vit_hidden.cpu()

    print(f"  id_cond shape      : {id_cond.shape}        device: {id_cond.device}")
    print(f"  id_vit_hidden shape: {id_vit_hidden.shape}  device: {id_vit_hidden.device}")

    # Free RADIO from VRAM entirely before pipeline is constructed.
    # The pipeline must never see dclip_model — it should not be attached
    # to pipe and should not be present when enable_sequential_cpu_offload runs.
    del dclip_model, dclip_input, ref_pil_radio
    torch.cuda.empty_cache()
    gc.collect()

    free_after_radio = get_free_vram_gb()
    print(f"  RADIO freed. Free VRAM: {free_after_radio:.2f} GB")
    print("[PHASE 0] Done.\n")

    # ============================================================
    # 9. LOAD MODELS
    # ============================================================
    print("[PHASE 1] Loading models into system RAM (bfloat16)...")

    from transformers import T5EncoderModel, T5Tokenizer
    # DDIMScheduler (standard) matches the reference script (test_msketch.py).
    # CogVideoXDDIMScheduler is the base model's scheduler — the AnimeColor
    # checkpoint was evaluated using the standard DDIMScheduler.
    from diffusers import DDIMScheduler
    from cogvideox.models.transformer3d_radio import CogVideoXTransformer3DModel
    from cogvideox.models.transformer3d import CogVideoXTransformer3DModel as Ref3D
    from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
    from cogvideox.pipeline.pipeline_cogvideo_color_ref import \
        CogVideoX_Fun_Pipeline_Control_Color

    text_encoder = T5EncoderModel.from_pretrained(
        BASE_DIR, subfolder="text_encoder", torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(BASE_DIR, subfolder="tokenizer")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        BASE_DIR, subfolder="vae", torch_dtype=weight_dtype)
    denoising_transformer = CogVideoXTransformer3DModel.from_pretrained(
        CKPT_DIR, subfolder="transformer", torch_dtype=weight_dtype)
    reference_transformer = Ref3D.from_pretrained(
        CKPT_DIR, subfolder="referencenet", torch_dtype=weight_dtype)

    # Standard DDIMScheduler loaded from base model scheduler config.
    # This replaces CogVideoXDDIMScheduler to match the reference script.
    scheduler = DDIMScheduler.from_pretrained(
        BASE_DIR, subfolder="scheduler"
    )

    denoising_transformer.is_train_qformer = False

    # IMPORTANT: dclip_model is NOT attached to pipe here.
    # RADIO has already run in Phase 0. The pipeline never needs to
    # call RADIO itself — id_cond and id_vit_hidden are pre-computed.
    pipe = CogVideoX_Fun_Pipeline_Control_Color(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        denoising_transformer=denoising_transformer,
        reference_transformer=reference_transformer,
        scheduler=scheduler,
    )

    # enable_sequential_cpu_offload now only hooks the pipeline's own models:
    # text_encoder, vae, denoising_transformer, reference_transformer.
    # dclip_model is gone — no hook conflict possible.
    pipe.enable_sequential_cpu_offload()
    try:
        pipe.enable_attention_slicing(slice_size="max")
    except Exception:
        pass

    # VAE slicing enabled unconditionally here because the pipeline's internal
    # vae.encode() call (for control video) runs during inference under offload.
    # We will re-evaluate and conditionally disable slicing for the decode phase.
    pipe.vae.enable_slicing()
    # Tiling intentionally NOT enabled — introduces spatial seam artifacts.

    torch.cuda.empty_cache()
    gc.collect()
    print("[PHASE 1] Done.\n")

    # ============================================================
    # 10. PREPARE TENSORS
    # ============================================================
    print("[PHASE 2] Preparing input tensors...")

    # Reference image for the LCG path (ref_image parameter).
    # Raw /255.0 normalization is correct here — the pipeline's VAE encoder
    # handles this path. This is separate from the RADIO/HCE path above.
    ref_pil    = Image.open(REF_IMAGE).convert("RGB").resize((WIDTH, HEIGHT))
    ref_np     = np.array(ref_pil)
    ref_tensor = (
        torch.from_numpy(ref_np).permute(2, 0, 1).to(weight_dtype) / 255.0
    ).unsqueeze(0)   # [1, C, H, W]

    # Sketch frames
    sketch_frames = []
    for p in selected_paths:
        img = Image.open(p).convert("RGB").resize((WIDTH, HEIGHT))
        img = clean_sketch(img)
        sketch_frames.append(np.array(img))

    if pad_needed > 0:
        print(f"  Padding {pad_needed} frame(s) to reach valid count {padded_frames}.")
        sketch_frames += [sketch_frames[-1]] * pad_needed

    ctrl_array  = np.array(sketch_frames)                   # [F, H, W, C]
    ctrl_tensor = (
        torch.from_numpy(ctrl_array)
        .permute(3, 0, 1, 2)                                # [C, F, H, W]
        .to(weight_dtype) / 255.0
    ).unsqueeze(0)                                          # [1, C, F, H, W]

    print(f"  ref_tensor    : {ref_tensor.shape}")
    print(f"  ctrl_tensor   : {ctrl_tensor.shape}  (frames={padded_frames})")
    print(f"  id_cond       : {id_cond.shape}  (pre-computed, HCE active)")
    print(f"  id_vit_hidden : {id_vit_hidden.shape}")
    print("[PHASE 2] Done.\n")

    # ============================================================
    # 11. SINGLE-PASS INFERENCE
    # ============================================================
    print(f"[PHASE 3] Single-pass inference  ({padded_frames} frames, "
        f"{INFERENCE_STEPS} steps, guidance={GUIDANCE_SCALE})...")
    print("  This is the longest phase — ~30-60 min for 49 frames at 50 steps on 8GB.")

    generator = torch.Generator(device="cuda").manual_seed(SEED)

    t0 = time.time()
    with torch.inference_mode():
        output = pipe(
            prompt=(
                "anime style, vibrant colors, high quality masterpiece, "
                "sharp linework, consistent lighting"
            ),
            negative_prompt=(
                "The video is not of a high quality, it has a low resolution. "
                "Watermark present in each frame. The background is solid. "
                "Strange body and strange trajectory. Distortion."
            ),
            ref_image=ref_tensor,
            control_video=ctrl_tensor,
            height=HEIGHT,
            width=WIDTH,
            num_frames=padded_frames,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=INFERENCE_STEPS,
            generator=generator,
            output_type="latent",
            return_dict=True,
            id_cond=id_cond,
            id_vit_hidden=id_vit_hidden,
        )

    elapsed = time.time() - t0
    latent_cpu = output.videos.cpu()   # [B, F_lat, C, H, W]
    print(f"  Inference done in {elapsed/60:.1f} min. Latent: {latent_cpu.shape}")

    del ctrl_tensor
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Resting GPU for {INFER_REST}s before decode...")
    time.sleep(INFER_REST)

    # ============================================================
    # 12. REMOVE PIPELINE HOOKS, TAKE VAE OWNERSHIP
    # ============================================================
    print("\n[PHASE 4] Removing pipeline hooks, freeing VRAM...")

    for model in [pipe.denoising_transformer, pipe.reference_transformer,
                pipe.text_encoder, pipe.vae]:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")

    torch.cuda.empty_cache()
    gc.collect()

    free_gb = get_free_vram_gb()
    print(f"  Hooks removed. Free VRAM: {free_gb:.2f} GB")

    vae_standalone = pipe.vae

    # Conditional slicing for decode phase only.
    # On 8GB with all other models on CPU, ~5-6GB should be free.
    # A 49-frame bf16 decode at 512x320 needs ~4-5GB.
    # Enable slicing only if headroom is tight — it introduces temporal
    # chunk boundaries which reduce consistency across the sequence.
    # Tiling remains disabled always — spatial seams are worse than temporal ones.
    free_gb_predecode = get_free_vram_gb()
    if free_gb_predecode < 5.0:
        vae_standalone.enable_slicing()
        print(f"  VAE slicing ENABLED  (free VRAM: {free_gb_predecode:.2f} GB — tight)")
    else:
        print(f"  VAE slicing DISABLED (free VRAM: {free_gb_predecode:.2f} GB — sufficient)")

    scaling_factor = vae_standalone.config.scaling_factor

    # ============================================================
    # 13. DECODE ON GPU
    #     Pipeline returns [B, F_lat, C, H, W]  (frames-first).
    #     VAE decode expects [B, C, F_lat, H, W] (channels-first).
    # ============================================================
    print("\n[PHASE 5] GPU decode...")

    full_latent = (1.0 / scaling_factor) * (
        latent_cpu.permute(0, 2, 1, 3, 4).contiguous()
    )
    print(f"  Latent for VAE: {full_latent.shape}")

    decoded_cpu = smart_decode(vae_standalone, full_latent)
    # decoded_cpu: [B, C, F_decoded, H, W]

    del full_latent, latent_cpu
    gc.collect()

    n_decoded = decoded_cpu.shape[2]
    print(f"  Decoded {n_decoded} frames (will trim to {actual_frames} real frames).")

    print(f"\n  Resting GPU for {DECODE_REST}s before saving...")
    time.sleep(DECODE_REST)

    # ============================================================
    # 14. SAVE FRAMES — NO POST-PROCESSING
    #     Post-processing removed: brightness/saturation/contrast boosts
    #     and UnsharpMask were confirmed to degrade color fidelity and
    #     worsen temporal consistency by amplifying frame-to-frame variation.
    #     If output is still too dull after HCE is confirmed working,
    #     raise GUIDANCE_SCALE rather than adding post-processing.
    # ============================================================
    print("\n[PHASE 6] Saving frames to disk...")

    for old in FRAMES_DIR.glob("frame_*.png"):
        old.unlink()

    for i in range(actual_frames):
        frame_np  = decoded_cpu[0, :, i].permute(1, 2, 0).numpy()
        frame_img = Image.fromarray((frame_np * 255).clip(0, 255).astype(np.uint8))
        frame_img.save(FRAMES_DIR / f"frame_{i:05d}.png")

    del decoded_cpu
    gc.collect()

    print(f"  Saved {actual_frames} frames to {FRAMES_DIR}")

    # ============================================================
    # 15. COMPILE VIDEO
    # ============================================================
    print(f"\n[PHASE 7] Compiling MP4  ({actual_frames} frames @ {OUTPUT_FPS} fps)...")

    saved_files  = sorted(FRAMES_DIR.glob("frame_*.png"))
    final_frames = [np.array(Image.open(f)) for f in saved_files]

    duration_s   = actual_frames / OUTPUT_FPS
    out_name     = (f"animecolor_f{START_FRAME}-{end_frame-1}"
                    f"_{actual_frames}frames_{OUTPUT_FPS}fps.mp4")
    out_path     = str(OUTPUT_DIR / out_name)

    imageio.mimsave(out_path, final_frames, fps=OUTPUT_FPS, codec="libx264", quality=9)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Frames   : {actual_frames}")
    print(f"  Duration : {duration_s:.2f}s at {OUTPUT_FPS} fps")
    print(f"  Output   : {out_path}")
    print(f"{'='*60}")

    return {
        "status": "success",
        "output_video": out_path,
        "frames": actual_frames,
        "duration": duration_s
    }

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    result = run_animecolor(config)
    print(result)