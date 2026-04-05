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
    GUIDANCE_SCALE: 7.5 default; raise to 8–9 for stronger ref adherence
    INFER_REST    : GPU cooldown after inference (seconds)
    DECODE_REST   : GPU cooldown after decode (seconds)
    """

    import os, sys, gc, torch, imageio, warnings, time, re
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from accelerate.hooks import remove_hook_from_module
    import importlib.util

    from PIL import ImageEnhance, ImageFilter

    def post_process_frame(img: Image.Image) -> Image.Image:
        # Order matters: brighten/saturate first, sharpen last
        # (sharpening after colour boost enhances the right edges)
        img = ImageEnhance.Brightness(img).enhance(1.25)
        img = ImageEnhance.Color(img).enhance(1.70)
        img = ImageEnhance.Contrast(img).enhance(1.30)
        # Unsharp mask: recovers the crisp anime line quality lost in diffusion decode
        # radius=1, percent=200, threshold=2 hits sharpness ~590 vs ref ~1537
        # raise percent to 250 for even crisper lines (risk: halos on fine detail)
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=2))
        return img

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
    START_FRAME      = config.get("start_frame", 0) # Change to process a different 49-frame window
    NUM_FRAMES       = config.get("num_frames", 49) # Native CogVideoX-2B context — do not exceed this
    WIDTH            = config.get("width", 512)
    HEIGHT           = config.get("height", 320)
    OUTPUT_FPS       = config.get("output_fps", 24) # 24 = anime standard  |  12 = slower/smoother feel
    GUIDANCE_SCALE   = config.get("guidance_scale", 7.5) # raise to 8-9 for stronger reference adherence
    INFERENCE_STEPS  = config.get("inference_steps", 25) # 25 is the paper default; 30 for slightly better quality
    INFER_REST       = config.get("infer_rest", 10) # seconds to rest GPU after inference
    DECODE_REST      = config.get("decode_rest", 5) # seconds to rest GPU after decode
    SEED             = config.get("seed", 42)

    weight_dtype = torch.bfloat16   # RTX 4070 native bf16

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
    #
    #   After inference the transformer weights are back on CPU via
    #   sequential offload.  The VAE is the only model needed for
    #   decode, so it can have the full 12 GB VRAM to itself.
    #   A single 49-frame decode in bf16 uses ~4-5 GB — well within
    #   the RTX 4070's budget after inference models offload.
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
    #    Binarise lineart before passing it to the model.
    #    The paper uses XDoG-extracted sketches which are clean
    #    binary black-on-white.  Noisy or anti-aliased lines cause
    #    the model to ghost at edge boundaries.
    # ============================================================
    def clean_sketch(img: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Convert sketch to clean binary black lines on white background.
        Threshold 128 works for most XDoG/Anime2Sketch outputs.
        Lower the threshold (e.g. 100) to keep finer lines.
        """
        gray = img.convert("L")
        # Binarise: pixels darker than threshold -> black (0), rest -> white (255)
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

    # The model requires num_frames to be of the form 4k+1 (1,5,9,...,49)
    # 49 = 4*12+1 ✓   If we have fewer, round down to nearest valid value
    def nearest_valid_frame_count(n: int) -> int:
        """CogVideoX requires frames = 4k+1. Find largest valid count <= n."""
        if n <= 0: return 1
        k = (n - 1) // 4
        return 4 * k + 1

    padded_frames = nearest_valid_frame_count(actual_frames)
    pad_needed    = padded_frames - actual_frames   # might be 0

    print(f"\n[CONFIG] Processing frames {START_FRAME}–{end_frame-1} "
        f"({actual_frames} real + {pad_needed} pad = {padded_frames} total)")
    print(f"         Resolution: {WIDTH}×{HEIGHT}  |  FPS: {OUTPUT_FPS}  |  "
        f"Duration: {actual_frames/OUTPUT_FPS:.2f}s")

    # ============================================================
    # 8. LOAD MODELS
    # ============================================================
    print("\n[PHASE 1] Loading models into system RAM (bfloat16)...")

    from transformers import T5EncoderModel, T5Tokenizer, AutoModel, CLIPImageProcessor
    from diffusers import CogVideoXDDIMScheduler
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
    dclip_model = AutoModel.from_pretrained(
        RADIO_DIR, trust_remote_code=True, torch_dtype=weight_dtype)
    dclip_processor = CLIPImageProcessor.from_pretrained(
        RADIO_DIR, torch_dtype=weight_dtype)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        BASE_DIR, subfolder="scheduler")

    denoising_transformer.is_train_qformer = False

    pipe = CogVideoX_Fun_Pipeline_Control_Color(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        denoising_transformer=denoising_transformer,
        reference_transformer=reference_transformer,
        scheduler=scheduler,
    )
    pipe.dclip_model     = dclip_model
    pipe.dclip_processor = dclip_processor

    # enable_sequential_cpu_offload installs accelerate hooks on ALL models
    # including the VAE.  These hooks are REQUIRED during inference because
    # the pipeline calls vae.encode() internally for the control video.
    # We remove them only AFTER inference is complete (Phase 4).
    pipe.enable_sequential_cpu_offload()
    try:
        pipe.enable_attention_slicing(slice_size="max")
    except Exception:
        pass
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.cuda.empty_cache()
    gc.collect()
    print("[PHASE 1] Done.\n")

    # ============================================================
    # 9. PREPARE TENSORS
    # ============================================================
    print("[PHASE 2] Preparing input tensors...")

    # Reference image
    ref_pil    = Image.open(REF_IMAGE).convert("RGB").resize((WIDTH, HEIGHT))
    ref_np     = np.array(ref_pil)
    ref_tensor = (
        torch.from_numpy(ref_np).permute(2, 0, 1).to(weight_dtype) / 255.0
    ).unsqueeze(0)   # [1, C, H, W]

    # Sketch frames — clean, resize, load
    sketch_frames = []
    for p in selected_paths:
        img = Image.open(p).convert("RGB").resize((WIDTH, HEIGHT))
        img = clean_sketch(img)   # binarise for clean model input
        sketch_frames.append(np.array(img))

    # Pad to nearest valid frame count (repeat last frame)
    if pad_needed > 0:
        print(f"  Padding {pad_needed} frame(s) to reach valid count {padded_frames}.")
        sketch_frames += [sketch_frames[-1]] * pad_needed

    # Build control tensor: [1, C, F, H, W]
    ctrl_array  = np.array(sketch_frames)                   # [F, H, W, C]
    ctrl_tensor = (
        torch.from_numpy(ctrl_array)
        .permute(3, 0, 1, 2)                                # [C, F, H, W]
        .to(weight_dtype) / 255.0
    ).unsqueeze(0)                                          # [1, C, F, H, W]

    print(f"  ref_tensor  : {ref_tensor.shape}")
    print(f"  ctrl_tensor : {ctrl_tensor.shape}   (frames={padded_frames})")
    print("[PHASE 2] Done.\n")

    # ============================================================
    # 10. SINGLE-PASS INFERENCE
    #     All padded_frames processed together — the 3D attention
    #     across the full sequence is what gives temporal consistency.
    # ============================================================
    print(f"[PHASE 3] Single-pass inference  ({padded_frames} frames, "
        f"{INFERENCE_STEPS} steps)...")
    print("  This is the longest phase — ~15-30 min on RTX 4070 for 49 frames.")

    pipe.dclip_model.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    t0 = time.time()
    with torch.inference_mode():
        output = pipe(
            prompt=(
                "anime style, vibrant colors, high quality masterpiece, "
                "sharp linework, consistent lighting"
            ),
            negative_prompt=(
                "low quality, blurry, distortion, desaturated, dark, "
                "grayscale, inconsistent colors, flickering, ghosting"
            ),
            ref_image=ref_tensor,
            control_video=ctrl_tensor,
            height=HEIGHT,
            width=WIDTH,
            num_frames=padded_frames,       # full sequence in one pass
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=INFERENCE_STEPS,
            generator=generator,
            output_type="latent",
            return_dict=True,
            id_cond=None,
            id_vit_hidden=None,
        )

    elapsed = time.time() - t0
    latent_cpu = output.videos.cpu()   # [B, F_lat, C, H, W]  (frames-first)
    print(f"  Inference done in {elapsed/60:.1f} min. Latent: {latent_cpu.shape}")

    pipe.dclip_model.to("cpu")
    del ctrl_tensor
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Resting GPU for {INFER_REST}s before decode...")
    time.sleep(INFER_REST)

    # ============================================================
    # 11. REMOVE PIPELINE HOOKS, TAKE VAE OWNERSHIP
    #     NOW (and only now) we remove accelerate hooks so we can
    #     move the VAE freely for GPU decode.
    # ============================================================
    print("\n[PHASE 4] Removing pipeline hooks, freeing VRAM...")

    for model in [pipe.denoising_transformer, pipe.reference_transformer,
                pipe.text_encoder, pipe.vae]:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
    if hasattr(pipe, "dclip_model"):
        pipe.dclip_model.to("cpu")

    torch.cuda.empty_cache()
    gc.collect()

    free_gb = get_free_vram_gb()
    print(f"  Hooks removed. Free VRAM: {free_gb:.2f} GB")

    vae_standalone = pipe.vae
    vae_standalone.enable_slicing()
    vae_standalone.enable_tiling()
    scaling_factor = vae_standalone.config.scaling_factor

    # ============================================================
    # 12. DECODE ON GPU
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
    # 13. SAVE FRAMES
    #     Trim padding, save clean PNGs — NO colour post-processing.
    #     We intentionally skip ImageEnhance here: the model already
    #     has guidance_scale pushing it toward the reference colour;
    #     post-processing boosts noise along with colour and was a
    #     contributing factor to ghosting artefacts.
    #     If the raw output is still too dull, raise GUIDANCE_SCALE
    #     rather than adding post-processing.
    # ============================================================
    print("\n[PHASE 6] Saving frames to disk...")

    # Clean out any leftover frames from a previous run
    for old in FRAMES_DIR.glob("frame_*.png"):
        old.unlink()

    for i in range(actual_frames):   # trim padding here
        frame_np  = decoded_cpu[0, :, i].permute(1, 2, 0).numpy()
        frame_img = Image.fromarray((frame_np * 255).clip(0, 255).astype(np.uint8))
        frame_img = post_process_frame(frame_img)
        frame_img.save(FRAMES_DIR / f"frame_{i:05d}.png")

    del decoded_cpu
    gc.collect()

    print(f"  Saved {actual_frames} frames to {FRAMES_DIR}")

    # ============================================================
    # 14. COMPILE VIDEO
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