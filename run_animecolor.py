# run_animecolor.py
import os, sys, gc, torch, decord, imageio, warnings, time
import numpy as np
from PIL import Image
from pathlib import Path
from accelerate.hooks import remove_hook_from_module

# 1. MUTE WARNINGS & MEMORY FRAGMENTATION FIX
warnings.filterwarnings("ignore")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 2. DEFINE LOCAL PATHS (Update to match your D: drive)
BASE_PATH = Path(r"D:\Ixnel\dev\AnimeColor")
SRC_DIR   = BASE_PATH / "AnimeColor_Code"  

CKPT_DIR  = BASE_PATH / "pretrained_weights/animecolor-weights"
BASE_DIR  = BASE_PATH / "pretrained_weights/cogvideox-fun-base"
RADIO_DIR = BASE_PATH / "pretrained_weights/radio-model"

LINEART_DIR = BASE_PATH / "inputs/lineart"
REF_IMAGE   = BASE_PATH / "inputs/ref.png"
OUTPUT_DIR  = BASE_PATH / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 3. SELF-HEALING SYSTEM
if not SRC_DIR.exists(): raise FileNotFoundError("Missing source code!")
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

import importlib.util
diffusers_spec = importlib.util.find_spec("diffusers")
if diffusers_spec and diffusers_spec.origin:
    dep_path = Path(diffusers_spec.origin).parent / "utils" / "deprecation_utils.py"
    if dep_path.exists():
        content = dep_path.read_text(encoding="utf-8")
        if "if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):" in content:
            content = content.replace("if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):", "if False:")
            dep_path.write_text(content, encoding="utf-8")

pipe_file = SRC_DIR / "cogvideox/pipeline/pipeline_cogvideo_color_ref.py"
if pipe_file.exists():
    content = pipe_file.read_text(encoding="utf-8")
    if "video = torch.from_numpy(video)" in content:
        content = content.replace("video = torch.from_numpy(video)", "video = torch.as_tensor(video)")
        pipe_file.write_text(content, encoding="utf-8")

import re
for t_file in ["transformer3d_radio.py", "transformer3d.py"]:
    tf_path = SRC_DIR / f"cogvideox/models/{t_file}"
    if tf_path.exists():
        content = tf_path.read_text(encoding="utf-8")
        new_content = re.sub(r'torch\.from_numpy\((\w+_pos_embedding)\)', r'torch.as_tensor(\1)', content)
        if new_content != content: tf_path.write_text(new_content, encoding="utf-8")

for py_file in RADIO_DIR.rglob("*.py"):
    content = py_file.read_text(encoding="utf-8")
    if "class RADIOModel" in content and "all_tied_weights_keys" not in content:
        content = content.replace("def forward(self", "@property\n    def all_tied_weights_keys(self):\n        return {}\n\n    def forward(self")
        py_file.write_text(content, encoding="utf-8")

# ============================================================
# NEW SETTINGS: BFLOAT16 (Fixes Color) & CHUNKING (Fixes Length)
# ============================================================
WIDTH, HEIGHT = 512, 320
CHUNK_SIZE = 7  # Process 7 frames per GPU cycle
REST_DELAY = 15 # Seconds to let GPU cool down between chunks

# CRITICAL FIX: RTX 4070 supports Brain-Float 16 natively. This restores full color vitality!
weight_dtype = torch.bfloat16 

print("\n[PHASE 1] Loading Models in BFLOAT16 into 32GB System RAM...")
from transformers import T5EncoderModel, T5Tokenizer, AutoModel, CLIPImageProcessor
from diffusers import CogVideoXDDIMScheduler
from cogvideox.models.transformer3d_radio import CogVideoXTransformer3DModel
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel as Ref3D
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideo_color_ref import CogVideoX_Fun_Pipeline_Control_Color

text_encoder = T5EncoderModel.from_pretrained(BASE_DIR, subfolder="text_encoder", torch_dtype=weight_dtype)
tokenizer = T5Tokenizer.from_pretrained(BASE_DIR, subfolder="tokenizer")
vae = AutoencoderKLCogVideoX.from_pretrained(BASE_DIR, subfolder="vae", torch_dtype=weight_dtype)

denoising_transformer = CogVideoXTransformer3DModel.from_pretrained(CKPT_DIR, subfolder="transformer", torch_dtype=weight_dtype)
denoising_transformer.is_train_qformer = False 
reference_transformer = Ref3D.from_pretrained(CKPT_DIR, subfolder="referencenet", torch_dtype=weight_dtype)

dclip_model = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True, torch_dtype=weight_dtype)
dclip_processor = CLIPImageProcessor.from_pretrained(RADIO_DIR, torch_dtype=weight_dtype)
scheduler = CogVideoXDDIMScheduler.from_pretrained(BASE_DIR, subfolder="scheduler")

pipe = CogVideoX_Fun_Pipeline_Control_Color(
    tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
    denoising_transformer=denoising_transformer, reference_transformer=reference_transformer, scheduler=scheduler,
)
pipe.dclip_model = dclip_model
pipe.dclip_processor = dclip_processor

pipe.enable_sequential_cpu_offload() 
try: pipe.enable_attention_slicing(slice_size="max")
except: pass
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

torch.cuda.empty_cache()
gc.collect()

# ============================================================
# PREPARE IMAGES & BATCHING
# ============================================================
# Use bfloat16 for tensors so color data doesn't clip/ghost!
ref_np = np.array(Image.open(REF_IMAGE).convert("RGB").resize((WIDTH, HEIGHT)))
ref_tensor = torch.from_numpy(ref_np).permute(2,0,1).to(weight_dtype) / 255.0
ref_tensor = ref_tensor.unsqueeze(0)

# Read ALL frames in the lineart folder
all_frame_paths = sorted([p for p in LINEART_DIR.iterdir() if p.suffix.lower() in (".png",".jpg")])
if not all_frame_paths: raise ValueError("No frames found in inputs/lineart/")

all_sketch_frames =[np.array(Image.open(f).convert("RGB").resize((WIDTH, HEIGHT))) for f in all_frame_paths]

# Split frames into chunks of 7
chunks =[all_sketch_frames[i:i + CHUNK_SIZE] for i in range(0, len(all_sketch_frames), CHUNK_SIZE)]
total_frames = len(all_sketch_frames)

print(f"\n[PHASE 2] Loaded {total_frames} frames. Splitting into {len(chunks)} chunks.")

# ============================================================
# CHUNKED INFERENCE LOOP
# ============================================================
all_generated_latents =[]

for idx, chunk in enumerate(chunks):
    actual_length = len(chunk)
    
    # --- CRITICAL 3D VAE CRASH FIX (Pad and Trim) ---
    # If the last chunk has too few frames, the 3D VAE temporal convolutions collapse to 0.
    # We pad the chunk with duplicates of the last frame to hit the safe CHUNK_SIZE.
    if actual_length < CHUNK_SIZE:
        pad_needed = CHUNK_SIZE - actual_length
        chunk = chunk + [chunk[-1]] * pad_needed
        print(f"\n>>> Processing Chunk {idx+1}/{len(chunks)} (Padded from {actual_length} to {CHUNK_SIZE} frames to prevent 3D VAE crash) <<<")
    else:
        print(f"\n>>> Processing Chunk {idx+1}/{len(chunks)} ({CHUNK_SIZE} frames) <<<")
    # ------------------------------------------------
    
    ctrl_tensor = torch.from_numpy(np.array(chunk)).permute(3,0,1,2).to(weight_dtype) / 255.0
    ctrl_tensor = ctrl_tensor.unsqueeze(0)

    # Force RADIO to GPU just for this cycle
    pipe.dclip_model.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(42)

    with torch.inference_mode():
        output = pipe(
            prompt="anime style, high quality masterpiece", 
            negative_prompt="low quality, distortion",
            ref_image=ref_tensor, control_video=ctrl_tensor,
            height=HEIGHT, width=WIDTH, num_frames=CHUNK_SIZE, # Force to CHUNK_SIZE
            guidance_scale=6.0, num_inference_steps=25, generator=generator,
            output_type="latent",  
            return_dict=True, id_cond=None, id_vit_hidden=None,
        )

    # Save to CPU RAM immediately
    all_generated_latents.append(output.videos.cpu())
    
    # Aggressively flush GPU
    pipe.dclip_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    
    # The Resting Delay (unless it's the last chunk)
    if idx < len(chunks) - 1:
        print(f"Chunk complete. Resting system for {REST_DELAY} seconds...")
        time.sleep(REST_DELAY)

# Stitch all the chunks together seamlessly!
print("\n[PHASE 3] Stitching chunks together...")
final_latents = torch.cat(all_generated_latents, dim=1) # Concat along the 'Frames' dimension

final_latents = torch.cat(all_generated_latents, dim=1)
print(f"[DEBUG] Stitched latent shape: {final_latents.shape}")  # Expect [1, total_F_lat, C, H, W]

# ============================================================
# PURGE AND DECODE
# ============================================================
print("\n[PHASE 4] Purging GPU for Final Decode...")
for model in[pipe.denoising_transformer, pipe.reference_transformer, pipe.text_encoder, pipe.vae]:
    if hasattr(pipe, "dclip_model"): pipe.dclip_model.to("cpu")
    remove_hook_from_module(model, recurse=True)
    model.to("cpu")

torch.cuda.empty_cache()
gc.collect()

print(f"\n[PHASE 5] Decoding FULL {total_frames}-Frame Video on CPU RAM...")
# Then permute to [B, C, F, H, W] for the VAE
scaling_factor = pipe.vae.config.scaling_factor
full_latent = (1 / scaling_factor) * final_latents.permute(0, 2, 1, 3, 4).contiguous()

# Decode on CPU in float32 for maximum safety
pipe.vae = pipe.vae.to(torch.float32).cpu()

with torch.inference_mode():
    decoded = pipe.vae.decode(full_latent.to(torch.float32)).sample
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)

decoded_frames = []
for i in range(decoded.shape[2]):
    frame_np = decoded[0, :, i].permute(1, 2, 0).numpy()
    decoded_frames.append((frame_np * 255).clip(0, 255).astype(np.uint8))

# --- REMOVE PADDED FRAMES BEFORE SAVING ---
# We slice the list back down to the exact number of original input sketches
decoded_frames = decoded_frames[:total_frames]
print(f"Successfully decoded and trimmed to exact original length: {len(decoded_frames)} frames!")
# ------------------------------------------

out_path = str(OUTPUT_DIR / f"long_animecolor_output_{total_frames}frames.mp4")
imageio.mimsave(out_path, decoded_frames, fps=8, codec="libx264", quality=8)
print(f"\n✅ Success! Vibrant, full-length video saved to: {out_path}")