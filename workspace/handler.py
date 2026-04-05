import runpod
import os
import uuid
import shutil
from run_animecolor import run_animecolor

def normalize_path(path):
    if os.name != "nt":
        path = path.replace("\\", "/")

        if "/workspace/" in path:
            return path  # already correct

        if "/inputs/" in path:
            return "/workspace/inputs/" + path.split("/inputs/")[-1]

    return path

def handler(job):
    try:
        job_input = job["input"]
        
        print("JOB INPUT:", job_input)

        # =============================
        # 1. CREATE JOB FOLDERS
        # =============================
        job_id = str(uuid.uuid4())
        base_path = os.environ.get("BASE_PATH", os.getcwd())

        input_dir = f"{base_path}/inputs/{job_id}"
        lineart_dir = f"{input_dir}/lineart"
        output_dir = f"{base_path}/outputs/{job_id}"

        os.makedirs(lineart_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # =============================
        # 2. HANDLE INPUT FILES
        # =============================

        # Expecting:
        # job_input = {
        #   "lineart_paths": [list of file paths (for now)],
        #   "ref_image": file path
        # }

        ref_image_path = normalize_path(job_input["ref_image"])

        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Ref image not found: {ref_image_path}")

        # =============================
        # SUPPORT BOTH MODES
        # =============================

        if "lineart_frames" in job_input:
            # Mode 1: explicit file list
            lineart_paths =[normalize_path(p) for p in job_input["lineart_frames"]]

        elif "lineart_dir" in job_input:
            src_dir = normalize_path(job_input["lineart_dir"])

            print(f"[DEBUG] Using lineart_dir: {src_dir}")

            if not os.path.exists(src_dir):
                raise FileNotFoundError(f"Lineart dir not found: {src_dir}")

            lineart_paths = sorted([
                os.path.join(src_dir, f)
                for f in os.listdir(src_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if len(lineart_paths) == 0:
                raise ValueError(f"No images found in {src_dir}")
        else:
            raise ValueError("Provide either 'lineart_frames' or 'lineart_dir'")

        # Copy frames
        for i, src_path in enumerate(lineart_paths):
            dst_path = os.path.join(lineart_dir, f"frame_{i:05d}.png")
            shutil.copy(src_path, dst_path)

        # Copy reference image
        ref_dst = os.path.join(input_dir, "ref.png")
        shutil.copy(ref_image_path, ref_dst)
        
        print("FILES FOUND:", len(lineart_paths))
        print("FIRST 3 FILES:", lineart_paths[:3])

        # =============================
        # 3. BUILD CONFIG FOR MODEL
        # =============================
        config = {
            "base_path": base_path,
            "src_dir": "AnimeColor_Code",
            "ckpt_dir": "pretrained_weights/animecolor-weights",
            "base_model_dir": "pretrained_weights/cogvideox-fun-base",
            "radio_dir": "pretrained_weights/radio-model",

            "lineart_dir": lineart_dir,
            "ref_image": ref_dst,
            "output_dir": output_dir,

            "start_frame": job_input.get("start_frame", 0),
            "num_frames": job_input.get("num_frames", 49),
            "width": job_input.get("width", 512),
            "height": job_input.get("height", 320),
            "output_fps": job_input.get("output_fps", 24),
            "guidance_scale": job_input.get("guidance_scale", 8.5),
            "inference_steps": job_input.get("inference_steps", 25)
        }

        # =============================
        # 4. RUN MODEL
        # =============================
        result = run_animecolor(config)

        return {
            "status": "success",
            "job_id": job_id,
            "output": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# =============================
# START RUNPOD WORKER
# =============================
runpod.serverless.start({"handler": handler})