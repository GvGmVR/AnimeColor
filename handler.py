import runpod
import os
import uuid
import json
from run_animecolor import run_animecolor

def handler(job):
    try:
        job_input = job["input"]

        # Unique job folder
        job_id = str(uuid.uuid4())
        base_path = "/workspace"

        input_dir = f"{base_path}/inputs/{job_id}"
        output_dir = f"{base_path}/outputs/{job_id}"

        os.makedirs(input_dir + "/lineart", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # =============================
        # INPUT HANDLING
        # =============================

        # Expecting:
        # job_input = {
        #   "lineart_frames": [base64 or URLs],
        #   "ref_image": base64 or URL
        # }

        # NOTE: For now assume files already mounted or simple paths
        # Later we can upgrade to base64 / URL download

        config = {
            "base_path": base_path,
            "src_dir": "AnimeColor_Code",
            "ckpt_dir": "pretrained_weights/animecolor-weights",
            "base_model_dir": "pretrained_weights/cogvideox-fun-base",
            "radio_dir": "pretrained_weights/radio-model",

            "lineart_dir": job_input["lineart_dir"],  # temporary shortcut
            "ref_image": job_input["ref_image"],
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
        # RUN MODEL
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

# Start worker
runpod.serverless.start({"handler": handler})