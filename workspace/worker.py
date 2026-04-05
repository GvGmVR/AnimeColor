import json
import os
from handler import handler


def load_job():
    """
    Loads job from test_input.json (for local testing)
    In production, this will be replaced by API/queue input.
    """
    with open("test_input.json", "r") as f:
        return json.load(f)


def prepare_input(job):
    """
    Converts lineart_dir → list of frame paths
    """
    input_data = job.get("input", {})

    lineart_dir = input_data.get("lineart_dir")
    ref_image = input_data.get("ref_image")
    start_frame = input_data.get("start_frame", 0)
    num_frames = input_data.get("num_frames", 0)

    if not os.path.exists(lineart_dir):
        raise ValueError(f"lineart_dir not found: {lineart_dir}")

    frames = sorted([
        os.path.join(lineart_dir, f)
        for f in os.listdir(lineart_dir)
        if f.endswith(".png")
    ])

    if num_frames > 0:
        frames = frames[start_frame:start_frame + num_frames]

    return {
        "lineart_frames": frames,
        "ref_image": ref_image,
        "start_frame": start_frame,
        "num_frames": len(frames)
    }


def main():
    try:
        print("[WORKER] Loading job...")
        job = load_job()

        print("[WORKER] Preparing input...")
        prepared_input = prepare_input(job)

        print(f"[WORKER] Running inference on {len(prepared_input['lineart_frames'])} frames...")

        result = handler({
            "input": prepared_input
        })

        print("[WORKER] Completed successfully.")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print("[WORKER] ERROR:", str(e))
        result = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()