import json
import os
from handler import handler

# Auto collect frames
lineart_dir = "D:/Ixnel/dev/AnimeColor/workspace/inputs/lineart"

frames = sorted([
    os.path.join(lineart_dir, f)
    for f in os.listdir(lineart_dir)
    if f.endswith(".png")
])

job = {
    "input": {
        "lineart_frames": frames,
        "ref_image": "D:/Ixnel/dev/AnimeColor/workspace/inputs/ref.png",
        "start_frame": 0,
        "num_frames": 49
    }
}

result = handler(job)
print(json.dumps(result, indent=2))