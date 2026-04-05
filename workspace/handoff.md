# AnimeColor Worker — Developer Handoff Document (Final)

---

# 📌 Overview

This module is a **GPU-based AI worker system** that converts:

👉 **Lineart frames + reference image → fully colored anime video**

It is designed as a **self-contained compute service** that:

1. Accepts structured input
2. Prepares data
3. Runs deep learning inference on GPU
4. Outputs a video + metadata

---

# 🧠 System Architecture (IMPORTANT)

```text
Frontend (User Uploads)
        ↓
Backend API (YOU)
        ↓
Prepare Input JSON
        ↓
Docker Worker (THIS MODULE)
        ↓
AI Model Inference (GPU)
        ↓
Outputs (Video + Metadata)
        ↓
Backend stores + returns result
```

---

# ✅ What is Already DONE (AI + Infra)

## ✔ AI Pipeline

* Lineart → Anime colorization
* Reference-guided coloring
* Multi-frame video generation
* Stable GPU execution (RTX 4070 tested)

---

## ✔ Worker System

* `worker.py` → entry point
* `handler.py` → core pipeline logic
* Automatic job folder creation (UUID-based)
* Input parsing + validation
* Output generation

---

## ✔ Dockerized Environment

* CUDA + GPU support configured
* PyTorch + dependencies aligned
* Fully runnable container
* Cross-platform (Windows → Docker → Linux handled)

---

## ✔ Path Handling (IMPORTANT FIX DONE)

* Windows paths automatically converted to container paths
* Works locally + Docker + production

---

# 📂 Project Structure

```
workspace/
│
├── worker.py              # Entry point (RUNS THE SYSTEM)
├── handler.py             # Core inference logic
├── run_animecolor.py      # Model execution
│
├── test_input.json        # Local test input
├── input_schema.json      # Contract
├── output_schema.json     # Contract
│
├── inputs/                # Mounted / uploaded data
│   ├── lineart/
│   └── ref.png
│
├── outputs/               # Generated results
│   └── <job_id>/
│
├── AnimeColor_Code/       # Model source
├── pretrained_weights/    # (NOT in Docker)
│
├── Dockerfile
├── start.sh
```

---

# 🧾 Input Contract (STRICT)

```json
{
  "input": {
    "lineart_dir": "/workspace/inputs/lineart",
    "ref_image": "/workspace/inputs/ref.png",
    "start_frame": 0,
    "num_frames": 49
  }
}
```

---

## 🔹 Field Details

| Field       | Type   | Description                |
| ----------- | ------ | -------------------------- |
| lineart_dir | string | Folder of `.png` frames    |
| ref_image   | string | Reference style image      |
| start_frame | int    | Start index                |
| num_frames  | int    | Number of frames (0 = all) |

---

# 📤 Output Contract

```json
{
  "status": "success",
  "job_id": "uuid",
  "output": {
    "output_video": "outputs/<job_id>/video.mp4",
    "frames": 49,
    "duration": 2.04
  }
}
```

---

## ❌ Error Format

```json
{
  "status": "error",
  "message": "error description"
}
```

---

# 🧪 Local Testing (FOR DEV)

## Step 1: Prepare Input

Place files:

```
inputs/
 ├── lineart/
 └── ref.png
```

---

## Step 2: Run

```bash
python worker.py
```

---

## Step 3: Output

```
outputs/<job_id>/
```

---

# 🐳 Docker Workflow (CRITICAL)

## 🔧 Build Image

```bash
docker build -t animecolor-worker .
```

---

## ▶ Run Locally (FULL PROJECT MOUNT)

```bash
docker run --gpus all -it ^
-v D:/Ixnel/dev/AnimeColor/workspace:/workspace ^
animecolor-worker
```

---

## 🔍 What this does

* Mounts your project into container
* Allows access to inputs/outputs
* Runs worker inside GPU environment

---

# 🚀 Production Execution

## Run Worker (Background)

```bash
docker run --gpus all -d animecolor-worker
```

---

## 🔍 What happens

* Container starts
* Worker waits for jobs
* Runs inference when triggered

---

# ⚠️ VERY IMPORTANT (WEIGHTS HANDLING)

### ❌ Not inside Docker:

```
pretrained_weights/
```

### ✔ Required externally:

* Must be mounted OR downloaded at runtime

---

## ✔ Recommended (Production)

Option A:

* Download weights inside container on startup

Option B:

* Mount weights:

```bash
-v /weights:/workspace/pretrained_weights
```

---

# 🔌 Backend Responsibilities (YOU)

## 1. Accept User Input

* Upload:

  * Lineart frames
  * Reference image

---

## 2. Store Files

Options:

* Local disk
* S3 / Cloud storage

---

## 3. Prepare Job JSON

```json
{
  "input": {
    "lineart_dir": "/workspace/inputs/<job_id>/lineart",
    "ref_image": "/workspace/inputs/<job_id>/ref.png"
  }
}
```

---

## 4. Trigger Worker

Options:

* Docker run
* Subprocess call
* Queue system (recommended later)

---

## 5. Handle Output

* Read returned JSON
* Store:

  * video path
  * metadata
* Send result to frontend

---

# 🔁 Internal Worker Flow

```text
handler.py

1. Receive job input
2. Normalize paths
3. Copy files into job folder
4. Build config
5. Run model
6. Save outputs
7. Return result
```

---

# ⚙️ GPU Requirements

## Minimum:

* NVIDIA GPU (8GB VRAM)

---

## Performance:

* ~49 frames → 3–5 minutes

---

## Notes:

* Uses mixed precision (bfloat16)
* Memory optimized (no xformers needed)
* Single job per GPU recommended

---

# ⚠️ Critical Rules

✔ Input frames must:

* Be `.png`
* Be correctly ordered

✔ Paths must:

* Be valid inside container (`/workspace/...`)

✔ GPU must:

* Be available (`torch.cuda.is_available()`)

---

# 🚀 Deployment Strategy (IMPORTANT)

## Phase 1 (Now)

✔ Local Docker testing
✔ Manual inputs

---

## Phase 2

* Integrate backend API
* Automate job triggering

---

## Phase 3

* Deploy to RunPod / cloud GPU
* Add queue system

---

## Phase 4

* Scale (multiple workers)
* Add caching / batching

---

# ❗ What This Module DOES NOT Handle

* API endpoints
* Authentication
* Database
* File uploads
* UI

👉 This is **only the AI execution engine**

---

# 🧠 Final Summary

| Component     | Responsibility   |
| ------------- | ---------------- |
| This Worker   | AI processing    |
| Backend (YOU) | orchestration    |
| Frontend      | user interaction |

---

# ✅ Current Status

✔ Fully working locally
✔ Docker working with GPU
✔ Cross-platform compatibility fixed
✔ Ready for backend integration

---

# 🚀 What You Should Do Next

1. Build API layer
2. Connect uploads → worker
3. Store outputs
4. Return video to UI

---

# 🤝 If Issues Occur

Check:

* Paths (`/workspace/...`)
* Input format
* GPU availability
* Docker mount correctness

---

**This system is now production-ready from AI side.**
