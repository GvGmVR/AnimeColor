# AnimeColor Worker — Developer Handoff Document

## 📌 Overview

This module is a **GPU-based AI worker** that converts:

* **Lineart frames + reference image → colored anime video**

It is designed as a **self-contained compute unit** that:

1. Accepts structured input
2. Runs model inference
3. Produces a video output
4. Returns metadata

---

## ✅ What is Already Implemented (AI Side)

### ✔ Core Capabilities

* Lineart → Anime colorization pipeline
* Multi-frame processing (video generation)
* Reference-guided style transfer
* GPU-accelerated inference (tested on RTX 4070)

---

### ✔ Worker System

* `worker.py` → Main execution entrypoint
* `handler.py` → Core inference logic
* Input parsing + frame loading
* Output generation (video + metadata)

---

### ✔ Input Handling

* Accepts:

  * directory of lineart frames
  * reference image
  * frame range controls

---

### ✔ Output

* MP4 video file
* Metadata (frames, duration)

---

## 📂 Project Structure (Relevant)

```
workspace/
│
├── worker.py              # Entry point (YOU will trigger this)
├── handler.py             # AI pipeline
├── test_input.json        # Sample input format
├── input_schema.json      # Input contract
├── output_schema.json     # Output contract
│
├── inputs/
│   ├── lineart/           # Input frames
│   └── ref.png            # Reference image
│
├── outputs/
│   └── <job_id>/          # Generated results
```

---

## 🧾 Input Contract (IMPORTANT)

### Expected JSON format:

```json
{
  "input": {
    "lineart_dir": "path/to/lineart_frames/",
    "ref_image": "path/to/reference.png",
    "start_frame": 0,
    "num_frames": 49
  }
}
```

---

### 🔹 Field Details

| Field         | Type   | Description                           |
| ------------- | ------ | ------------------------------------- |
| `lineart_dir` | string | Directory containing `.png` frames    |
| `ref_image`   | string | Style reference image                 |
| `start_frame` | int    | Starting frame index                  |
| `num_frames`  | int    | Number of frames to process (0 = all) |

---

## 📤 Output Contract

```json
{
  "status": "success",
  "job_id": "uuid",
  "output": {
    "output_video": "path/to/video.mp4",
    "frames": 49,
    "duration": 2.04
  }
}
```

---

### 🔹 Error Case

```json
{
  "status": "error",
  "message": "error description"
}
```

---

## 🔁 Execution Flow

```text
User (UI)
   ↓
Backend API (YOU)
   ↓
Prepare input JSON
   ↓
Run worker (this module)
   ↓
Model inference (GPU)
   ↓
Output video + metadata
   ↓
Store result + return to UI
```

---

## 🧪 Local Testing Instructions

### 1. Prepare Input

Edit `test_input.json`:

```json
{
  "input": {
    "lineart_dir": "inputs/lineart",
    "ref_image": "inputs/ref.png",
    "start_frame": 0,
    "num_frames": 49
  }
}
```

---

### 2. Run Worker

```bash
python worker.py
```

---

### 3. Output

* Video saved in:

```
outputs/<job_id>/
```

---

## 🐳 (Optional) Docker Execution

### Build:

```bash
docker build -t anime-worker .
```

### Run:

```bash
docker run --gpus all -v %cd%:/app anime-worker
```

---

## 🔌 Backend Integration (Your Responsibility)

You need to:

### 1. Accept user input from UI

* Upload lineart frames
* Upload reference image

---

### 2. Store inputs

* Save to local disk or object storage (S3, etc.)

---

### 3. Create job payload

```json
{
  "input": {
    "lineart_dir": "...",
    "ref_image": "...",
    "start_frame": 0,
    "num_frames": 49
  }
}
```

---

### 4. Trigger worker

Options:

* Run as subprocess
* Call container
* Use queue (future scaling)

---

### 5. Handle output

* Read returned JSON
* Store video path
* Save metadata in DB
* Return result to frontend

---

## ❗ What is NOT handled here

The following are **NOT part of this module**:

* ❌ API endpoints
* ❌ Authentication
* ❌ Database storage
* ❌ File upload handling
* ❌ UI rendering

---

## ⚙️ GPU Requirements

### Minimum:

* NVIDIA GPU (tested: RTX 4070, 8GB VRAM)

---

### Performance:

* ~49 frames → ~3–5 minutes
* Depends on:

  * resolution
  * number of frames
  * GPU

---

### Notes:

* Uses mixed precision (bfloat16)
* VRAM is freed between stages
* Avoid running multiple jobs on same GPU

---

## ⚠️ Important Notes

* Input frames must be:

  * `.png`
  * ordered correctly
* Reference image must exist
* Paths must be valid on the system running worker

---

## 🚀 Future Improvements (Optional)

* Queue-based job system
* API wrapper (FastAPI)
* ONNX optimization
* Batch processing
* Cloud GPU deployment

---

## 🧠 Summary

* This module = **AI execution engine**
* You (backend dev) = **system + integration layer**

---

## 🤝 Contact / Clarification

If integration issues occur:

* Check input format
* Verify paths
* Ensure GPU availability

---
