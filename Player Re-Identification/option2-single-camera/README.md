# ASSIGNMENT : Player Re-Identification in Sports Footage
# ⚽ Option 2: Re-Identification in single feed

This project implements a robust player re-identification system using deep learning and computer vision. It focuses on **intra-camera re-identification**, ensuring that each player maintains a consistent ID across all frames of a single video.


## 📌 Task Overview

- **Objective:** Re-identify and consistently track soccer players within a broadcast or tacticam video.
- **Approach:** Combine YOLOv8 for player detection, OSNet for appearance embeddings, and DeepSORT for ID tracking.


## 🧠 Pipeline Overview

Video Input → YOLOv11 Detection → OSNet Embedding → DeepSORT Tracking → Output Video with IDs

---

### ✔ Components:

| Module      | Function                                         |
| ----------- | ------------------------------------------------ |
| `detect.py` | Extracts detections + re-ID embeddings (pickled) |
| `main.py`   | Generates video with bounding boxes + player IDs |
| `track.py`  | Initializes DeepSORT tracker                     |
| `utils.py`  | Embedding model (Torchreid) and helper functions |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone and Prepare Environment
```
git clone https://github.com/Bhakthi-Shetty7811/player-reidentification-sports
cd option2-single-camera
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

> ℹ️ If using Anaconda:
```
conda create -n player-reid python=3.9
conda activate player-reid
pip install -r requirements.txt
```

### 3️⃣ Torchreid Compatibility Fix (if needed)
```
pip uninstall torchreid
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## ▶️ Run the Project
### Generate Output Video with Re-ID
```
python main.py
> ✅ Output video will be saved as `output.mp4`.
```

---

## 📦 Key Dependencies

* [`ultralytics`](https://github.com/ultralytics/ultralytics)
* [`deep_sort_realtime`](https://github.com/levan92/deep_sort_realtime)
* [`torchreid`](https://github.com/KaiyangZhou/deep-person-reid)
* OpenCV, PyTorch, NumPy

---

## 💡 Highlights & Innovations

* Extracted **appearance embeddings** using pretrained **OSNet** model.
* Used **DeepSORT** with embedding fusion to maintain ID consistency.
* Included **center-distance filtering** and **embedding proximity** for stable matching.

---

## 🧠 Author

**Bhakthi Shetty**
Final-Year B.Tech (IT), UMIT SNDT



