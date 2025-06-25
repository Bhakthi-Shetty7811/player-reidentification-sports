# ASSIGNMENT : Player Re-Identification in Sports Footage
# 🎥 Option 1: Cross-Camera Player Mapping

This project implements a robust cross-camera player mapping system using deep learning and computer vision. It ensures that each player retains a **consistent ID across two different camera views** — namely, `broadcast.mp4` and `tacticam.mp4`.


## 📌 Task Overview

- **Objective:** Given two clips of the same gameplay from different camera angles, identify and assign a **consistent ID** to each player across both feeds.
- **Approach:** Use YOLOv11 for player detection, OSNet for extracting visual appearance embeddings, and DeepSORT for tracking players. Match players between videos using cosine similarity + Hungarian algorithm.


## 🧠 Pipeline Overview

Broadcast & Tacticam Video → YOLOv11 Detection → OSNet Embedding → DeepSORT Tracking → Cross-Video Player Mapping using Embedding Similarity → Output Videos with Consistent IDs

### ✔ Components:

| Module             | Function                                                         |
| ------------------ | ---------------------------------------------------------------- |
| `detect.py`        | Extracts detections + re-ID embeddings (pickled)                 |
| `match_players.py` | Matches player identities across views using Hungarian algorithm |
| `run_visualise.py` | Generates video with bounding boxes + consistent player IDs      |
| `run_detection.py` | Triggers detection on both broadcast and tacticam videos         |


## 📁 Directory Structure

option1-cross-camera/
├── run_detection.py
├── match_players.py
├── run_visualise.py
├── src/
│ └── detect.py
├── model/
│ └── yolov11.pt
├── data/
│ └── broadcast.mp4
│ └── tacticam.mp4
├── outputs/
│ └── broadcast_data.pkl
│ └── tacticam_data.pkl
│ └── player_id_mapping.pkl
│ └── broadcast_labeled.mp4
│ └── tacticam_labeled.mp4
├── README.md
├── report.md
└── requirements.txt


## ⚙️ Setup Instructions

### 1️⃣ Clone and Prepare Environment

git clone <repository-link>
cd option1-cross-camera

2️⃣ Install Dependencies

pip install -r requirements.txt

ℹ️ If using Anaconda:

conda create -n player-reid python=3.9
conda activate player-reid
pip install -r requirements.txt

3️⃣ Torchreid Compatibility Fix (if needed)

pip uninstall torchreid
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

▶️ Run the Project
Step-by-step Execution:

python run_detection.py       # Extract embeddings from both videos
python match_players.py       # Match player identities across views
python run_visualise.py       # Generate final labeled videos
✅ Final videos will be saved in the outputs/ folder.

📦 Key Dependencies
* ultralytics - YOLOv11
* deep_sort_realtime - Multi-object tracking
* torchreid - Person re-ID
* OpenCV, PyTorch, NumPy, SciPy

💡 Highlights & Innovations
- Extracted appearance embeddings using pretrained OSNet model.
- Used DeepSORT with embedding fusion to maintain ID consistency across frames.
- Performed cross-view mapping using cosine distance + Hungarian algorithm.
- Included center-distance filtering for robust association of tracks and detections.

🧠 Author
Bhakthi Shetty
Final-Year B.Tech (IT), UMIT SNDT