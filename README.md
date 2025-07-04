# 🧠 Player Re-Identification in Sports Footage

This repository explores two deep learning pipelines for **player re-identification** in soccer videos — one across **multiple camera angles**, and the other within a **single video stream**. Both options aim to assign consistent IDs to players using object detection, tracking, and appearance-based embeddings.


## 🎯 Project Overview

| Option                  | Scenario                             | Goal                                                       |
|------------------------ |------------------------------------- |------------------------------------------------------------|
| `option1-cross-camera`  | Multiple feed(e.g., different angles)| Maintain consistent IDs across multiple views              |
| `option2-single-camera` | Single feed (e.g., match footage)    | Maintain consistent IDs within a single continuous video   |

Each method uses:
- **YOLOv11** for player detection  
- **OSNet (Torchreid)** for Re-ID embeddings  
- **DeepSORT** for player tracking  
- **Cosine Similarity + Hungarian Algorithm** (Option 1 only) for cross-view matching  


## 🔍 Option 1: Cross-Camera Re-Identification

📂 [`option1-cross-camera/`](./option1-cross-camera)

Re-identifies players across two separate video feeds (`broadcast.mp4` and `tacticam.mp4`) using embedding matching and the Hungarian algorithm.

📌 **Key Features**:
- Tracks players independently in both views  
- Matches IDs using cosine similarity  
- Handles unmatched players with fallback logic

📄 [View Report](./option1-cross-camera/report.md)


## 🔁 Option 2: Single-Camera Re-Identification

📂 [`option2-single-camera/`](./option2-single-camera)

Tracks and assigns consistent IDs to players within a single camera feed using YOLOv11 + OSNet + DeepSORT.

📌 **Key Features**:
- Real-time player detection + tracking  
- Uses appearance and motion for ID consistency  
- Lightweight and modular design

📄 [View Report](./option2-single-camera/report.md)


## 📥 Setup & Installation


---

## 🔍 Option 1: Cross-Camera Re-Identification

📂 [`option1-cross-camera/`](./option1-cross-camera)

Re-identifies players across two separate video feeds using embedding matching and the Hungarian algorithm.

📌 **Key Features**:
- Tracks players independently in both views  
- Matches IDs using cosine similarity  
- Handles unmatched players with fallback logic

📄 [View Report](./option1-cross-camera/report.md)

---

## 🔁 Option 2: Single-Camera Re-Identification

📂 [`option2-single-camera/`](./option2-single-camera)

Tracks and assigns consistent IDs to players within a single camera feed using YOLOv11 + OSNet + DeepSORT.

📌 **Key Features**:
- Real-time player detection + tracking  
- Uses appearance and motion for ID consistency  
- Lightweight and modular design

📄 [View Report](./option2-single-camera/report.md)

---

## 📥 Setup & Installation

```bash
git clone https://github.com/Bhakthi-Shetty7811/player-reidentification-sports
cd player-re-identification
pip install -r requirements.txt
```


> Note: Place your input videos in `data/` and YOLOv11 model in `model/`. See each subfolder's README for details.


## 📎 External Resources

* [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics)
* [Torchreid (OSNet)](https://github.com/KaiyangZhou/deep-person-reid)
* [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime)


## 👩‍💻 Author

**Bhakthi Shetty**
Final-Year B.Tech IT 2025, UMIT SNDT, GitHub: [Bhakthi-Shetty7811](https://github.com/Bhakthi-Shetty7811), Email: bhakthi.shetty7811@gmail.com




