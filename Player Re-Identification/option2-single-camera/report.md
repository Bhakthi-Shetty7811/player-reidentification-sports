# ASSIGNMENT : Player Re-Identification in Sports Footage
# ⚽ Option 2: Re-Identification in single feed
## 🧭 1. Approach and Methodology

This project focuses on the task of **re-identifying soccer players across frames within a single video** (broadcast or tacticam). The goal is to assign and **maintain consistent player IDs** throughout the video, even under:

- Fast player movement
- Occlusions or overlaps
- Changes in scale or orientation

### 🧠 Pipeline Components:

- 🎯 **YOLOv11 (Ultralytics)** – For detecting player bounding boxes.
- 🧬 **OSNet (Torchreid)** – For generating appearance-based Re-ID embeddings.
- 🛰 **DeepSORT** – For assigning and maintaining consistent track IDs using motion and appearance cues.

All components work together to produce **frame-consistent identity tracking** of all players.


## 🧩 2. Code Modules Overview

### `main.py`
- Loads the input video.
- Detects players using YOLOv11.
- Extracts embeddings via OSNet for each detected player.
- Tracks players using DeepSORT.
- Draws bounding boxes and IDs, then saves the output video.

### `detect.py`
- Runs frame-wise player detection.
- Extracts bounding box, Re-ID embedding, and assigned track ID.
- Uses Euclidean distance to verify consistency between detector and tracker.
- Saves data in `.pkl` format for reuse.

### `track.py`
- Initializes DeepSORT tracker.
- Loads OSNet model as the appearance embedder.

### `utils.py`
- Handles all preprocessing (resize, normalization) for embedding.
- Loads YOLOv11 model.
- Defines OpenCV helper functions for drawing labeled frames.


## 🔁 3. Re-Identification Logic

| **Stage**  | **Module**   | **Logic**                                | **Outcome**                     |
|------------|--------------|------------------------------------------|-------------------------------- |
| Detection  | `YOLOv11`    | Class 0 or label "player"                | ✅ Accurate bboxes             |
| Embedding  | `OSNet`      | Resize → Normalize → 512D vector         | 🧠 Appearance descriptor       |
| Tracking   | `DeepSORT`   | Kalman + IOU + embedding match           | 🎯 ID consistency in video     |
| Matching   | `detect.py`  | Compare tracker vs detector centers      | 🔄 Sync embedding to ID        |


## ⚗️ 4. Techniques Tried and Outcomes

| **Technique**            | **Purpose**              | **Outcome**      | **Remarks**                         |
|--------------------------|--------------------------|------------------|------------------------------------ |
| DeepSORT w/o ReID        | Motion-only tracking     | ❌ Failed        | IDs swapped after occlusion        |
| YOLO + ReID Head         | Unified model            | ❌ Complex       | Poor embedding quality             |
| Custom CNN               | Replace OSNet            | ❌ Weak          | Failed under similar uniforms      |
| Distance Only Matching   | Skip DeepSORT            | ❌ Unstable      | No smoothing; abrupt ID changes    |



## 🧱 5. Challenges Encountered

- 🎭 **ReID Misalignment**  
  → Players with similar uniforms or blurry frames caused nearly identical embeddings.

- 🚶‍♂️ **Player Occlusions**  
  → DeepSORT lost tracks during overlaps and occasionally reassigned incorrect IDs.

- 🧩 **Integration Complexity**  
  → Torchreid required patching for compatibility with DeepSORT. Needed correct OSNet checkpoints.

- 🏷️ **YOLO Label Conflicts**  
  → Certain YOLO models labeled all classes (ball, ref, etc.). Custom logic was required to keep only class `0`.


## ✅ 6. Conclusion

The **Option 2 pipeline** provides a reliable, modular approach for consistent player tracking **within a single video** using:

- **YOLOv11** for detection  
- **OSNet** for appearance-based features  
- **DeepSORT** for multi-frame tracking

Though challenges like occlusions and look-alike players affect accuracy, the use of pretrained Re-ID embeddings and careful logic makes this implementation suitable for real-time analysis and annotation tasks.

> With future upgrades such as temporal smoothing or team-based heuristics, this pipeline can serve as a foundation for intelligent sports video analysis systems.


## 👤 Author

**Bhakthi Shetty**  
🎓 Final-Year B.Tech (IT), UMIT SNDT  
📁 [Project Repository](-----------------------)


