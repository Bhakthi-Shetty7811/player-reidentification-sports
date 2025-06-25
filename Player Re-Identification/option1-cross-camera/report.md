# ASSIGNMENT : Player Re-Identification in Sports Footage
# 🎥 Option 1: Cross-Camera Player Mapping
## 🧭 1. Approach and Methodology

This implementation addresses the challenge of **re-identifying players across two different camera feeds** (e.g., `broadcast.mp4` and `tacticam.mp4`) of the same soccer game.  
The goal is to assign **consistent player IDs** across both views, even under variations in:

- Camera angle
- Scale
- Lighting
- Player pose or orientation

### 🧠 Pipeline Components:

- 🎯 **YOLOv11** – For detecting players in each frame.
- 🔍 **OSNet (Torchreid)** – For generating visual appearance embeddings.
- 🛰 **DeepSORT** – For tracking players per video using motion + embeddings.
- 🔗 **Cosine Similarity + Hungarian Algorithm** – For final player ID mapping across views.

Each player is **tracked independently** in each video and then **matched across views** using embedding similarity.


## 🧩 2. Code Modules Overview

### `run_detection.py`
- Runs detection and tracking on both videos.
- Extracts bounding boxes, embeddings, and IDs.
- Saves results to `.pkl` files.

### `src/detect.py`
- Loads YOLOv11 + OSNet model.
- Processes cropped players into embeddings.
- Feeds detection results into DeepSORT for tracking.
- Outputs list of tuples → `(track_id, bbox, embedding)`.

### `match_players.py`
- Loads detections from both videos.
- Averages embeddings across frames.
- Computes cosine similarity between player embeddings.
- Applies Hungarian algorithm for best mapping.
- Applies threshold; unmatched IDs prefixed with `T`.

### `run_visualise.py`
- Loads ID mappings and detection data.
- Annotates frames with consistent IDs.
- Saves final labeled videos.


## 🔁 3. Re-Identification Logic 

| **Stage**  | **Module**        | **Logic**                                       | **Outcome**                    |
|------------|-------------------|------------------------------------------------ |------------------------------- |
| Detect     | `YOLOv11`         | Detect players (class 0 or `"player"`)          | 🎯 Good bounding boxes        |
| Embed      | `OSNet`           | Resize → Normalize → 512D vector                | 🧠 Appearance features        |
| Track      | `DeepSORT`        | IOU + Kalman + embedding                        | 🎥 Temp IDs (per video)       |
| Match      | `match_players.py`| Cosine sim. + Hungarian + threshold             | 🔄 Cross-view ID mapping      |
| Fallback   | `match_players.py`| Assign `T<ID>` if match is weak                 | 🚫 Avoids false re-matches    |


## ⚗️ 4. Techniques & Outcomes 

| **Technique**        | **Purpose**                  | **Outcome**      | **Note**                                   |
|--------------------- |------------------------------|------------------|------------------------------------------- |
| YOLOv11 (Custom)     | Detect players               | ✅ Good          | Sometimes detected refs/partials          |
| DeepSORT Tracker     | Temp IDs per video           | ⚠️ Consistent    | Not usable across videos                  |
| OSNet (Torchreid)    | Appearance embeddings        | ⚠️ OKish         | Failed with blur/small crops              |
| Cosine Matching      | Compare across views         | ⚠️ Partial       | Sensitive to lighting/angle               |
| Spatial Proximity    | Match by closeness in frame  | ❌ Discarded     | Views too different                       |
| T-prefixed IDs       | Handle unmatched cases       | ✅ Helpful       | Avoids wrong matches                      |
| Conf. + Size Filter  | Improve detection quality    | ✅ Improved      | Reduced noise/crops                       |
| DeepSORT w/o ReID    | Track via motion only        | ❌ Weak          | Broke under occlusion                     |
| Shallow CNN Emb.     | Custom embeddings            | ❌ Poor          | Low generalization                        |
| YOLO + ReID Head     | Multi-task ReID + detect     | ❌ Failed        | Too complex, poor performance             |


## 🧱 5. Challenges Encountered

- 🎭 **ReID Similarity Confusion**  
  → Players with similar shirts or blurring confused the embedder.

- 📷 **Perspective Shift**  
  → Different camera angles led to bounding box variations, hampering embedding alignment.

- 🔧 **Torchreid Compatibility**  
  → Required source install and correct OSNet weights (not available via pip).

- ⚖️ **Embedding Instability**  
  → Small players or extreme angles produced noisy or uninformative embeddings.

- 🧮 **Hungarian Assignment Sensitivity**  
  → Global optimality sometimes prioritized weak matches due to close distances.


## ✅ 6. Conclusion

The Option 1 pipeline successfully performs **player mapping across dual video feeds** using a structured combination of:

- Detection (YOLOv11)  
- Appearance Embedding (OSNet)  
- Tracking (DeepSORT)  
- Assignment (Hungarian Algorithm + Cosine Distance)

Despite certain challenges (e.g., occlusions, embedding failures), the fallback mechanisms (`T<ID>`), size filtering, and cosine-based assignment yield **robust cross-camera player identity alignment**.

> This system can be further improved with multi-view geometry, camera calibration, or fine-tuned re-ID models.

## 👤 Author

**Bhakthi Shetty**  
🎓 Final-Year B.Tech (IT), UMIT SNDT  
📁 [Project Repository](-----------------------)

