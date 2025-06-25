"""
Runs player detection and tracking on a video file using YOLOv8 + DeepSORT + TorchReID.
Outputs an annotated video showing player IDs consistently tracked across frames.
"""

import cv2
import os
from src.track import init_tracker
from src.utils import draw_boxes, get_embedding, load_yolo

# Paths
video_path = 'data/15sec_input_720p.mp4'   # Input video
output_path = 'output.mp4'                 # Output video path
yolo_model = load_yolo()                   # Load detection model

# Setup video capture and output writer
cap = cv2.VideoCapture(video_path)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Initialize the tracker
tracker = init_tracker()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection using YOLO
    results = yolo_model.predict(frame, conf=0.5, iou=0.45, verbose=False)[0]
    detections = []
    embeddings = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = yolo_model.names[cls]

        w_box, h_box = x2 - x1, y2 - y1
        if label.lower() == 'player' and w_box > 30 and h_box > 60:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            emb = get_embedding(crop)
            detections.append(([x1, y1, w_box, h_box], conf, label))
            embeddings.append(emb)

    # Update tracker with new detections
    tracks = tracker.update_tracks(detections, embeds=embeddings, frame=frame)

    # Collect confirmed tracks for drawing
    dets, ids = [], []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        dets.append([x1, y1, x2, y2])
        ids.append(track_id)

    # Annotate and write frame
    frame = draw_boxes(frame, dets, ids)
    out.write(frame)
    frame_idx += 1

# Finalize
cap.release()
out.release()
print("✅ Output video saved as:", output_path)
