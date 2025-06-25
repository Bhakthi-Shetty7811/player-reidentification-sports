"""
This module handles the detection and feature embedding of players in a video using YOLOv8 for detection,
Torchreid (OSNet) for feature extraction, and DeepSORT for tracking.
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import torchreid
import os
import pickle
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model for player detection
yolo = YOLO('model/yolov11.pt')

# Load OSNet ReID model for feature extraction
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval().to(device)

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=25,
    n_init=3,
    max_iou_distance=0.6,
    embedder='torchreid',
    half=True
)

# Image pre-processing pipeline for OSNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_embedding(image):
    """Generate embedding using OSNet for a cropped player image."""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(image)
    return features.cpu().numpy().flatten()

def detect_players(video_path, save_path):
    """
    Detects and tracks players in the input video, and saves the resulting player track data.

    Args:
        video_path (str): Path to the input video.
        save_path (str): Path to save the player tracking data (Pickle).
    """
    cap = cv2.VideoCapture(video_path)
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection on current frame
        results = yolo.predict(frame, conf=0.6, iou=0.4, verbose=False)[0]
        detections = []

        # Filter and process player detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = yolo.names[cls]

            if label.lower() == "player" or cls == 0:
                w, h = x2 - x1, y2 - y1
                if w < 20 or h < 40:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emb = get_embedding(crop)
                detections.append(([x1, y1, w, h], conf, label, emb))

        # Update tracker with detections and embeddings
        tracks = tracker.update_tracks(
            [(d[0], d[1], d[2]) for d in detections],
            embeds=[d[3] for d in detections],
            frame=frame
        )

        frame_output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            # Match track with corresponding embedding using spatial proximity
            for d in detections:
                cx1 = d[0][0] + d[0][2] / 2
                cy1 = d[0][1] + d[0][3] / 2
                cx2 = (x1 + x2) / 2
                cy2 = (y1 + y2) / 2
                if np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2])) < 40:
                    frame_output.append((track_id, [x1, y1, x2, y2], d[3]))
                    break

        data.append(frame_output)

    # Save results as pickle
    cap.release()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Detections saved to {save_path}")


