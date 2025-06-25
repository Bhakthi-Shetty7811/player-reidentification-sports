"""
Contains helper functions for:
- Computing embeddings using OSNet ReID model
- Drawing bounding boxes with player IDs
- Loading YOLOv8 model
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import torchreid

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load OSNet ReID model
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval().to(device)

# Transformation for person crops (as expected by OSNet)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_embedding(image):
    """
    Extracts a ReID embedding vector from a cropped player image.

    Args:
        image (np.array): Cropped image of player.

    Returns:
        np.array: Flattened ReID feature vector.
    """
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(image)
    return features.cpu().numpy().flatten()

def draw_boxes(frame, detections, player_ids):
    """
    Draws bounding boxes and player IDs on the frame.

    Args:
        frame (np.array): Original video frame.
        detections (list): List of bounding box coordinates.
        player_ids (list): List of corresponding track IDs.

    Returns:
        np.array: Annotated frame.
    """
    for det, pid in zip(detections, player_ids):
        x1, y1, x2, y2 = map(int, det[:4])
        color = (0, 255, 0)  # Green box for confirmed tracks
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def load_yolo():
    """
    Loads the YOLOv8 model for player detection.

    Returns:
        YOLO: Loaded YOLO model instance.
    """
    return YOLO('model/yolov11.pt')
