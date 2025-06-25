import cv2
import pickle

def load_detections(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def draw_boxes(frame, detections, ids):
    for det, pid in zip(detections, ids):
        x1, y1, x2, y2 = map(int, det)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def visualize(video_path, detections, id_map, is_tacticam, out_path):
    """Overlay bounding boxes and IDs on video frames."""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

    for frame_data in detections:
        ret, frame = cap.read()
        if not ret:
            break

        dets, ids = [], []
        for tid, box, _ in frame_data:
            player_id = id_map.get(tid, f"T{tid}") if is_tacticam else tid
            dets.append(box)
            ids.append(player_id)

        frame = draw_boxes(frame, dets, ids)
        out.write(frame)

    cap.release()
    out.release()
    print(f"\u2705 Saved output to {out_path}")

# Load detections and ID mappings
broadcast_dets = load_detections("outputs/broadcast_data.pkl")
tacticam_dets = load_detections("outputs/tacticam_data.pkl")
id_mapping = pickle.load(open("outputs/player_id_mapping.pkl", "rb"))

# Save labeled videos
visualize("data/broadcast.mp4", broadcast_dets, id_mapping, False, "outputs/broadcast_labeled.mp4")
visualize("data/tacticam.mp4", tacticam_dets, id_mapping, True, "outputs/tacticam_labeled.mp4")

