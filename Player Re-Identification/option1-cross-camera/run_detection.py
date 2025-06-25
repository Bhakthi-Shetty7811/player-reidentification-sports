from src.detect import detect_players
import os

os.makedirs("outputs", exist_ok=True)

# Run detection on both camera sources
detect_players("data/broadcast.mp4", "outputs/broadcast_data.pkl")
detect_players("data/tacticam.mp4", "outputs/tacticam_data.pkl")

