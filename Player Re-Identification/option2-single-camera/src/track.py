"""
Initializes the DeepSORT tracker using Torchreid embeddings for person re-identification.
This module sets the configuration for the tracker such as maximum age, minimum hits, and IOU threshold.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort

def init_tracker():
    """
    Initializes and returns a DeepSort tracker instance with specified configurations.

    Returns:
        DeepSort: An instance of the DeepSort tracker.
    """
    return DeepSort(
        max_age=30,               # Number of frames to keep track before deleting an unmatched object
        n_init=3,                 # Minimum detections before confirming a track
        max_iou_distance=0.6,     # IOU threshold for association
        embedder='torchreid',     # Re-identification model to be used
        half=True                 # Run inference in half precision for speed (if supported)
    )










