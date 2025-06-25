import pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def average_embeddings(data, min_obs=3):
    """Average embeddings per track ID if seen at least `min_obs` times."""
    emb_dict = defaultdict(list)
    for frame in data:
        for tid, _, emb in frame:
            emb_dict[tid].append(emb)
    return {tid: np.mean(embs, axis=0) for tid, embs in emb_dict.items() if len(embs) >= min_obs}

# Load tracked detections from both views
b_data = load("outputs/broadcast_data.pkl")
t_data = load("outputs/tacticam_data.pkl")
b_embs = average_embeddings(b_data)
t_embs = average_embeddings(t_data)

# Prepare vectors
b_ids = list(b_embs.keys())
t_ids = list(t_embs.keys())
b_vecs = np.vstack([b_embs[i] for i in b_ids])
t_vecs = np.vstack([t_embs[j] for j in t_ids])

dist_matrix = cdist(t_vecs, b_vecs, metric='cosine')
row_ind, col_ind = linear_sum_assignment(dist_matrix)

# Match tacticam IDs to broadcast IDs using a distance threshold
mapping = {}
THRESH = 0.32 # Empirically chosen threshold for cosine similarity between embeddings
used = set()
for i, j in zip(row_ind, col_ind):
    if dist_matrix[i, j] < THRESH:
        mapping[t_ids[i]] = b_ids[j]
        used.add(b_ids[j])
    else:
        mapping[t_ids[i]] = f"T{t_ids[i]}"

for tid in t_ids:
    mapping.setdefault(tid, f"T{tid}")

# Save mapping
with open("outputs/player_id_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print(f"\u2705 Matched: {sum(1 for v in mapping.values() if not str(v).startswith('T'))} / {len(t_ids)}")




