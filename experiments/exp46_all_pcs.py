"""
exp46: Fetch Neuronpedia features for ALL PCs from exp45 (diverse triples, L3).

We have PC1-3 in the exp45 output. This extends to PC4 through PC10 for a
complete view of what the principal axes of the schema cluster look like
semantically.
"""
import json
import re
import time

import numpy as np
import requests
import torch


print("Loading exp45 results...")
data = torch.load("/Users/macn/Documents/embeddingexp/exp45_results.pt", weights_only=False)
components = data["pca_components"]
var_ratios = data["explained_variance_ratio"]
axis_names = data["axis_names"]
axes = data["axes"]

LAYER = 3
NEURONPEDIA_MODEL = "pythia-70m-deduped"
np_sae = f"{LAYER}-res-sm"
API_DELAY = 0.3


def fetch_feature(feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{np_sae}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except json.JSONDecodeError:
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            return json.loads(clean, strict=False)
    except Exception as e:
        return {"_error": str(e)}


def feat_desc(d):
    if d is None or "_error" in d:
        return "[no data]"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:160]
    return "(no description)"


def top_features(vec, top_n=8):
    abs_loads = np.abs(vec)
    sorted_idx = np.argsort(abs_loads)[::-1]
    positives, negatives = [], []
    for idx in sorted_idx:
        if vec[idx] > 0 and len(positives) < top_n:
            positives.append((int(idx), float(vec[idx])))
        elif vec[idx] < 0 and len(negatives) < top_n:
            negatives.append((int(idx), float(vec[idx])))
        if len(positives) >= top_n and len(negatives) >= top_n:
            break
    return positives, negatives


print(f"\n=== ALL PCs from exp45 (diverse triples, L3) ===\n")

for pc_idx in range(min(len(components), 10)):
    pc = components[pc_idx]
    var = var_ratios[pc_idx] * 100
    pc_unit = pc / np.linalg.norm(pc)

    # Axis loadings (which input axes load on this PC)
    loadings = [(n, float(axes[n] @ pc_unit)) for n in axis_names]
    loadings.sort(key=lambda x: abs(x[1]), reverse=True)
    loading_str = ", ".join(f"{n}{l:+.2f}" for n, l in loadings[:6])

    # Feature lookup
    pos, neg = top_features(pc, top_n=8)

    print(f"\n{'='*72}")
    print(f"PC{pc_idx+1} ({var:.1f}% variance)")
    print(f"{'='*72}")
    print(f"Axis loadings: {loading_str}")
    print(f"\nPositive pole:")
    for f_idx, ld in pos:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
        time.sleep(API_DELAY)
    print(f"\nNegative pole:")
    for f_idx, ld in neg:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
        time.sleep(API_DELAY)

print("\nDone.")
