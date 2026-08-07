"""
exp44: Verify the polarity / semantic content of the candidate axes themselves
in SAE space. We need to confirm:

1. Does +COHERENCE_axis correspond to "consistent/ordered" content (as designed)?
2. Does +EXISTENCE_axis correspond to "created/built" content (as designed)?
3. Does +SUCCESS_axis correspond to "win/succeed" content (as designed)?
4. Does +VALENCE_axis correspond to "pleasant/good" content (as designed)?
5. Does +AROUSAL_axis correspond to "intense/urgent" content (as designed)?
6. Does +LOSS_axis correspond to "gain/security" content (as designed)?

Then we can verify the PC1 sign-loadings interpretation:
- SUCCESS +0.94: PC1+ should align with WIN content
- AROUSAL -0.89: PC1+ should be CALM content (low-arousal)
- EXISTENCE -0.91: PC1+ should align with DESTROYED content (anti-creation)
- COHERENCE -0.87: PC1+ should align with ANOMALOUS content (anti-consistent)

But the PC1+ features were formal-legal-affirmation-gratitude (which sounds like
CALM-CONSISTENT-CREATED, not the anti-versions). So one of my polarity readings
must be wrong. Need to check.
"""
import json
import re
import time

import numpy as np
import requests
import torch

NEURONPEDIA_MODEL = "pythia-70m-deduped"
API_DELAY = 0.3
LAYER = 3
np_sae = f"{LAYER}-res-sm"


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


def top_features(vec, top_n=10):
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


# Load saved L3 axes
print(f"Loading axes from exp43 (layer {LAYER})...")
data = torch.load("/Users/macn/Documents/embeddingexp/exp43_results.pt", weights_only=False)
axes = data[LAYER]["axes"]

# For each axis we want to check, fetch top features
AXES_TO_CHECK = ["VALENCE", "AROUSAL", "EXISTENCE", "COHERENCE", "SUCCESS", "LOSS"]

for axis_name in AXES_TO_CHECK:
    axis = axes[axis_name]
    pos, neg = top_features(axis, top_n=8)

    print(f"\n{'='*60}")
    print(f"{axis_name} axis (layer {LAYER})")
    print(f"{'='*60}")

    print(f"\nPositive pole (top features that load + on this axis):")
    for f_idx, ld in pos:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.4f}): {feat_desc(d)}")
        time.sleep(API_DELAY)

    print(f"\nNegative pole (top features that load − on this axis):")
    for f_idx, ld in neg:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.4f}): {feat_desc(d)}")
        time.sleep(API_DELAY)

print("\nDone.")
