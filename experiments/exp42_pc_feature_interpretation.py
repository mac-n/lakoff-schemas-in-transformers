"""
exp42: Identify what the SAE PCA components actually ARE semantically by
looking up the top-loading SAE features on Neuronpedia.

In SAE space, each principal component vector is 32k-dimensional, where each
dimension corresponds to an interpretable SAE feature. We can find the
features that load most strongly on each PC, look up their auto-interp
descriptions on Neuronpedia, and read off what the PC represents semantically.

This is the SAE-substrate equivalent of nearest-neighbor word lookup in
word2vec — but more informative because the SAE features have explicit
interpretable descriptions.

Focus on mid-layers (L2, L3, L4) where PC1 had the strongest structure.
For each layer:
  - PC1 (the "IO vs cluster" axis we found)
  - PC2 (the "LR" axis)
  - PC3 (whatever's next)

For each PC, get top 10 features by absolute loading on positive pole + 10 on
negative pole. Look up Neuronpedia descriptions. Print the semantic shape.
"""
import json
import re
import time

import numpy as np
import requests
import torch


# Load PCA results
print("Loading exp41 PCA results...")
data = torch.load("/Users/macn/Documents/embeddingexp/exp41_results.pt", weights_only=False)
per_layer_pca = data["per_layer_pca"]

NEURONPEDIA_MODEL = "pythia-70m-deduped"
API_DELAY = 0.3


def fetch_feature(np_sae, feat_idx):
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


def feature_summary(d):
    if d is None:
        return "[no data]"
    if "_error" in d:
        return f"[error: {d['_error']}]"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:180]
    return "(no auto-interp description)"


def top_features_on_pc(pc_vec, top_n=12):
    """Return (positive pole top features, negative pole top features) as (idx, loading) lists."""
    # PC vec is in 32k-dim SAE feature space
    abs_loadings = np.abs(pc_vec)
    # Top by absolute, then split by sign
    top_indices = np.argsort(abs_loadings)[::-1][:top_n*4]  # grab extra so we have enough of each sign
    positives = []
    negatives = []
    for idx in top_indices:
        if pc_vec[idx] > 0:
            positives.append((int(idx), float(pc_vec[idx])))
        else:
            negatives.append((int(idx), float(pc_vec[idx])))
        if len(positives) >= top_n and len(negatives) >= top_n:
            break
    return positives[:top_n], negatives[:top_n]


# Focus on layers 2, 3, 4 (strongest cluster structure per exp41)
for layer in [2, 3, 4]:
    np_sae = f"{layer}-res-sm"
    print(f"\n{'='*72}")
    print(f"LAYER {layer} ({np_sae})")
    print(f"{'='*72}")

    pca_result = per_layer_pca[layer]
    components = pca_result["components"]  # (n_pcs, 32768)
    variance_ratios = pca_result["explained_variance_ratio"]
    axis_names = pca_result["axis_names"]

    print(f"\nVariance ratios: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(variance_ratios)))

    # Show input-axis loadings as a refresher
    print(f"\nInput-axis loadings (for reference):")
    schema_axes = pca_result["schema_axes"]
    for pc_idx in range(min(3, len(components))):
        pc = components[pc_idx]
        pc_unit = pc / np.linalg.norm(pc)
        loadings = [(name, float(schema_axes[name] @ pc_unit)) for name in axis_names]
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  PC{pc_idx+1}: " + ", ".join(f"{n}{l:+.2f}" for n, l in loadings))

    # For each PC, look up the top-loading features
    for pc_idx in range(min(3, len(components))):
        pc_vec = components[pc_idx]
        var_pct = variance_ratios[pc_idx] * 100

        positives, negatives = top_features_on_pc(pc_vec, top_n=10)

        print(f"\n--- PC{pc_idx+1} ({var_pct:.1f}% variance) ---")
        print(f"\n  Positive pole (top 10 features):")
        for feat_idx, loading in positives:
            data = fetch_feature(np_sae, feat_idx)
            desc = feature_summary(data)
            print(f"    feat {feat_idx:>5d} (load {loading:+.4f}): {desc}")
            time.sleep(API_DELAY)

        print(f"\n  Negative pole (top 10 features):")
        for feat_idx, loading in negatives:
            data = fetch_feature(np_sae, feat_idx)
            desc = feature_summary(data)
            print(f"    feat {feat_idx:>5d} (load {loading:+.4f}): {desc}")
            time.sleep(API_DELAY)

print("\nDone.")
