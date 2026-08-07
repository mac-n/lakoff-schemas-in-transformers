"""
exp8_neuronpedia_lookup.py — read what's on the principal components.

For the res-sm layer 5 SAE of Pythia 70m-deduped (the strongest "real
concentration not explained by anisotropy" signal from exp7c):

  1. Load SAE, PCA the centred decoder
  2. For each of the top K principal components:
       - Get the top N features by absolute loading on that PC
  3. For each unique top feature: query Neuronpedia API for auto-interp
     description and basic stats
  4. Pretty-print per-PC tables: which features load on each PC + what they
     fire on (per Neuronpedia)

The actual interpretive question: do the top PCs of res-sm L5's decoder
cluster around schema-shaped feature groups (UP, CONTAINER, etc.), or
mechanism-shaped (next-token-is-noun, predict-period), or topic-shaped
(math, code, sports)?
"""

import json
import time
import re

import numpy as np
import requests
import torch
from sae_lens import SAE

# ---- Config ----
RELEASE = "pythia-70m-deduped-res-sm"
SAE_ID = "blocks.5.hook_resid_post"
NEURONPEDIA_SAE = "5-res-sm"  # the URL/API path component
NEURONPEDIA_MODEL = "pythia-70m-deduped"
TOP_K_PCS = 10
TOP_N_FEATURES_PER_PC = 10
API_DELAY = 0.3  # seconds between API calls — be polite

# ---- 1. Load SAE ----
print(f"Loading SAE {RELEASE} / {SAE_ID}...")
result = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID, device="cpu")
sae = result[0] if isinstance(result, tuple) else result
W_dec = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
print(f"  decoder shape: {W_dec.shape}")
W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
n_features, d_model = W_dec.shape

# ---- 2. PCA on centred decoder ----
print(f"\nPCA on centred decoder...")
W_centered = W_dec - W_dec.mean(0, keepdims=True)
U, S, Vh = np.linalg.svd(W_centered, full_matrices=False)
var_ratio = (S ** 2) / (S ** 2).sum()
print(f"  Top {TOP_K_PCS} PC variance ratios: {[f'{v:.4f}' for v in var_ratio[:TOP_K_PCS]]}")

# Feature loadings on each PC: project each feature onto each PC direction
# shape (n_features, d_model)
feature_loadings = W_centered @ Vh.T  # column j = loading of all features on PC j

# ---- 3. Top features per PC ----
print(f"\nIdentifying top {TOP_N_FEATURES_PER_PC} features per top {TOP_K_PCS} PCs...")
top_features_per_pc = {}  # pc_idx -> list of (feat_idx, signed_loading, abs_loading)
all_unique_features = set()
for pc in range(TOP_K_PCS):
    loadings = feature_loadings[:, pc]
    abs_loadings = np.abs(loadings)
    top_idx = np.argsort(-abs_loadings)[:TOP_N_FEATURES_PER_PC]
    top_features_per_pc[pc] = [(int(i), float(loadings[i]), float(abs_loadings[i])) for i in top_idx]
    all_unique_features.update(int(i) for i in top_idx)
print(f"  {len(all_unique_features)} unique features to look up.")


# ---- 4. Query Neuronpedia API ----
def fetch_feature(feat_idx):
    """Fetch a single feature's data from Neuronpedia. Returns dict or None on failure."""
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_SAE}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        # Try requests' built-in JSON parser first
        try:
            return r.json()
        except json.JSONDecodeError:
            # Fall back: strip raw control characters and retry
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            try:
                return json.loads(clean, strict=False)
            except Exception:
                return None
    except Exception as e:
        return {"_error": str(e)}


print(f"\nQuerying Neuronpedia API for {len(all_unique_features)} features "
      f"(~{len(all_unique_features) * API_DELAY:.0f}s minimum)...")
feature_cache = {}
for i, feat_idx in enumerate(sorted(all_unique_features)):
    if i % 10 == 0:
        print(f"  {i}/{len(all_unique_features)}")
    d = fetch_feature(feat_idx)
    feature_cache[feat_idx] = d
    time.sleep(API_DELAY)


# ---- 5. Pretty-print per-PC tables ----
def feature_summary(feat_idx):
    """Return a one-liner string summary for the feature."""
    d = feature_cache.get(feat_idx)
    if d is None:
        return "  [API returned no data]"
    if "_error" in d:
        return f"  [API error: {d['_error']}]"
    expls = d.get("explanations") or []
    expl_str = expls[0].get("description", "").strip() if expls else "(no auto-interp description)"
    expl_str = expl_str[:180]
    acts = d.get("activations") or []
    n_acts = len(acts)
    max_act = d.get("maxActApprox", 0) or 0
    pos_tokens = (d.get("pos_str") or [])[:5]
    neg_tokens = (d.get("neg_str") or [])[:5]
    pos_str = " ".join(repr(t) for t in pos_tokens) if pos_tokens else ""
    return f"  desc: {expl_str}  | maxAct≈{max_act:.1f}, n_top_acts={n_acts}  | pos_logits: {pos_str}"


out_path_md = "/Users/macn/Documents/embeddingexp/results_exp8.md"
with open(out_path_md, "w") as f_md:
    def out(s=""):
        print(s)
        f_md.write(s + "\n")

    out("=" * 100)
    out(f"exp8: Neuronpedia lookup of top-loading features per top PC")
    out(f"SAE: {RELEASE} / {SAE_ID}  (Pythia 70m-deduped, residual stream, final layer)")
    out(f"PCA on (centred, unit-normalised) decoder, n_features={n_features}, d_model={d_model}")
    out("=" * 100)
    out()
    out(f"Top {TOP_K_PCS} PC variance ratios: {[f'{v:.4f}' for v in var_ratio[:TOP_K_PCS]]}")
    out(f"Cumulative variance of top {TOP_K_PCS}: {var_ratio[:TOP_K_PCS].sum():.4f}")
    out()

    for pc in range(TOP_K_PCS):
        out("-" * 100)
        out(f"PC {pc}  (variance ratio = {var_ratio[pc]:.4f}, "
            f"cumulative through this PC = {var_ratio[:pc+1].sum():.4f})")
        out("-" * 100)
        for feat_idx, signed_loading, abs_loading in top_features_per_pc[pc]:
            out(f"feat {feat_idx:>5d}  loading {signed_loading:+.4f}  "
                f"[https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{NEURONPEDIA_SAE}/{feat_idx}]")
            out(feature_summary(feat_idx))
        out()

print(f"\nSaved to {out_path_md}")

# Also save raw cache for downstream analysis
cache_path = "/Users/macn/Documents/embeddingexp/exp8_neuronpedia_cache.pt"
torch.save({
    "feature_cache": feature_cache,
    "top_features_per_pc": top_features_per_pc,
    "var_ratio": var_ratio,
    "feature_loadings": feature_loadings,
}, cache_path)
print(f"Cache saved to {cache_path}")
