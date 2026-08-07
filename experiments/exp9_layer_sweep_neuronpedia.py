"""
exp9_layer_sweep_neuronpedia.py — run the exp8 pipeline on multiple layers.

Building an interpretive map of what each Pythia 70m-deduped SAE's geometric
structure is "about." Now that we have a clean negative result for image
schemas at res-sm L5 (it's about Pile-domain register), we want to see how
representation organisation varies across the network.

Layers to characterise:
  - mlp-sm layer 5  : also concentrated (PR=163, top1=6%). Last MLP. Different
                      substrate at same depth — does it tell the same register
                      story or something complementary?
  - res-sm layer 3  : NOT concentrated (PR=423, top1=1.4%). Mid-layer where
                      "SAE-evenness" should rule. Top PCs should be diffuse
                      and not coherently interpretable. Acts as control: if
                      these PCs DO show coherent themes, the "no schema
                      structure here" story needs revising.
  - res-sm layer 0  : first residual stream (just after embedding + first
                      block). Modestly concentrated (PR=296, top1=3.2%).
                      What does the model "see" before any deep processing?
                      Expect lexical/grammatical structure if anything.

For each layer: PCA the SAE decoder, identify top 10 features per top 10 PCs,
look up via Neuronpedia API, write a per-layer .md report.

Final output: one consolidated `results_exp9_layer_sweep.md` with all three
layers + a summary comparison section.
"""

import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE

# ---- Config ----
LAYERS = [
    # (release, sae_id, neuronpedia_sae_id, short_label)
    ("pythia-70m-deduped-mlp-sm", "blocks.5.hook_mlp_out", "5-mlp-sm", "mlp-sm L5"),
    ("pythia-70m-deduped-res-sm", "blocks.3.hook_resid_post", "3-res-sm", "res-sm L3"),
    ("pythia-70m-deduped-res-sm", "blocks.0.hook_resid_post", "0-res-sm", "res-sm L0"),
]
NEURONPEDIA_MODEL = "pythia-70m-deduped"
TOP_K_PCS = 10
TOP_N_FEATURES_PER_PC = 10
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
            try:
                return json.loads(clean, strict=False)
            except Exception:
                return None
    except Exception as e:
        return {"_error": str(e)}


def feature_summary(d):
    if d is None:
        return "  [API returned no data]"
    if "_error" in d:
        return f"  [API error: {d['_error']}]"
    expls = d.get("explanations") or []
    expl_str = expls[0].get("description", "").strip() if expls else "(no auto-interp description)"
    expl_str = expl_str[:180]
    n_acts = len(d.get("activations") or [])
    max_act = d.get("maxActApprox", 0) or 0
    pos_tokens = (d.get("pos_str") or [])[:5]
    pos_str = " ".join(repr(t) for t in pos_tokens) if pos_tokens else ""
    return f"  desc: {expl_str}  | maxAct≈{max_act:.1f}, n_top_acts={n_acts}  | pos_logits: {pos_str}"


# ---- Step 1: PCA each SAE, collect top features ----
print(f"Stage 1: PCA on {len(LAYERS)} SAEs, identifying top features...")
per_layer_data = {}
all_needed_features = []  # list of (np_sae, feat_idx) to fetch

for release, sae_id, np_sae, label in LAYERS:
    print(f"\n  Loading {label} ({release} / {sae_id})...")
    result = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
    sae = result[0] if isinstance(result, tuple) else result
    W_dec = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
    W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    n_features, d_model = W_dec.shape
    W_centered = W_dec - W_dec.mean(0, keepdims=True)
    U, S, Vh = np.linalg.svd(W_centered, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()
    feature_loadings = W_centered @ Vh.T

    top_per_pc = {}
    for pc in range(TOP_K_PCS):
        loadings = feature_loadings[:, pc]
        abs_loadings = np.abs(loadings)
        top_idx = np.argsort(-abs_loadings)[:TOP_N_FEATURES_PER_PC]
        top_per_pc[pc] = [(int(i), float(loadings[i]), float(abs_loadings[i])) for i in top_idx]
        for i in top_idx:
            all_needed_features.append((np_sae, int(i)))

    per_layer_data[label] = {
        "release": release,
        "sae_id": sae_id,
        "np_sae": np_sae,
        "n_features": n_features,
        "d_model": d_model,
        "var_ratio": var_ratio,
        "top_per_pc": top_per_pc,
    }
    print(f"  PR={float((var_ratio.sum()**2) / (var_ratio**2).sum()):.1f}, top1_var={var_ratio[0]:.4f}")
    del sae, W_dec, W_centered, U, S, Vh, feature_loadings

# Deduplicate
unique_features = sorted(set(all_needed_features))
print(f"\nTotal feature lookups needed: {len(all_needed_features)}")
print(f"After deduplication: {len(unique_features)}")


# ---- Step 2: Batch-query Neuronpedia ----
print(f"\nStage 2: querying Neuronpedia ({len(unique_features)} features, "
      f"~{len(unique_features) * API_DELAY:.0f}s minimum)...")
feature_cache = {}  # (np_sae, feat_idx) -> data dict
for i, (np_sae, feat_idx) in enumerate(unique_features):
    if i % 20 == 0:
        print(f"  {i}/{len(unique_features)}")
    feature_cache[(np_sae, feat_idx)] = fetch_feature(np_sae, feat_idx)
    time.sleep(API_DELAY)


# ---- Step 3: Write per-layer report ----
out_path = "/Users/macn/Documents/embeddingexp/results_exp9_layer_sweep.md"
with open(out_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("=" * 100)
    out("exp9: SAE decoder PCA + Neuronpedia auto-interp lookup, layer sweep")
    out("=" * 100)
    out()
    out(f"Pipeline: decoder PCA → top-{TOP_N_FEATURES_PER_PC} features per top-{TOP_K_PCS} PC → Neuronpedia lookup.")
    out(f"Hypothesis after exp8: res-sm L5 organises around Pile-domain register (legal, biomedical, math, programming),")
    out(f"NOT image schemas. exp9 tests how this pattern generalises across layers/substrates.")
    out()

    for release, sae_id, np_sae, label in LAYERS:
        data = per_layer_data[label]
        out("#" * 100)
        out(f"# {label}  ({release} / {sae_id})")
        out("#" * 100)
        out()
        out(f"n_features={data['n_features']:,}, d_model={data['d_model']}")
        pr = float((data['var_ratio'].sum()**2) / (data['var_ratio']**2).sum())
        out(f"Participation ratio: {pr:.1f}  |  top {TOP_K_PCS} PC variance ratios: "
            f"{[f'{v:.4f}' for v in data['var_ratio'][:TOP_K_PCS]]}")
        out(f"Cumulative variance of top {TOP_K_PCS}: {data['var_ratio'][:TOP_K_PCS].sum():.4f}")
        out()

        for pc in range(TOP_K_PCS):
            out("-" * 100)
            out(f"PC {pc}  (variance ratio = {data['var_ratio'][pc]:.4f}, "
                f"cumulative = {data['var_ratio'][:pc+1].sum():.4f})")
            out("-" * 100)
            for feat_idx, signed_loading, abs_loading in data['top_per_pc'][pc]:
                d = feature_cache.get((np_sae, feat_idx))
                out(f"feat {feat_idx:>5d}  loading {signed_loading:+.4f}  "
                    f"[https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{np_sae}/{feat_idx}]")
                out(feature_summary(d))
            out()

    # Summary section
    out("=" * 100)
    out("SUMMARY: what each layer's geometric structure is 'about'")
    out("=" * 100)
    out("(Fill in interpretation manually after reading the per-layer PCs above.)")
    out()
    for release, sae_id, np_sae, label in LAYERS:
        data = per_layer_data[label]
        pr = float((data['var_ratio'].sum()**2) / (data['var_ratio']**2).sum())
        out(f"  {label:12s}  PR={pr:6.1f}  top1={data['var_ratio'][0]:.4f}  cum10={data['var_ratio'][:10].sum():.3f}")
    out()

print(f"\nSaved to {out_path}")

# Save raw cache
cache_path = "/Users/macn/Documents/embeddingexp/exp9_neuronpedia_cache.pt"
torch.save({
    "per_layer_data": per_layer_data,
    "feature_cache": {f"{k[0]}_{k[1]}": v for k, v in feature_cache.items()},
}, cache_path)
print(f"Cache saved to {cache_path}")
