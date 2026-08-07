"""
exp7c_sweep_wikitext.py — re-run of the exp7b sweep with the real wikitext
corpus instead of the 80-hardcoded-sentence fallback.

Niamh's catch: hardcoded sentences were uniform in length and grammatical
structure, so the covariance estimate they produced was unrepresentative of
the actual residual-stream anisotropy the SAE was trained on. The
covariance-matched null built from that estimate could be wrong in either
direction.

Same 6 layers × 2 substrates ({res-sm, mlp-sm}) sweep as exp7b. Same null
comparison structure (covariance-matched + isotropic). Only difference: the
neutral corpus.

Data: EleutherAI/wikitext_document_level / wikitext-2-raw-v1, saved as
/Users/macn/Documents/embeddingexp/wikitext2_docs.jsonl (629 documents).
"""

import json
import gc

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---- Config ----
LAYERS = list(range(6))
SUBSTRATES = {
    "res-sm": ("pythia-70m-deduped-res-sm", "blocks.{}.hook_resid_post"),
    "mlp-sm": ("pythia-70m-deduped-mlp-sm", "blocks.{}.hook_mlp_out"),
}
MAX_TOKENS_PER_DOC = 256

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 1. Load model ----
print("\nLoading Pythia 70m-deduped...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
model.eval()
print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ---- 2. Load corpus ----
print("\nLoading wikitext-2-raw-v1 document corpus...")
docs = []
with open("/Users/macn/Documents/embeddingexp/wikitext2_docs.jsonl") as f:
    for line in f:
        row = json.loads(line)
        text = row["page"].strip()
        if len(text) > 100:  # skip tiny / empty docs
            docs.append(text)
print(f"  {len(docs)} non-trivial documents loaded.")

# All hookpoints
all_hooks = set()
for substrate, (release, hook_template) in SUBSTRATES.items():
    for layer in LAYERS:
        all_hooks.add(hook_template.format(layer))
all_hooks = sorted(all_hooks)
print(f"  Hookpoints to capture ({len(all_hooks)}): {all_hooks[0]} ... {all_hooks[-1]}")

# ---- 3. Collect activations ----
print(f"\nCollecting activations from {len(docs)} documents (≤{MAX_TOKENS_PER_DOC} tokens each)...")
acts_by_hook = {h: [] for h in all_hooks}
total_tokens = 0
for idx, text in enumerate(docs):
    if idx % 100 == 0 and idx > 0:
        print(f"  {idx}/{len(docs)} docs, {total_tokens:,} tokens so far")
    tokens = model.to_tokens(text)
    if tokens.shape[1] > MAX_TOKENS_PER_DOC:
        tokens = tokens[:, :MAX_TOKENS_PER_DOC]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in all_hooks)
    for h in all_hooks:
        if h in cache:
            acts_by_hook[h].append(cache[h][0].cpu().float())
    total_tokens += tokens.shape[1]

print(f"\n  Collected {total_tokens:,} total token positions per hook.")

# Per-hook per-dim std
print("\nComputing per-hook activation std...")
sigma_by_hook = {}
for h in all_hooks:
    all_acts = torch.cat(acts_by_hook[h], dim=0)
    centered = all_acts - all_acts.mean(0)
    sigma_by_hook[h] = centered.std(0).numpy().astype(np.float64)
    sig = sigma_by_hook[h]
    print(f"  {h:35s}: N={all_acts.shape[0]:>7,}  σ range [{sig.min():.3f}, {sig.max():.3f}]  ratio {sig.max()/sig.min():.1f}")
    acts_by_hook[h] = None  # free memory

del model, acts_by_hook
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---- 4. Sweep (substrate, layer): PCA on decoder vs nulls ----
def participation_ratio(var_ratio):
    return float((var_ratio.sum() ** 2) / (var_ratio ** 2).sum())

def components_for(cum, frac):
    return int(np.searchsorted(cum, frac) + 1)

def pca_summary(W):
    W_centered = W - W.mean(0, keepdims=True)
    _, S, _ = np.linalg.svd(W_centered, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var)
    return participation_ratio(var), components_for(cum, 0.9), float(var[0])

results = {}
for substrate, (release, hook_template) in SUBSTRATES.items():
    for layer in LAYERS:
        hook = hook_template.format(layer)
        sae_id = hook
        print(f"\n--- {substrate} layer {layer} ({sae_id}) ---")
        try:
            sae_result = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
            sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        W_dec = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
        W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
        n_features, d_model = W_dec.shape
        sigma = sigma_by_hook[hook]

        pr_real, pcs90_real, top1_real = pca_summary(W_dec)

        rng = np.random.default_rng(0)
        R_aniso = rng.standard_normal((n_features, d_model)) * sigma[None, :]
        R_aniso = R_aniso / np.linalg.norm(R_aniso, axis=1, keepdims=True)
        pr_aniso, pcs90_aniso, top1_aniso = pca_summary(R_aniso)

        R_iso = rng.standard_normal((n_features, d_model))
        R_iso = R_iso / np.linalg.norm(R_iso, axis=1, keepdims=True)
        pr_iso, pcs90_iso, top1_iso = pca_summary(R_iso)

        results[(substrate, layer)] = {
            "n_features": n_features,
            "d_model": d_model,
            "pr_real": pr_real,
            "pr_aniso_null": pr_aniso,
            "pr_iso_null": pr_iso,
            "pcs90_real": pcs90_real,
            "pcs90_aniso_null": pcs90_aniso,
            "pcs90_iso_null": pcs90_iso,
            "top1_real": top1_real,
            "top1_aniso_null": top1_aniso,
            "top1_iso_null": top1_iso,
            "real_frac_of_iso": pr_real / pr_iso,
        }
        print(f"  PR: real={pr_real:6.1f}  aniso-null={pr_aniso:6.1f}  iso-null={pr_iso:6.1f}  "
              f"(real is {pr_real/pr_iso*100:5.1f}% of iso)")
        del sae, W_dec, R_aniso, R_iso
        gc.collect()


# ---- 5. Summary + save ----
print("\n" + "=" * 110)
print("SUMMARY: Pythia 70m-deduped, SAE decoder PCA -- with wikitext-2-raw covariance estimate")
print("=" * 110)
print(f"  {'substrate':<10s} {'layer':>5s}  {'d_model':>7s}  {'n_feat':>7s}  {'PR_real':>8s}  {'PR_aniso':>9s}  {'PR_iso':>7s}  {'real/iso':>9s}  {'PCs90_real':>11s}  {'top1_real':>10s}")
for (substrate, layer), r in sorted(results.items()):
    print(f"  {substrate:<10s} {layer:>5d}  {r['d_model']:>7d}  {r['n_features']:>7d}  "
          f"{r['pr_real']:>8.1f}  {r['pr_aniso_null']:>9.1f}  {r['pr_iso_null']:>7.1f}  "
          f"{r['real_frac_of_iso']*100:>8.1f}%  {r['pcs90_real']:>11d}  {r['top1_real']:>10.4f}")

out_path = "/Users/macn/Documents/embeddingexp/exp7c_sweep_wikitext_results.pt"
torch.save(results, out_path)
print(f"\nSaved to {out_path}")
