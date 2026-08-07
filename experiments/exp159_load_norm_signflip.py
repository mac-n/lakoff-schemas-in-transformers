"""
exp159_load_norm_signflip.py — diagnostic (exploratory; no decision rule).
Why does corr(MLP load, residual norm) flip sign by layer in Pythia 410m
(exp158: +0.94 @L4, −0.99 @L8-16, +0.94 @L20), and does it happen in
other models?

Measures, at EVERY layer of three models, across the 489 protocol words:
  - corr(load, norm): load = L1(mlp.hook_post) last token;
    norm = ||resid_post|| last token; both token-count-controlled.
    (This scalar's SIGN is what cos(d_load,d_norm) tracked in exp158.)
  - mean residual norm per layer (locate each model's norm explosion).
Dense sampling settles "single dip vs aliased oscillation" and tests the
norm-regime hypothesis cross-model.

Models: pythia-410m (24L, LN, norm explodes ~L4-8), gpt2-medium (24L, LN),
Llama-3.2-1B (16L, RMSNorm — different norm regime; the discriminating case).

EXPLORATORY EXPECTATIONS (2026-06-11, before running; calibration only):
  E1 Pythia: single dip, zero-crossings flanking the norm explosion
     (positive pre-explosion, negative in the high-norm plateau,
     recovering late) — NOT a periodic wave.
  E2 GPT-2 (also pre-LN with norm growth): a qualitatively similar flip,
     located at ITS norm-growth region (not necessarily same layers).
  E3 Llama (RMSNorm, no centering, different norm behaviour): least sure
     — if the flip tracks the norm regime it should differ in shape or
     location; a flat/positive corr(load,norm) would support "the flip
     is a LayerNorm-massive-activation phenomenon".
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from markedness_norm_protocol import build_word_lists

all_words, _, _ = build_word_lists()
MODELS = [("pythia-410m", "pythia-410m"),
          ("gpt2-medium", "gpt2-medium"),
          ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B")]
results = {}


def tc_residualize(scalar_by_word, tc):
    v = np.array([scalar_by_word[w] for w in all_words], float)
    A = np.vstack([tc, np.ones_like(tc)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


for tag, repo in MODELS:
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    model = HookedTransformer.from_pretrained(repo, device="mps")
    model.eval()
    nL = model.cfg.n_layers
    rhooks = [f"blocks.{L}.hook_resid_post" for L in range(nL)]
    mhooks = [f"blocks.{L}.mlp.hook_post" for L in range(nL)]
    norm = {L: {} for L in range(nL)}
    load = {L: {} for L in range(nL)}
    tc = []
    for k, w in enumerate(all_words):
        toks = model.to_tokens(w)
        tc.append(toks.shape[1])
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=rhooks + mhooks)
        for L in range(nL):
            norm[L][w] = float(cache[f"blocks.{L}.hook_resid_post"][0, -1, :].float().norm())
            load[L][w] = float(cache[f"blocks.{L}.mlp.hook_post"][0, -1, :].float().abs().sum())
        if (k + 1) % 150 == 0:
            print(f"  {k+1}/{len(all_words)}")
    tc = np.array(tc, float)
    rows = []
    for L in range(nL):
        nr = tc_residualize(norm[L], tc)
        lr = tc_residualize(load[L], tc)
        r = float(np.corrcoef(lr, nr)[0, 1])
        mn = float(np.mean(list(norm[L].values())))
        rows.append((L, r, mn))
    results[tag] = rows
    print(f"  {'L':>3} {'corr(load,norm)':>16} {'mean_norm':>12}")
    for L, r, mn in rows:
        flag = "  <-- sign flip" if L > 0 and r * rows[L-1][1] < 0 else ""
        print(f"  {L:>3} {r:>+16.3f} {mn:>12.1f}{flag}")
    del model, cache, norm, load
    gc.collect(); torch.mps.empty_cache()

# ---- plot ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (tag, _) in zip(axes, MODELS):
    rows = results[tag]
    Ls = [r[0] for r in rows]
    corr = [r[1] for r in rows]
    mn = [r[2] for r in rows]
    ax.axhline(0, color="gray", lw=0.6)
    ax.plot(Ls, corr, "o-", color="tab:blue", label="corr(load, norm)")
    ax.set_xlabel("layer"); ax.set_ylim(-1.05, 1.05)
    ax.set_title(tag); ax.set_ylabel("corr(load, norm)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(Ls, mn, "s--", color="tab:red", alpha=0.6, label="mean norm")
    ax2.set_ylabel("mean residual norm", color="tab:red")
plt.suptitle("exp159 — corr(MLP load, residual norm) across depth (tc-controlled)")
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp159_load_norm_signflip.png", dpi=120)
print("\nSaved exp159_load_norm_signflip.png")

# ---- shape summary ----
print("\n" + "=" * 70)
print("SHAPE SUMMARY (sign changes = candidate 'wave' vs single dip)")
print("=" * 70)
for tag, _ in MODELS:
    corr = [r[1] for r in results[tag]]
    flips = sum(1 for L in range(1, len(corr)) if corr[L]*corr[L-1] < 0)
    print(f"  {tag:<14} {flips} sign change(s); "
          f"range [{min(corr):+.2f}, {max(corr):+.2f}]")
