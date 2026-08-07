"""
exp156b_gpt2_zipf_norm.py — POST-HOC diagnostic (exp150b/c precedent):
fill the missing three-model-table cell — corr(zipf, residual norm) in
GPT-2 medium. References: Pythia +0.71 at L8+ (exp155); Llama ≈ 0 and
briefly NEGATIVE (−0.40 at L8) (exp152b). exp156's Δnorm went POSITIVE
at late layers, hinting GPT-2's norm regime may be inverted vs Pythia.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from wordfreq import zipf_frequency

from markedness_norm_protocol import build_word_lists

LAYERS = [4, 8, 12, 16, 20]
all_words, _, _ = build_word_lists()

model = HookedTransformer.from_pretrained("gpt2-medium", device="mps")
model.eval()
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

zipf = np.array([zipf_frequency(w, "en") for w in all_words])
norms = {L: [] for L in LAYERS}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    for L in LAYERS:
        norms[L].append(float(cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                              .float().norm()))
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")

print("\ncorr(zipf, residual norm) in GPT-2 medium")
print("(Pythia ref: +0.71 at L8+; Llama ref: ~0, −0.40 at its L8)")
for L in LAYERS:
    nv = np.array(norms[L])
    r = float(np.corrcoef(zipf, nv)[0, 1])
    rs = float(np.corrcoef(np.argsort(np.argsort(zipf)),
                           np.argsort(np.argsort(nv)))[0, 1])
    print(f"  L{L:>2}: corr(zipf, norm) = {r:+.3f}   (Spearman {rs:+.3f})   "
          f"mean norm {nv.mean():.1f}")
