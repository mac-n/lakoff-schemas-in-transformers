"""
exp152b_llama_diagnostics.py — post-hoc diagnostics for exp152 (labelled
post-hoc per the exp150b/c precedent; no pre-registration, these are
sanity checks the A2 verdict mandated before anything gets written).

Q1 Does Llama-3.2-1B have frequency-norm coupling at all?
   corr(zipf, residual norm) per probe layer. Pythia reference: +0.71 at
   L8+ (exp155). If Llama lacks it, the whole norm-physiology regime
   (not just the BALANCE coupling) is absent there — a different and
   bigger statement than "no concept-physiology alignment".

Q2 Is the attenuated sink a tokenization artifact?
   Multi-token fraction + mean token count over the 489 protocol words
   and over the inflected suffix-pair forms specifically, for the Pythia
   and Llama tokenizers. If Llama splits the inflected forms much more
   aggressively, the last-token protocol measures suffix-token geometry
   rather than word geometry, and exp152's sink numbers are not
   commensurable with Pythia's.
"""

import numpy as np
import torch
from transformers import AutoTokenizer
from wordfreq import zipf_frequency

from markedness_norm_protocol import SUFFIX_PAIRS, build_word_lists

all_words, _, _ = build_word_lists()
inflected = sorted({i for pairs in SUFFIX_PAIRS.values() for _, i in pairs})

# ---- Q2 first (no model load needed) ----
print("=" * 78)
print("Q2 — tokenization comparison (words tokenized bare, no leading space,")
print("     matching the protocol's to_tokens usage; BOS excluded from counts)")
print("=" * 78)
for name, repo in [("Pythia", "EleutherAI/pythia-410m"),
                   ("Llama", "meta-llama/Llama-3.2-1B")]:
    tok = AutoTokenizer.from_pretrained(repo)
    def counts(words):
        n = [len(tok(w, add_special_tokens=False)["input_ids"]) for w in words]
        return np.array(n)
    c_all, c_inf = counts(all_words), counts(inflected)
    print(f"  {name:<7} all 489 words: multi-token {100*(c_all>1).mean():.0f}%, "
          f"mean {c_all.mean():.2f} tokens | "
          f"inflected forms: multi-token {100*(c_inf>1).mean():.0f}%, "
          f"mean {c_inf.mean():.2f} tokens")

# ---- Q1: frequency-norm coupling in Llama ----
print()
print("=" * 78)
print("Q1 — corr(zipf, residual norm) in Llama-3.2-1B (Pythia ref: +0.71 at L8+)")
print("=" * 78)
from transformer_lens import HookedTransformer

LAYERS = [3, 5, 8, 11, 13]
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device="mps")
model.eval()
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

zipf = {w: zipf_frequency(w, "en") for w in all_words}
norms = {L: [] for L in LAYERS}
words_used = []
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    for L in LAYERS:
        norms[L].append(float(cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                              .float().norm()))
    words_used.append(w)
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")

zv = np.array([zipf[w] for w in words_used])
for L in LAYERS:
    nv = np.array(norms[L])
    r = float(np.corrcoef(zv, nv)[0, 1])
    # also rank-based, in case the relation is monotone but nonlinear
    rs = float(np.corrcoef(np.argsort(np.argsort(zv)),
                           np.argsort(np.argsort(nv)))[0, 1])
    print(f"  L{L:>2}: corr(zipf, norm) = {r:+.3f}   (Spearman {rs:+.3f})   "
          f"mean norm {nv.mean():.2f}")
