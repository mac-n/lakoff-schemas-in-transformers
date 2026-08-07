"""
exp157b_singletoken_control.py — is the step-0 corr(zipf, norm) = +0.54
a learned physiology or a tokenization artifact?

exp157 found corr(zipf, residual norm) = +0.543 at Pythia 410M random
init (step 0), before any training. The residuals are LAST-TOKEN; frequent
words tend to be single-token, rarer words multi-token, so last-token
residual norm could correlate with frequency purely via tokenization /
which token is last — a measurement confound, not physiology. This
touches a load-bearing number (the "+0.71 frequency-norm regime" cited in
exp155/152b as the substrate Pythia grounds BALANCE in).

Design: recompute corr(zipf, norm) at step 0 AND step 143000 (final),
three ways:
  (1) ALL words (reproduces exp157's number — parity anchor)
  (2) SINGLE-TOKEN words only (the confound-free subset)
  (3) ALL words, but regressing token-count out of norm first (uses all
      data, removes the linear token-count component)
If (2)/(3) at step 0 collapse toward ~0 while step-143000 stays high,
the LEARNED frequency-norm coupling is real and we cite the controlled
number. If step-0 stays high in (2)/(3) too, a real chunk of the regime
is architectural-at-init and the exp155/152b framing needs a footnote.

POST-HOC control (labelled per exp150b/c precedent — no point prediction;
this is a confound check, not a hypothesis test). Reported expectation
only: I expect single-token step-0 corr to drop substantially (the
confound is real) but am genuinely unsure if it goes all the way to 0.
"""

import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from transformer_lens import HookedTransformer
from wordfreq import zipf_frequency

from markedness_norm_protocol import build_word_lists

LAYERS = [4, 8, 12, 16, 20]
all_words, _, _ = build_word_lists()
zipf_all = np.array([zipf_frequency(w, "en") for w in all_words])


def analyze(model, tag):
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    tok_counts = np.array([model.to_tokens(w).shape[1] - 1  # minus BOS
                           for w in all_words])
    single = tok_counts == 1
    norms = {L: [] for L in LAYERS}
    for w in all_words:
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_names)
        for L in LAYERS:
            norms[L].append(float(cache[f"blocks.{L}.hook_resid_post"][0, -1, :]
                                  .float().norm()))
    print(f"\n--- {tag} ---  (single-token: {single.sum()}/{len(all_words)} words)")
    print(f"  {'L':>3} {'all':>8} {'single-tok':>11} {'tokcount-resid':>15}")
    for L in LAYERS:
        nv = np.array(norms[L])
        r_all = np.corrcoef(zipf_all, nv)[0, 1]
        r_single = np.corrcoef(zipf_all[single], nv[single])[0, 1]
        # regress token count out of norm, then correlate residual with zipf
        A = np.vstack([tok_counts, np.ones_like(tok_counts)]).T.astype(float)
        coef, *_ = np.linalg.lstsq(A, nv, rcond=None)
        nv_resid = nv - A @ coef
        r_resid = np.corrcoef(zipf_all, nv_resid)[0, 1]
        print(f"  {L:>3} {r_all:>+8.3f} {r_single:>+11.3f} {r_resid:>+15.3f}")


print("Loading Pythia 410M step 0 (random init)...")
m0 = HookedTransformer.from_pretrained("pythia-410m", checkpoint_value=0, device="mps")
m0.eval()
analyze(m0, "step 0 (random init)")
del m0; import gc; gc.collect(); torch.mps.empty_cache()

print("\nLoading Pythia 410M final (step 143000)...")
mf = HookedTransformer.from_pretrained("pythia-410m", device="mps")
mf.eval()
analyze(mf, "step 143000 (final)")
