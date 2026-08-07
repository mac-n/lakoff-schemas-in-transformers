"""
exp152_llama_rmsnorm.py — BALANCE-norm coupling on an RMSNorm model
(meta-llama/Llama-3.2-1B). THE discriminating test for the substrate
hypothesis: the whole grounding story now runs through the norm, and
RMSNorm rescales WITHOUT re-centering. Reserved as exp152 by the v5
next-steps section.

Protocol: full exp154c (held-out d_norm STANDARD per exp154c/d) via
markedness_norm_protocol.py (verdict logic synthetic-tested; the script
re-runs the selftest before loading the model).

Llama-3.2-1B: 16 layers, d_model 2048, RMSNorm, BOS IS prepended by the
tokenizer (unlike Pythia-in-TL) — collect_residuals indexes the last
token, which is BOS-safe. Probe layers fractional-depth-matched to the
Pythia runs: Pythia [4,8,12,16,20]/24 -> Llama [3,5,8,11,13]/16.

Needs gated-repo access: accept the license on the model page under the
HF account whose token is stored (hf auth login), else from_pretrained
raises 401/403.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed predictions:
    P1 the inflectional sink reproduces (<= -0.2 mean at decision
       layers) — markedness geometry should be a property of trained
       LMs, not of one family.
    P2 inflected forms still sit at LOWER residual norm (Δnorm < 0 for
       all five inflectional suffixes at decision layers) — frequency-
       norm coupling is general.
    P3 coupling present but WEAKER than Pythia: cos(BALANCE,
       d_norm_heldout) in +0.3..+0.5 at decision layers (vs Pythia's
       +0.64..+0.78). Reasoning: norm displacement is general (P2), but
       the tight concept-physiology alignment was hypothesised to be
       fostered by LayerNorm's re-centering geometry.
  Decision rule (components q1/q2/q3 reported independently;
  architecture verdict from the coupling magnitude, taking
  cos_mid = mean cos(BALANCE, d_norm_heldout) over decision layers):
    A1 cos_mid >= +0.5 (comparable to Pythia): coupling is NOT
       LayerNorm-specific — retreat to "rides norm-deviation
       generally" (handoff's weaker-but-publishable branch).
    A2 cos_mid <= +0.3 (incl. absent/negative): coupling is
       architecture-specific — Part 2 gets teeth. Check P2 before
       celebrating: if Δnorm also vanished, the story is "no norm
       displacement in Llama at all", which is a DIFFERENT (bigger)
       claim and needs its own controls (frequency first, exp155
       protocol).
    A3 +0.3 < cos_mid < +0.5: gradient, not dichotomy; report as such
       and let exp151 (1.4B) + this bracket the effect size.
  If P1 fails (no sink in Llama), all coupling questions are moot for
  the sink story but cos(BALANCE, d_norm) is still reported — the
  coupling claim is about axes, not the sink.
"""

import numpy as np
from transformer_lens import HookedTransformer

from markedness_norm_protocol import (
    build_word_lists, collect_residuals, analyze_layer, report_layer,
    verdict_components, selftest_verdict, INFL,
)

LAYERS = [3, 5, 8, 11, 13]
DECISION_LAYERS = [5, 8, 11, 13]

print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict(tuple(DECISION_LAYERS))

print("Loading Llama-3.2-1B (gated; needs hf auth + accepted license)...")
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device="mps")
model.eval()
assert model.cfg.n_layers == 16, f"expected 16 layers, got {model.cfg.n_layers}"
print(f"normalization_type = {model.cfg.normalization_type}")  # expect RMS

all_words, est_words, test_words = build_word_lists()
print(f"Word split: {len(all_words)} total = {len(est_words)} estimation "
      f"+ {len(test_words)} held-out test")
print(f"Collecting residuals for {len(all_words)} words at {LAYERS}...")
residuals = collect_residuals(model, LAYERS, all_words)

print("\n" + "=" * 78)
print("exp152 — Llama-3.2-1B (RMSNorm) — held-out d_norm protocol")
print("=" * 78)

rng = np.random.default_rng(7)
summary = {}
for L in LAYERS:
    summary[L] = analyze_layer(residuals, L, all_words, est_words, test_words, rng)
    report_layer(L, summary[L])

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
v = verdict_components(summary, DECISION_LAYERS, sink_thresh=-0.15,
                       coupling_thresh=0.5)
cos_mid = float(np.mean([summary[L]["cos_bal_dnorm"] for L in DECISION_LAYERS]))
dnorm_all_neg = all(summary[L]["dnorm_suffix"][sn][0] < 0
                    for L in DECISION_LAYERS for sn in INFL)
print(f"  q1 sink exists: {v['q1_sink']}")
print(f"  P2 inflectional Δnorm negative everywhere: {dnorm_all_neg}")
print(f"  coupling: cos_mid = {cos_mid:+.3f} "
      f"(per layer: " + ", ".join(f"L{L}:{summary[L]['cos_bal_dnorm']:+.2f}"
                                  for L in DECISION_LAYERS) + ")")
print(f"  q3 partial residue: {v['q3_residue']} "
      f"(mean retained = {100*v['mean_retained']:.0f}%)")
print(f"  EST flips positive after held-out strip (majority): {v['est_flip']}")
if cos_mid >= 0.5:
    print("  -> A1: coupling comparable to Pythia — NOT LayerNorm-specific.")
    print("     Retreat to 'rides norm-deviation generally'; still publishable.")
elif cos_mid <= 0.3:
    print("  -> A2: coupling weak/absent on RMSNorm — architecture-specific.")
    print("     Part 2 gets teeth. CHECK P2 above: if Δnorm also vanished,")
    print("     the claim changes shape (no displacement at all) — run the")
    print("     frequency control (exp155 protocol) before writing anything.")
else:
    print("  -> A3: intermediate coupling — gradient, not dichotomy. Report")
    print("     with exp151 to bracket the effect size.")
