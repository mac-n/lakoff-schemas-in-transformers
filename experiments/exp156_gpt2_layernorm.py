"""
exp156_gpt2_layernorm.py — the attribution control for exp152's
dissociation: GPT-2 medium (LayerNorm, 24 layers, ~355M) shares Pythia's
norm TYPE but nothing else (data, tokenizer, era, attention details).

After exp151 (Pythia 1.4B: full replication) and exp152 (Llama RMSNorm:
coupling and displacement absent, sink attenuated), the live question is
WHAT the Pythia/Llama difference tracks. GPT-2 discriminates:
  - coupling present in GPT-2 → follows norm type (LayerNorm side):
    "LayerNorm fosters concept-physiology alignment" survives its first
    real test (still not proof — GPT-2 and Pythia share more than LN —
    but the deflationary reading loses its cheapest version).
  - coupling absent in GPT-2 → follows the Pythia family, not the norm
    type: "LayerNorm fosters it" deflates to "Pythia does it";
    the paper's Part 2 claim must be about substrates-differ, not
    LN-vs-RMS mechanism.

Protocol: full exp154c held-out d_norm standard via
markedness_norm_protocol.py; selftest run first per convention.
gpt2-medium: 24 layers — probe [4,8,12,16,20] matches Pythia directly.
TL prepends BOS for GPT-2; last-token protocol is BOS-safe.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Where I actually stand after today's misses: my LN-recentering
  mechanism sketch predicts coupling here; the "Pythia-specific" reading
  predicts not. I commit to the mechanism I argued for this afternoon —
  if it's wrong I want it to die cleanly:
    P1 sink present: mean inflectional suffix x BALANCE <= -0.2 at
       decision layers [8,12,16,20].
    P2 displacement present: Δnorm < 0 for all five inflectional
       suffixes at decision layers, with magnitudes a substantial
       fraction of mean word norm (GPT-2's anisotropy/degeneration
       literature suggests strong frequency-norm structure).
    P3 coupling present: cos(BALANCE, d_norm_heldout) >= +0.5 at
       decision layers.
    P4 held-out residue in 15-60% (the two-component structure).
  Decision rule:
    G1 P1 & P2 & P3: norm-type story strengthens — LayerNorm models
       show the physiology, the RMSNorm model doesn't. Part 2 writes
       LN-vs-RMS as the candidate mechanism (with the shared-ancestry
       caveat: a Mistral/Qwen RMSNorm replication and/or an OLMo-style
       LN model would be the next tightening).
    G2 P3 fails (coupling absent, |cos_mid| < 0.3): Pythia-specific.
       Deflate the mechanism claim; Part 2 becomes "substrates differ,
       cause unknown" — still a real dissociation, weaker story.
    G3 mixed (e.g. displacement without coupling, or coupling without
       sink): report components; the mechanism story fragments and the
       paper should present the three-model table without a mechanism
       claim.
"""

import numpy as np
from transformer_lens import HookedTransformer

from markedness_norm_protocol import (
    build_word_lists, collect_residuals, analyze_layer, report_layer,
    verdict_components, selftest_verdict, INFL,
)

LAYERS = [4, 8, 12, 16, 20]
DECISION_LAYERS = [8, 12, 16, 20]

print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict(tuple(DECISION_LAYERS))

print("Loading GPT-2 medium...")
model = HookedTransformer.from_pretrained("gpt2-medium", device="mps")
model.eval()
assert model.cfg.n_layers == 24, f"expected 24 layers, got {model.cfg.n_layers}"
print(f"normalization_type = {model.cfg.normalization_type}")  # expect LN

all_words, est_words, test_words = build_word_lists()
print(f"Word split: {len(all_words)} total = {len(est_words)} estimation "
      f"+ {len(test_words)} held-out test")
print(f"Collecting residuals for {len(all_words)} words at {LAYERS}...")
residuals = collect_residuals(model, LAYERS, all_words)

print("\n" + "=" * 78)
print("exp156 — GPT-2 medium (LayerNorm) — held-out d_norm protocol")
print("=" * 78)

rng = np.random.default_rng(7)
summary = {}
for L in LAYERS:
    summary[L] = analyze_layer(residuals, L, all_words, est_words, test_words, rng)
    report_layer(L, summary[L])

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
v = verdict_components(summary, DECISION_LAYERS, sink_thresh=-0.2,
                       coupling_thresh=0.5)
cos_mid = float(np.mean([summary[L]["cos_bal_dnorm"] for L in DECISION_LAYERS]))
dnorm_all_neg = all(summary[L]["dnorm_suffix"][sn][0] < 0
                    for L in DECISION_LAYERS for sn in INFL)
print(f"  P1 sink (<= -0.2 all decision layers): {v['q1_sink']}")
print(f"  P2 inflectional Δnorm negative everywhere: {dnorm_all_neg}")
print(f"  P3 coupling: cos_mid = {cos_mid:+.3f} "
      f"(per layer: " + ", ".join(f"L{L}:{summary[L]['cos_bal_dnorm']:+.2f}"
                                  for L in DECISION_LAYERS) + ")")
print(f"  P4 residue 15-60%: {v['q3_residue']} "
      f"(mean retained = {100*v['mean_retained']:.0f}%)")
if v["q1_sink"] and dnorm_all_neg and v["q2_coupling"]:
    print("  -> G1: LayerNorm side confirmed on an unrelated LN model.")
    print("     Norm-type mechanism survives; write Part 2 with the")
    print("     shared-ancestry caveat and queue a second RMSNorm family.")
elif abs(cos_mid) < 0.3:
    print("  -> G2: coupling absent on GPT-2 — the physiology follows the")
    print("     Pythia family, NOT the norm type. Deflate the LN mechanism;")
    print("     Part 2 = 'substrates differ, cause unknown'.")
else:
    print("  -> G3: mixed/fragmented — present the three-model component")
    print("     table without a mechanism claim.")
