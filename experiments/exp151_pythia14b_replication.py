"""
exp151_pythia14b_replication.py — is the markedness/norm story a Pythia
410M quirk? Full exp154c protocol (held-out d_norm STANDARD) on Pythia
1.4B (24 layers, d_model 2048 — same depth as 410M, so the same probe
layers transfer directly).

Reserved as exp151 by the v5 next-steps section. Uses
markedness_norm_protocol.py (shared module extracted 2026-06-11; its
verdict logic is synthetic-tested per the new convention — run
`python markedness_norm_protocol.py` to verify before trusting this).

Reports per layer: sink before strip, after FULL-set strip (for
comparability with exp154's original numbers), after HELD-OUT strip (the
honest number per exp154c/d), random-2D band, coupling cosine, Δnorm
table. Also checks the EST sign-flip burrow for free (BURROWS.md
2026-06-10).

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed predictions:
    P1 the inflectional BALANCE sink reproduces: mean inflectional
       suffix x BALANCE <= -0.25 at L8-L20.
    P2 the concept-physiology coupling reproduces:
       cos(BALANCE, d_norm_heldout) >= +0.5 at L8-L20.
    P3 the two-component structure reproduces: held-out strip retains
       20-45% at L8+ while the full-set strip collapses to <= ~10%
       (the circularity gap should appear here too).
    P4 EST x BALANCE flips positive after held-out strip at a majority
       of decision layers (the 410M flip is real, not noise).
  Decision rule (decision layers L8, L12, L16, L20; components reported
  independently via verdict_components):
    R1 q1 AND q2 AND q3: full replication — 410M findings generalise
       within the Pythia family; paper claims stand on both sizes.
    R2 NOT q1: the sink itself is a 410M quirk — major caveat, the
       grounding chain needs rework before exp152 means anything.
    R3 q1 AND NOT q2: sink without coupling — the norm-grounding is
       size-specific; report and re-examine exp141 on 1.4B.
    R4 q1 AND q2 AND NOT q3: sink + coupling but residue structure
       differs (full collapse OR full survival) — report retained
       fraction; the two-component story is size-dependent.
  P4 is reported either way and settles the EST burrow's kill condition.
"""

import numpy as np
from transformer_lens import HookedTransformer

from markedness_norm_protocol import (
    build_word_lists, collect_residuals, analyze_layer, report_layer,
    verdict_components, selftest_verdict,
)

LAYERS = [4, 8, 12, 16, 20]
DECISION_LAYERS = [8, 12, 16, 20]

print("Running verdict-logic selftest first (synthetic-test convention)...")
selftest_verdict(tuple(DECISION_LAYERS))

print("Loading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device="mps")
model.eval()
assert model.cfg.n_layers == 24, f"expected 24 layers, got {model.cfg.n_layers}"

all_words, est_words, test_words = build_word_lists()
print(f"Word split: {len(all_words)} total = {len(est_words)} estimation "
      f"+ {len(test_words)} held-out test")
print(f"Collecting residuals for {len(all_words)} words at {LAYERS}...")
residuals = collect_residuals(model, LAYERS, all_words)

print("\n" + "=" * 78)
print("exp151 — Pythia 1.4B replication (held-out d_norm protocol)")
print("=" * 78)

rng = np.random.default_rng(7)
summary = {}
for L in LAYERS:
    summary[L] = analyze_layer(residuals, L, all_words, est_words, test_words, rng)
    report_layer(L, summary[L])

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
v = verdict_components(summary, DECISION_LAYERS)
print(f"  q1 sink exists (<= -0.15 all decision layers): {v['q1_sink']}")
print(f"  q2 coupling exists (cos >= +0.5 all decision layers): {v['q2_coupling']}")
print(f"  q3 partial residue (mean retained in [15%, 60%]): {v['q3_residue']} "
      f"(mean retained = {100*v['mean_retained']:.0f}%)")
print(f"  P4 EST flips positive after held-out strip (majority): {v['est_flip']}")
if v["q1_sink"] and v["q2_coupling"] and v["q3_residue"]:
    print("  -> R1: full replication. 410M findings generalise within the")
    print("     Pythia family; paper claims stand on both sizes.")
elif not v["q1_sink"]:
    print("  -> R2: NO SINK on 1.4B — 410M quirk. Major caveat; rework the")
    print("     grounding chain before exp152.")
elif not v["q2_coupling"]:
    print("  -> R3: sink without coupling — norm-grounding is size-specific.")
    print("     Re-examine exp141 on 1.4B.")
else:
    print("  -> R4: sink + coupling replicate; residue structure differs.")
    print("     Two-component story is size-dependent; report retained fraction.")
