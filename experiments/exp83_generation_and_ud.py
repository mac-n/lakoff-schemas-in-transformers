"""
exp83 — Build target_GENERATION and test UD as candidate primitive.

Two related tests:

PART A — target_GENERATION construction.
  Anchor pairs spanning the cluster (possibility, actuality), (latent, expressed),
  (could-be, is), (option, choice), (alternative, decision), (undetermined,
  determined), (airy, grounded), (imaginable, instantiated), etc.
  Includes Niamh's "determine" addition and the air/ground verticality probe.

PART B — UD as candidate primitive.
  Build UP_DOWN axis from UP_DOWN_MML anchors. Test cos with full 12-axis
  basis and with GENERATION. Hypothesis: UD is a COMPOSITE that lexicalizes
  multiple primitives at once, which is why it captures so much Lakoff space.

Tests:
  1 — Build GENERATION + sanity (pole vocab, OOV check)
  2 — Cross-bleed: cos(GENERATION, 12-axis basis + UD)
  3 — cos(GENERATION, imagining_dir from exp81) — convergence with probe-derived direction
  4 — cos(GENERATION, TIME_PROTO) — Niamh's time-component prediction
  5 — Build UD axis; cos(UD, 12-axis basis); is UD a primitive or composite?
  6 — Pole vocab for GENERATION
  7 — Probe test: generation-status word pairs
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")

from project_axis_vocabulary import (
    TARGET_REWARD_COMPOSITE_PAIRS, TARGET_WEIGHT_PAIRS,
    ATTENTION_CLEAN_PAIRS, INTENTION_CLEAN_PAIRS,
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, TARGET_SURPRISAL_PAIRS,
    TARGET_DECISION_VERDICT_PAIRS, TARGET_MARKOV_BLANKET_PAIRS,
    TARGET_EPISTEMIC_VALUE_PAIRS,
    ABSTRACT_CONCRETE_PAIRS, REAL_IMAGINARY_PAIRS,
    TARGET_TIME_PROTO_PAIRS,
)
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN, UP_DOWN_MML
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(wv, pairs):
    used = []
    missing = []
    offs = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c))
    if not offs:
        return None, used, missing
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


def build_R(wv):
    eq, _, _ = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v, _, _ = build_axis(wv, VALENCE_PAIRS)
    a, _, _ = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ v) * v
    r = r - (r @ a) * a
    return unit(r)


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")

print("Building 12-axis basis...")
C, _, _   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W, _, _   = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT, _, _ = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT, _, _ = build_axis(wv, INTENTION_CLEAN_PAIRS)
R = build_R(wv)
D, _, _   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO, _, _  = build_axis(wv, IN_OUT_MML_CLEAN)
DV, _, _  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB, _, _  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV, _, _  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
ABS, _, _ = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
REAL_IMAG, _, _ = build_axis(wv, REAL_IMAGINARY_PAIRS)

basis_names = ["C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV", "ABS", "REAL_IMAG"]
basis_vecs  = [ C,   W,   ATT,   INT,   R,   D,   IO,   DV,   MB,   EV,   ABS,   REAL_IMAG ]


# ============================================================================
# PART A — Build target_GENERATION
# ============================================================================
print("\n" + "=" * 78)
print("PART A — Build target_GENERATION")
print("=" * 78)

GENERATION_PAIRS = [
    ("possibility",   "actuality"),
    ("potential",     "manifest"),
    ("latent",        "expressed"),
    ("uncommitted",   "committed"),
    ("could-be",      "is"),               # might be OOV (hyphenated)
    ("option",        "choice"),
    ("alternative",   "decision"),
    ("undetermined",  "determined"),
    ("airy",          "grounded"),         # air/ground verticality probe per Niamh
    ("imaginable",    "instantiated"),
]

GEN, used, missing = build_axis(wv, GENERATION_PAIRS)
print(f"\nUsed {len(used)}/{len(GENERATION_PAIRS)} pairs:")
for a, c in used:
    print(f"  ({a}, {c})")
if missing:
    print(f"\nMissing (OOV):")
    for a, c in missing:
        print(f"  ({a}, {c})")


# ============================================================================
# PART B — Cross-bleed of GENERATION with 12-axis basis
# ============================================================================
print("\n" + "=" * 78)
print("PART B — cos(GENERATION, 12-axis basis)")
print("=" * 78)

print(f"\n{'axis':<12}  {'cos(GEN, .)':>12}")
print("-" * 28)
gen_cosines = {}
for name, vec in zip(basis_names, basis_vecs):
    c = cos(GEN, vec)
    gen_cosines[name] = c
    print(f"{name:<12}  {c:>+12.4f}")

max_g = max(gen_cosines.items(), key=lambda kv: abs(kv[1]))
print(f"\nMax |cos| with basis: {abs(max_g[1]):.4f} (with {max_g[0]})")
if abs(max_g[1]) > 0.5:
    print("→ GENERATION is NOT a primitive — it's in the basis subspace")
elif abs(max_g[1]) > 0.35:
    print("→ GENERATION has substantial overlap; borderline; not a clean primitive")
else:
    print("→ GENERATION could be a 13th primitive (cleaner than expected)")


# ============================================================================
# PART C — cos(GENERATION, imagining_dir from exp81) — convergence test
# ============================================================================
print("\n" + "=" * 78)
print("PART C — Convergence with exp81's imagining direction")
print("=" * 78)

exp81 = np.load("exp81_results.npz")
mean_delta = exp81["mean_delta"]
print(f"\ncos(GENERATION, imagining_dir_exp81) = {cos(GEN, mean_delta):+.4f}")
print(f"cos(GENERATION, -imagining_dir_exp81) = {cos(GEN, -mean_delta):+.4f}")
print("(one of these should be high if GEN converges with the probe-derived direction)")


# ============================================================================
# PART D — cos(GENERATION, TIME_PROTO) — Niamh's time-component prediction
# ============================================================================
print("\n" + "=" * 78)
print("PART D — Time-component test")
print("=" * 78)

TIME, _, _ = build_axis(wv, TARGET_TIME_PROTO_PAIRS)
print(f"\ncos(GENERATION, TIME_PROTO) = {cos(GEN, TIME):+.4f}")
print("(Niamh's prediction: positive substantial, ~+0.3)")


# ============================================================================
# PART E — Build UD axis, test as candidate primitive
# ============================================================================
print("\n" + "=" * 78)
print("PART E — UD as candidate primitive")
print("=" * 78)

UD, _, _ = build_axis(wv, UP_DOWN_MML)
print(f"\n{'axis':<12}  {'cos(UD, .)':>12}")
print("-" * 28)
ud_cosines = {}
for name, vec in zip(basis_names + ["GENERATION"], basis_vecs + [GEN]):
    c = cos(UD, vec)
    ud_cosines[name] = c
    marker = ""
    if abs(c) > 0.4:
        marker = "  ←"
    print(f"{name:<12}  {c:>+12.4f}{marker}")

max_u = max(ud_cosines.items(), key=lambda kv: abs(kv[1]))
print(f"\nMax |cos| with basis (+ GEN): {abs(max_u[1]):.4f} (with {max_u[0]})")

# Compositionality test: project UD onto the basis, see what's left
print(f"\n--- Compositionality test: UD onto the 12-axis basis ---")
basis_matrix = np.stack(basis_vecs)
coeffs, res, rk, sv = np.linalg.lstsq(basis_matrix.T, UD, rcond=None)
fit = coeffs @ basis_matrix
residual = UD - fit
print(f"  UD = " + " + ".join([f"{coeffs[i]:+.3f}·{basis_names[i]}"
                                for i in range(12) if abs(coeffs[i]) > 0.1]))
print(f"  Residual magnitude: {np.linalg.norm(residual):.4f}")
print(f"  Fraction of UD in 12-axis basis: "
      f"{1 - (np.linalg.norm(residual) / np.linalg.norm(UD))**2 * 100/100:.4f}")
r2_ud = 1 - (residual @ residual) / (UD @ UD)
print(f"  R² (variance of UD in 12-axis basis): {r2_ud * 100:.1f}%")

if r2_ud > 0.5:
    print("\n  → UD is largely IN the 12-axis basis subspace")
    print("    UD is COMPOSITE — lexicalizes several primitives at once.")
    print("    Confirms hypothesis: UD's foundational status in Lakoff comes from")
    print("    being the multi-primitive composition language uses most.")
elif r2_ud > 0.3:
    print("\n  → UD has substantial basis content + significant residual.")
    print("    Partly composite, partly its own content.")
else:
    print("\n  → UD has independent content beyond the 12-axis basis.")
    print("    Could be added as a 13th primitive on its own merits.")


# ============================================================================
# PART F — Pole vocabulary
# ============================================================================
print("\n" + "=" * 78)
print("PART F — Pole vocabulary for GENERATION")
print("=" * 78)

print("\nGENERATION positive pole (uncollapsed / possibility side, top 12):")
for w, s in wv.similar_by_vector(GEN.astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")

print("\nGENERATION negative pole (collapsed / actualized side, top 12):")
for w, s in wv.similar_by_vector((-GEN).astype(np.float32), topn=12):
    print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# PART G — Probe test for GENERATION
# ============================================================================
print("\n" + "=" * 78)
print("PART G — Probe test: words that semantically differ in generation-status")
print("=" * 78)

probe_pairs = [
    ("uncertainty", "certainty"),
    ("hypothesis",  "fact"),
    ("draft",       "publication"),
    ("blueprint",   "building"),
    ("plan",        "execution"),
    ("intention",   "action"),
    ("suggestion",  "implementation"),
    ("candidate",   "winner"),
    ("nominee",     "elected"),
    ("rehearsal",   "performance"),
]

print(f"\n{'uncollapsed':<14}  {'collapsed':<14}  {'cos(unc, GEN)':>14}  {'cos(col, GEN)':>14}  Δ")
for unc, col in probe_pairs:
    if unc not in wv.key_to_index or col not in wv.key_to_index:
        continue
    v_unc = unit(wv[unc])
    v_col = unit(wv[col])
    c_unc = cos(v_unc, GEN)
    c_col = cos(v_col, GEN)
    marker = "  ← consistent" if c_unc > c_col else "  ← reversed"
    print(f"  {unc:<12}  {col:<14}  {c_unc:>+12.4f}    {c_col:>+12.4f}  {c_unc - c_col:>+.4f}{marker}")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp83_results.npz",
         GEN=GEN, UD=UD, TIME=TIME,
         gen_cosines={k: v for k, v in gen_cosines.items()},
         ud_cosines={k: v for k, v in ud_cosines.items()},
         ud_composition_coeffs=coeffs,
         ud_r2=r2_ud)
print("\nSaved exp83_results.npz")
