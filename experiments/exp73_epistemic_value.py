"""
exp73 — target_EPISTEMIC_VALUE construction (Thread NEW, post-exp70 refinement).

Active-inference reading: the second term of expected free energy.
  EFE(π) = pragmatic value − cost − epistemic value
where epistemic value drives exploration (information gain about hidden states
and model parameters). With C, W already in the basis, EPISTEMIC_VALUE would
complete the EFE-decomposition triplet at the basis level.

Anchor choice is state-based per exp70's finding:
  - State-based curiosity vocab (curious, intrigued, fascinated, ...) fits the
    (+ATT, −INT) "explore mode" quadrant.
  - Activity-vocab (investigating, exploring, probing, ...) loads on INT
    because these are committed-to-action verbs in language.
Use state-based only to avoid the INT cross-bleed exp70 revealed.

Predictions (committed by current Claude in this session's transcript):
  - cos(EV, ATT) in [+0.10, +0.30]
  - cos(EV, INT) in [-0.10, +0.10]    (near zero; >+0.20 = screening failed)
  - cos(EV, C)   in [+0.10, +0.25]
  - cos(EV, D)   in [+0.15, +0.30]
  - max |cos| with basis in [0.25, 0.35], most likely on D or C
Falsifiers:
  - max |cos| > 0.40 → derived-state, not primitive
  - max |cos| < 0.15 → unusually clean primitive
  - cos(EV, INT) > +0.20 → activity-verb leakage

Tests:
  1 — Build target_EPISTEMIC_VALUE + pole vocabulary sanity check
  2 — cos(EV, full 9-axis basis) — predictions check + verdict
  3 — cos(EV, Lakoff schemas) for context
  4 — Concept-word projections of curiosity/wonder/awe/inquiry/etc
  5 — Verdict: 10th-axis primitive, derived state, or other
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(pairs):
    offs, used, missing = [], [], []
    for a, c in pairs:
        in_a = a in wv.key_to_index
        in_c = c in wv.key_to_index
        if in_a and in_c:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c, in_a, in_c))
    if not offs:
        return None, used, missing
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


# Load saved basis axes
print("Loading exp60 + exp64 + exp69 saved arrays...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp64 = np.load("exp64_results.npz", allow_pickle=True)
exp69 = np.load("exp69_results.npz", allow_pickle=True)

C   = unit(basis_raw["C_rew"])
W   = unit(basis_raw["W_wgt"])
R   = unit(basis_raw["R_per"])
D   = unit(basis_raw["D_cmp"])
IO  = unit(basis_raw["IO_blk"])
ATT = unit(exp69["ATTENTION_CLEAN"])
INT = unit(exp69["INTENTION_CLEAN"])
DV  = unit(exp64["GATING"])
MB  = unit(exp64["MARKOV_BLANKET"])


# ============================================================================
# TEST 1 — Build target_EPISTEMIC_VALUE
# ============================================================================
print("\n" + "=" * 72)
print("TEST 1 — Build target_EPISTEMIC_VALUE (state-based anchors)")
print("=" * 72)

# State-based curiosity vocabulary. NO activity-verbs (investigate, explore,
# probe, inquire — those load on INT per exp70). Pairs aim at the
# "epistemic-engagement state" register.
EPISTEMIC_PAIRS = [
    ("curious",     "indifferent"),
    ("intrigued",   "dismissive"),
    ("fascinated",  "bored"),
    ("inquisitive", "incurious"),
    ("puzzled",     "settled"),
    ("wondering",   "knowing"),
    ("marveling",   "dismissing"),
    ("awestruck",   "jaded"),
    ("mystified",   "certain"),
    ("engaged",     "blase"),       # blasé without diacritic
]

EV, used, missing = build_axis(EPISTEMIC_PAIRS)
print(f"\nUsed {len(used)}/{len(EPISTEMIC_PAIRS)} pairs:")
for a, c in used:
    print(f"  ({a}, {c})")
if missing:
    print(f"\nMissing (OOV):")
    for a, c, in_a, in_c in missing:
        flag = []
        if not in_a:
            flag.append(f"'{a}' OOV")
        if not in_c:
            flag.append(f"'{c}' OOV")
        print(f"  ({a}, {c})  — {'; '.join(flag)}")

print("\nEPISTEMIC_VALUE positive pole (curious-state side, top 15):")
for w_, s in wv.similar_by_vector(EV.astype(np.float32), topn=15):
    print(f"  {w_:25s}  {s:+.4f}")

print("\nEPISTEMIC_VALUE negative pole (incurious side, top 15):")
for w_, s in wv.similar_by_vector((-EV).astype(np.float32), topn=15):
    print(f"  {w_:25s}  {s:+.4f}")


# ============================================================================
# TEST 2 — cos(EV, full 9-axis basis) + predictions check
# ============================================================================
print("\n" + "=" * 72)
print("TEST 2 — cos(EV, 9-axis basis): predictions check")
print("=" * 72)

basis_axes = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT),
              ("R", R), ("D", D), ("IO", IO), ("DV", DV), ("MB", MB)]

predictions = {
    "C":   (+0.10, +0.25),
    "ATT": (+0.10, +0.30),
    "INT": (-0.10, +0.10),
    "D":   (+0.15, +0.30),
}

print(f"\n{'axis':<6}  {'cos(EV, .)':>12}  {'predicted':>16}  result")
print("-" * 60)
all_cos = {}
for name, vec in basis_axes:
    c = cos(EV, vec)
    all_cos[name] = c
    if name in predictions:
        lo, hi = predictions[name]
        in_range = "HIT" if lo <= c <= hi else "MISS"
        pred_str = f"[{lo:+.2f}, {hi:+.2f}]"
        print(f"{name:<6}  {c:>+12.4f}  {pred_str:>16}  {in_range}")
    else:
        print(f"{name:<6}  {c:>+12.4f}  {'(no prediction)':>16}")

max_abs = max(abs(v) for v in all_cos.values())
max_axis = max(all_cos.items(), key=lambda kv: abs(kv[1]))[0]
print(f"\nmax |cos| with basis = {max_abs:.3f} (with {max_axis})")
print(f"  Predicted: [0.25, 0.35], most likely D or C")

# Verdict on primitive status
print()
if max_abs > 0.40:
    print("→ DERIVED STATE: max |cos| > 0.40. EV is largely captured by existing primitives.")
elif max_abs < 0.15:
    print("→ UNUSUALLY CLEAN PRIMITIVE: max |cos| < 0.15.")
elif 0.15 <= max_abs <= 0.40:
    print(f"→ PRIMITIVE CANDIDATE: max |cos| = {max_abs:.3f} in the clean range.")
    print(f"  EV would be a 10th-axis primitive on the 0.35 threshold.")

# Critical screening check
if all_cos["INT"] > 0.20:
    print(f"\n⚠ SCREENING FAILED: cos(EV, INT) = {all_cos['INT']:+.3f} > +0.20.")
    print(f"   Activity-verb content leaked through despite state-based anchor choice.")
elif all_cos["INT"] > 0.10:
    print(f"\n⚠ SCREENING PARTIAL: cos(EV, INT) = {all_cos['INT']:+.3f} above expected near-zero.")
else:
    print(f"\n✓ INT screening clean: cos(EV, INT) = {all_cos['INT']:+.3f}")


# ============================================================================
# TEST 3 — cos(EV, Lakoff schemas)
# ============================================================================
print("\n" + "=" * 72)
print("TEST 3 — cos(EV, Lakoff schemas)")
print("=" * 72)

schemas = {
    "VALENCE": build_axis(VALENCE_PAIRS)[0],
    "AROUSAL": build_axis(AROUSAL_PAIRS)[0],
    "COH":     build_axis(COHERENCE_PAIRS)[0],
    "SUC":     build_axis(SUCCESS_FAILURE_PAIRS)[0],
    "LOSS":    build_axis(LOSS_PAIRS)[0],
    "UD":      build_axis(UP_DOWN_MML)[0],
    "FB":      build_axis(FORWARD_BACK_MML)[0],
    "LD":      build_axis(LIGHT_DARK_MML)[0],
    "PATH":    build_axis(PATH_MOTION_MML)[0],
    "EXIST":   build_axis(EXISTENCE_MML)[0],
    "FORCE":   build_axis(FORCE_MML)[0],
    "BAL":     build_axis(BALANCE_MML)[0],
    "DIFF":    build_axis(DIFFICULTY_BURDEN_MML)[0],
}

print(f"\n{'schema':<10}  {'cos(EV, .)':>12}")
for n, s in schemas.items():
    print(f"{n:<10}  {cos(EV, s):>+12.4f}")


# ============================================================================
# TEST 4 — concept-word projections: curiosity/wonder/awe/inquiry
# ============================================================================
print("\n" + "=" * 72)
print("TEST 4 — Concept words onto basis + EV")
print("=" * 72)

# Curiosity-cluster concept words (a mix of state and activity)
ev_probe_words = [
    # Pure-state words (predicted to load on EV)
    "curious", "intrigued", "fascinated", "inquisitive", "puzzled",
    "wondering", "wonder", "awe", "awestruck", "marveling",
    "mystified", "engaged", "interested",
    # Activity-verbs (predicted to load on INT, not EV)
    "investigating", "exploring", "probing", "inquiring", "searching",
    "seeking", "studying", "examining",
    # Drift-cluster (predicted to load on neither EV nor INT, low both)
    "daydreaming", "drifting", "lethargic", "absent-minded",
    "unfocused", "boredom",
    # Mixed / other
    "curiosity", "interest", "fascination", "wonderment",
    "indifference", "disinterest",
]

print(f"\n{'word':<18} " + " ".join(f"{n:>7}" for n in ["EV", "ATT", "INT", "C", "D"]))
print("-" * 60)
for w_ in ev_probe_words:
    if w_ not in wv.key_to_index:
        print(f"  {w_:<16} (not in GloVe)")
        continue
    v = unit(wv[w_])
    row = f"  {w_:<16}"
    for ax_name, ax_vec in [("EV", EV), ("ATT", ATT), ("INT", INT),
                            ("C", C), ("D", D)]:
        row += f" {cos(v, ax_vec):>+7.3f}"
    print(row)


# ============================================================================
# TEST 5 — Where does EV sit relative to the (ATT, INT) plane from exp70?
# ============================================================================
print("\n" + "=" * 72)
print("TEST 5 — Where does EV sit relative to exp70's EXPLORATION direction?")
print("=" * 72)

EXPLORATION = unit(ATT - INT)
print(f"\nEXPLORATION = unit(ATT − INT)  (exp70's derived-state direction)")
print(f"  cos(EV, EXPLORATION) = {cos(EV, EXPLORATION):+.4f}")
print(f"  cos(EV, ATT)         = {cos(EV, ATT):+.4f}")
print(f"  cos(EV, INT)         = {cos(EV, INT):+.4f}")
print()
print("  If cos(EV, EXPL) is high (>+0.5), EV mostly captures the (+ATT, −INT)")
print("    quadrant — supporting Niamh's original derived-state hypothesis.")
print("  If cos(EV, EXPL) is moderate-low (<+0.3), EV captures something the")
print("    (ATT − INT) direction misses — the 'what drives exploration' content")
print("    that distinguishes active novelty-seeking from passive drift.")


# ============================================================================
# Save
# ============================================================================
np.savez("/Users/macn/Documents/embeddingexp/exp73_results.npz",
         EPISTEMIC_VALUE=EV,
         cos_with_basis=np.array([all_cos[n] for n, _ in basis_axes]),
         basis_names=np.array([n for n, _ in basis_axes]),
         cos_EV_EXPLORATION=cos(EV, EXPLORATION))
print("\nSaved exp73_results.npz")
