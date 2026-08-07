"""
exp52: Direct target-axis validation of word2vec PCs from exp40.

The writeup names the top 3 cluster PCs as:
  PC1 (21%) = SALIENCE (Russell's affect diagonal)
  PC2 (17%) = MOTION (going vs holding)
  PC3 (12%) = EQUILIBRIUM-vs-RUNAWAY (small accountability vs amplifying)

But the names were INFERRED from nearest-neighbor lookups, not directly tested.
This experiment builds three independent target axes from anchor pairs that
DO NOT OVERLAP with any input-axis vocabulary, then projects them onto the
PC directions to test whether the named primitives recover.

Methodological notes:
  - target_SALIENCE: Niamh's reframe. The Russell-diagonal version of salience
    is by construction valence-loaded (V x A combination). The cleaner question
    is whether the model has a salience/attention/importance axis that's
    ORTHOGONAL to valence. Anchors are intentionally valence-ambiguous
    (important things can be wonderful or terrible).
  - target_MOTION: locomotion-verbs vs body-position-state-verbs. Avoids all
    PATH vocabulary (moving/stationary/traveled/remained/etc) and all spatial-
    direction vocabulary.
  - target_EQUILIBRIUM_RUNAWAY: control-theoretic feedback-regulation vs
    positive-feedback cascade. Anchors drawn from regulation + harm-restoration
    vocabularies that don't appear in any existing axis.

Predictions:
  - If PC1 is salience-as-primitive (orthogonal-to-valence): target_SALIENCE
    should land cleanly on PC1.
  - If PC1 is just the V x A diagonal: target_SALIENCE should be LOW on PC1
    and may not land on any clean direction.
  - target_MOTION should land on PC2 if MOTION is real.
  - target_EQUILIBRIUM_RUNAWAY should land on PC3 if EQ-vs-RUN is real.

Off-diagonal hits are also informative.

Niamh's deeper hypothesis on PC3: it may operationalize "regulability" — the
structural axis distinguishing self-regulating dynamic systems from runaway
positive-feedback systems. Maps onto Polyvagal ventral-vagal vs unbounded
sympathetic activation; maps onto coherent generation vs K-shaped collapse
in LLM steering experiments (exp4b, exp17). If PC3 is real and target-validates,
it elevates from "novel cognitive primitive" to "candidate regulation axis."
"""
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)


# ============================================================
# Anchor pairs — reused from exp40 (V, A, COH, SUC, LOSS)
# ============================================================
VALENCE_PAIRS = [
    ("pleasant", "unpleasant"), ("desirable", "undesirable"),
    ("agreeable", "disagreeable"), ("enjoyable", "distasteful"),
    ("delightful", "awful"), ("beneficial", "harmful"),
    ("wonderful", "terrible"), ("excellent", "dreadful"),
    ("favorable", "unfavorable"), ("satisfying", "frustrating"),
    ("nice", "nasty"), ("kind", "cruel"),
]
AROUSAL_PAIRS = [
    ("intense", "mild"), ("intense", "gentle"),
    ("alert", "drowsy"), ("urgent", "leisurely"),
    ("frantic", "tranquil"), ("energetic", "lethargic"),
    ("aroused", "relaxed"), ("sharp", "dull"),
    ("acute", "subtle"), ("vivid", "faint"),
    ("electric", "placid"), ("turbulent", "still"),
]
COHERENCE_PAIRS = [
    ("coherent", "incoherent"), ("consistent", "inconsistent"),
    ("aligned", "misaligned"), ("ordered", "disordered"),
    ("organized", "disorganized"), ("harmonious", "discordant"),
    ("predictable", "surprising"), ("predictable", "unpredictable"),
    ("expected", "unexpected"), ("ordinary", "anomalous"),
    ("regular", "irregular"), ("normal", "aberrant"),
    ("orderly", "chaotic"), ("structured", "unstructured"),
    ("uniform", "erratic"),
]
SUCCESS_FAILURE_PAIRS = [
    ("win", "lose"), ("won", "lost"), ("winning", "losing"),
    ("succeed", "fail"), ("succeeded", "failed"), ("success", "failure"),
    ("successful", "unsuccessful"), ("score", "miss"),
    ("scored", "missed"), ("correct", "incorrect"),
    ("accomplish", "fail"), ("achievement", "failure"),
    ("triumph", "defeat"), ("victory", "defeat"),
    ("pass", "fail"), ("passed", "failed"),
    ("hit", "miss"), ("reward", "punishment"),
]
LOSS_PAIRS = [
    ("gain", "loss"), ("gained", "lost"), ("gaining", "losing"),
    ("profit", "loss"), ("abundance", "scarcity"), ("fortune", "misfortune"),
    ("security", "threat"), ("safety", "danger"), ("wealth", "poverty"),
    ("plenty", "lack"), ("prosperity", "ruin"), ("surplus", "deficit"),
    ("having", "lacking"), ("acquired", "deprived"), ("rich", "poor"),
    ("affluent", "destitute"), ("secure", "vulnerable"),
    ("protected", "exposed"), ("safe", "endangered"),
]


# ============================================================
# NEW target-axis anchor pairs
# ============================================================

# target_SALIENCE (Niamh's reframe): attention-allocation / importance,
# intentionally orthogonal to valence. Each pair is valence-ambiguous —
# important things can be wonderful or terrible, attention can be drawn to
# beauty or threat. Tests whether salience-as-primitive (separate from
# Russell's V x A diagonal) is recoverable.
TARGET_SALIENCE_PAIRS = [
    ("important",       "unimportant"),
    ("urgent",          "idle"),               # "relaxed" would have been in AROUSAL
    ("salient",         "irrelevant"),
    ("attentive",       "inattentive"),
    ("focused",         "unfocused"),
    ("directed",        "diffuse"),
    ("prominent",       "inconspicuous"),
    ("noticeable",      "unnoticeable"),
    ("foregrounded",    "backgrounded"),
    ("highlighted",     "overlooked"),
    ("conspicuous",     "unobtrusive"),
    ("pronounced",      "muted"),
]

# target_MOTION: locomotion-verbs (going-through-space) vs body-position-state-
# verbs (holding-pose). Avoids all PATH/spatial-direction vocabulary.
TARGET_MOTION_PAIRS = [
    ("running",   "sitting"),
    ("jogging",   "standing"),
    ("swimming",  "lying"),
    ("flying",    "crouching"),
    ("skating",   "kneeling"),
    ("sliding",   "perching"),
    ("rolling",   "slouching"),
    ("leaping",   "leaning"),
    ("vaulting",  "lounging"),
    ("springing", "reclining"),
    ("lunging",   "hunching"),
    ("galloping", "slumping"),
]

# target_EQUILIBRIUM_RUNAWAY: control-theoretic feedback-regulation vs
# positive-feedback cascade. Anchors drawn from regulation + harm-restoration
# vocabularies that don't appear in any existing axis.
TARGET_EQUILIBRIUM_RUNAWAY_PAIRS = [
    ("correcting",    "escalating"),
    ("adjusting",     "cascading"),
    ("recalibrating", "snowballing"),
    ("righting",      "spiraling"),
    ("stabilizing",   "mushrooming"),
    ("regulating",    "ballooning"),
    ("moderating",    "surging"),
    ("tempering",     "propagating"),
    ("dampening",     "amplifying"),
    ("restraining",   "intensifying"),
    ("restoring",     "exploding"),
    ("atoning",       "raging"),
    ("mending",       "festering"),
    ("reconciling",   "ravaging"),
]


# ============================================================
# Load embeddings + build axes
# ============================================================

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")
print(f"  vocab size: {len(wv.key_to_index)}")


def build_axis(pairs, label=""):
    offs = []
    missing = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            for w in (a, c):
                if w not in wv.key_to_index:
                    missing.append(w)
    if missing:
        print(f"  [{label}] missing words ({len(missing)}): {sorted(set(missing))}")
    if not offs:
        raise ValueError(f"No pairs survived OOV filtering for {label}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw), len(offs)


print("\nBuilding PCA-input axes (cluster set, 11 axes from exp40)...")
axes = {}
axes["VALENCE"],   _ = build_axis(VALENCE_PAIRS, "VALENCE")
axes["AROUSAL"],   _ = build_axis(AROUSAL_PAIRS, "AROUSAL")
axes["UD"],        _ = build_axis(UP_DOWN_MML, "UD")
axes["FB"],        _ = build_axis(FORWARD_BACK_MML, "FB")
axes["LD"],        _ = build_axis(LIGHT_DARK_MML, "LD")
axes["PATH"],      _ = build_axis(PATH_MOTION_MML, "PATH")
axes["EXIST"],     _ = build_axis(EXISTENCE_MML, "EXIST")
axes["BAL"],       _ = build_axis(BALANCE_MML, "BAL")
axes["COHERENCE"], _ = build_axis(COHERENCE_PAIRS, "COHERENCE")
axes["SUCCESS"],   _ = build_axis(SUCCESS_FAILURE_PAIRS, "SUCCESS")
axes["LOSS"],      _ = build_axis(LOSS_PAIRS, "LOSS")

axis_names = list(axes.keys())
M = np.stack([axes[n] for n in axis_names])  # (11, 300)


print("\nBuilding target axes (independent from PCA-input set)...")
target_sal, n_sal = build_axis(TARGET_SALIENCE_PAIRS, "target_SALIENCE")
target_mot, n_mot = build_axis(TARGET_MOTION_PAIRS, "target_MOTION")
target_eqr, n_eqr = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, "target_EQ_RUN")
print(f"  target_SALIENCE built from {n_sal}/{len(TARGET_SALIENCE_PAIRS)} pairs")
print(f"  target_MOTION    built from {n_mot}/{len(TARGET_MOTION_PAIRS)} pairs")
print(f"  target_EQ_RUN    built from {n_eqr}/{len(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)} pairs")

targets = {
    "target_SALIENCE":  target_sal,
    "target_MOTION":    target_mot,
    "target_EQ_RUN":    target_eqr,
}


# ============================================================
# PCA over cluster axes (replicating exp40's cluster-only run)
# ============================================================

print("\n" + "="*72)
print("PCA on cluster axes (11 axes — matches exp40 cluster-only run)")
print("="*72)

pca = PCA(n_components=min(len(axis_names), 10))
pca.fit(M)

print(f"\nVariance per PC:")
cum = 0.0
for i, ev in enumerate(pca.explained_variance_ratio_):
    cum += ev
    print(f"  PC{i+1}: {ev*100:>5.1f}%   (cumulative: {cum*100:>5.1f}%)")


# ============================================================
# THE CENTRAL TEST: target axes projected onto PCs
# ============================================================

print("\n" + "="*72)
print("CENTRAL TEST: target-axis cosines with PCs")
print("="*72)
print()
print("Predictions:")
print("  target_SALIENCE  high on PC1 if PC1 is salience-as-primitive;")
print("                   LOW on PC1 if PC1 is just the V x A diagonal")
print("  target_MOTION    high on PC2 if MOTION is a real primitive")
print("  target_EQ_RUN    high on PC3 if EQ-vs-RUNAWAY is a real primitive")
print()

header = "  " + " " * 18 + "  " + "  ".join(f"PC{i+1:<5}" for i in range(6))
print(header)
print("  " + "-" * (18 + 2 + 8*6))

for tname, tvec in targets.items():
    row = f"  {tname:<18}  "
    for i in range(6):
        pc = pca.components_[i]
        pc_unit = pc / np.linalg.norm(pc)
        c = float(tvec @ pc_unit)
        row += f"{c:>+6.3f}  "
    print(row)


# ============================================================
# Per-target deep dive: top PC match, nearest words
# ============================================================

print("\n" + "="*72)
print("Per-target deep dive")
print("="*72)

for tname, tvec in targets.items():
    print(f"\n--- {tname} ---")

    # PC alignment ranking
    pc_alignments = []
    for i in range(min(8, pca.n_components_)):
        pc_unit = pca.components_[i] / np.linalg.norm(pca.components_[i])
        pc_alignments.append((i+1, float(tvec @ pc_unit), pca.explained_variance_ratio_[i]))
    pc_alignments.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  Best PC alignments (sorted by |cos|):")
    for pcnum, c, var in pc_alignments[:5]:
        print(f"    PC{pcnum} ({var*100:>4.1f}% var): cos = {c:>+.4f}")

    # Cosines with input axes (sanity check: should NOT be high with any single
    # input axis, otherwise the target leaks into PCA input space)
    print(f"  Cosines with INPUT axes (sanity check for vocabulary leakage):")
    input_cos = [(n, float(tvec @ axes[n])) for n in axis_names]
    input_cos.sort(key=lambda x: abs(x[1]), reverse=True)
    for n, c in input_cos[:5]:
        print(f"    {n:>10}: {c:>+.4f}")

    # Nearest-neighbor words for the target axis itself
    print(f"  Positive pole words:")
    for w, s in wv.similar_by_vector(tvec, topn=10):
        print(f"    {s:>+.4f}  {w}")
    print(f"  Negative pole words:")
    for w, s in wv.similar_by_vector(-tvec, topn=10):
        print(f"    {s:>+.4f}  {w}")


# ============================================================
# Save
# ============================================================

np.savez(
    "/Users/macn/Documents/embeddingexp/exp52_results.npz",
    pca_components=pca.components_,
    pca_variance_ratio=pca.explained_variance_ratio_,
    axis_names=np.array(axis_names),
    input_axes=np.stack([axes[n] for n in axis_names]),
    target_salience=target_sal,
    target_motion=target_mot,
    target_eq_run=target_eqr,
)
print("\nSaved: exp52_results.npz")
