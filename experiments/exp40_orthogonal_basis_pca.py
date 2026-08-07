"""
exp40: Find the empirically orthogonal basis WITHIN the salience cluster by
PCA over the stacked candidate axes.

The methodological question Niamh raised: we've been treating valence as a
contaminant to subtract, but valence keeps showing up as the deepest primitive
in the data. Her conceptual basis (existence, valence, direction, motion,
containment) was meant to be mutually orthogonal. Empirically they're not —
they share variance through Lakoff's metaphorical encoding (GOOD IS UP makes
UD and VALENCE entangled).

The cleanest answer: PCA on the stacked candidate axes. The eigenvectors are
mutually orthogonal BY CONSTRUCTION. What do they semantically represent?

Two analyses:
  1. PCA on full set (cluster + independent primitives): the empirical basis
     of all the axes we've been constructing
  2. PCA on cluster only (excluding IO_CLEAN, FORCE, DIFF which are
     structurally independent): what's the orthogonal basis WITHIN the cluster

For each PC, report cumulative variance, which input axes load on it most,
and nearest words on both poles to identify what the PC semantically captures.
"""
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


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


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


axes = {
    "VALENCE":  build_axis(VALENCE_PAIRS),
    "AROUSAL":  build_axis(AROUSAL_PAIRS),
    "UD":       build_axis(UP_DOWN_MML),
    "FB":       build_axis(FORWARD_BACK_MML),
    "LD":       build_axis(LIGHT_DARK_MML),
    "PATH":     build_axis(PATH_MOTION_MML),
    "EXIST":    build_axis(EXISTENCE_MML),
    "BAL":      build_axis(BALANCE_MML),
    "COHERENCE":build_axis(COHERENCE_PAIRS),
    "SUCCESS":  build_axis(SUCCESS_FAILURE_PAIRS),
    "LOSS":     build_axis(LOSS_PAIRS),
    "IO_CLEAN": build_axis(IN_OUT_MML_CLEAN),
    "FORCE":    build_axis(FORCE_MML),
    "DIFF":     build_axis(DIFFICULTY_BURDEN_MML),
}

# Cluster-only subset (exclude the structurally independent primitives from Tier 2)
cluster_only_keys = ["VALENCE", "AROUSAL", "UD", "FB", "LD", "PATH", "EXIST",
                     "BAL", "COHERENCE", "SUCCESS", "LOSS"]


def run_pca_and_report(axis_dict, label, n_top=8):
    print(f"\n{'='*72}")
    print(f"PCA on {label} ({len(axis_dict)} axes)")
    print(f"{'='*72}")
    names = list(axis_dict.keys())
    M = np.stack([axis_dict[n] for n in names])  # (n_axes, 300)

    pca = PCA(n_components=min(len(names), 10))
    pca.fit(M)

    print(f"\nExplained variance ratio per PC:")
    cum = 0.0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cum += ev
        print(f"  PC{i+1}: {ev*100:>5.1f}%   (cumulative: {cum*100:>5.1f}%)")

    print(f"\nFor each PC, report which input axes load on it most + nearest words:")
    for pc_idx in range(min(n_top, len(pca.components_))):
        pc = pca.components_[pc_idx]
        pc_unit = pc / np.linalg.norm(pc)

        # Loadings: which input axes are most-aligned with this PC?
        loadings = [(name, float(axis_dict[name] @ pc_unit)) for name in names]
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  --- PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance) ---")
        print(f"  Input-axis loadings (most-aligned first):")
        for name, load in loadings[:6]:
            print(f"    {name:>10}: {load:>+.4f}")

        # Nearest words to the PC direction
        print(f"  Positive pole (top 12 words):")
        for w, s in wv.similar_by_vector(pc_unit, topn=12):
            print(f"    {s:>+.4f}  {w}")
        print(f"  Negative pole (top 12 words):")
        for w, s in wv.similar_by_vector(-pc_unit, topn=12):
            print(f"    {s:>+.4f}  {w}")


# ============================================================
# Analysis 1: PCA on the FULL set (cluster + independent)
# ============================================================
run_pca_and_report(axes, "FULL set (Tier 1 cluster + Tier 2 independent)", n_top=6)


# ============================================================
# Analysis 2: PCA on cluster only
# ============================================================
cluster_axes = {k: axes[k] for k in cluster_only_keys}
run_pca_and_report(cluster_axes, "CLUSTER only (excluding IO_CLEAN, FORCE, DIFF)", n_top=6)


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp40_results.npz",
    full_axes={n: v for n, v in axes.items()},
)
print("\nSaved: exp40_results.npz")
