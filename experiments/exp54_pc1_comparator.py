"""
exp54: PC1 comparator test — which candidate target axis best captures PC1?

Question (Niamh): what IS PC1 most precisely? PC1's input-axis loadings
(VALENCE +0.69, AROUSAL -0.65, SUCCESS +0.50, LOSS +0.50, UD +0.43) look like
a COMPOSITE — value bundled with peacefulness bundled with outcome-attainment
bundled with resource-status. Niamh's hypothesis: PC1 = the brain's reward
function / expected free energy. Active inference's framing: expected free
energy is the integrated quantity an agent minimizes via action selection,
bundling pragmatic-value with predicted-surprise with outcome with body-state.

Four candidates to compare:

  A: target_RUSSELL_DIAGONAL  — pure V x A diagonal (pleasant-calm vs
                                unpleasant-aroused). Already tested in
                                exp52 at cos=-0.17 with PC1, so we know
                                it's not the whole story.
  B: target_VALUE_PURE        — preference-over-outcomes vocabulary
                                (preferred, wanted, sought, cherished).
                                Pure pragmatic value without arousal/outcome.
  C: target_REWARD_COMPOSITE  — integrated reward / expected-free-energy
                                hypothesis. Anchors that bundle wellbeing,
                                outcome, fortune, body-state in single
                                words: flourishing/suffering, thriving/
                                struggling, blessed/cursed, etc.
  D: target_SURPRISAL         — predictability-vs-surprise vocabulary,
                                avoiding COHERENCE overlap. Tests whether
                                PC1 is specifically about surprise/novelty
                                (active-inference's surprisal term).

The candidate that best captures PC1 (highest |cos|) tells us what PC1
most precisely IS. That target axis then becomes the value-basis vector
for the subsequent inversion experiment (exp55: Lakoff schemas projected
onto AI-space).

Also reported: inter-candidate cosine matrix (how distinct are the
candidates from each other?), per-candidate input-axis loadings (sanity
check for circularity), nearest-neighbor words on each pole.
"""
import numpy as np
import gensim.downloader as api
from sklearn.decomposition import PCA
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
    TARGET_SALIENCE_PAIRS as TARGET_RUSSELL_DIAGONAL_PAIRS,
)


# ============================================================
# Four PC1-candidate target axes
# ============================================================

# Candidate A: Russell V x A diagonal (re-use exp52's original anchors)
# Already imported as TARGET_RUSSELL_DIAGONAL_PAIRS.

# Candidate B: pure preference-over-outcomes vocabulary.
# No overlap with V (which uses pleasant/desirable/agreeable/etc), no overlap
# with UD's despised/esteemed, no overlap with IO's dismissed.
TARGET_VALUE_PURE_PAIRS = [
    ("preferred",  "dispreferred"),
    ("wanted",     "unwanted"),
    ("sought",     "shunned"),
    ("cherished",  "loathed"),
    ("loved",      "hated"),
    ("treasured",  "abhorred"),
    ("valued",     "devalued"),
    ("welcomed",   "rebuffed"),
    ("embraced",   "rejected"),
    ("approached", "avoided"),
    ("adored",     "detested"),
    ("admired",    "scorned"),
]

# Candidate C: integrated-reward / expected-free-energy hypothesis.
# Each anchor word bundles multiple dimensions (wellbeing, outcome, fortune,
# resource-state) — testing whether PC1 is a composite-reward signal rather
# than a pure-value signal.
TARGET_REWARD_COMPOSITE_PAIRS = [
    ("flourishing", "suffering"),
    ("thriving",    "struggling"),
    ("prospering",  "declining"),
    ("blessed",     "cursed"),
    ("fortunate",   "unfortunate"),
    ("fulfilled",   "ruined"),
    ("privileged",  "oppressed"),
    ("graced",      "plagued"),
    ("charmed",     "hexed"),
    ("favored",     "disfavored"),
    ("lucky",       "unlucky"),
    ("wholesome",   "broken"),
]

# Candidate D: surprisal / predictability-vs-shock.
# Avoids COHERENCE vocabulary (predictable/surprising/expected/unexpected/
# ordinary/anomalous/regular/irregular/normal/aberrant/orderly/chaotic) and
# LD's known/unknown/revealed/concealed.
TARGET_SURPRISAL_PAIRS = [
    ("familiar",    "unfamiliar"),
    ("routine",     "novel"),
    ("recognized",  "unrecognized"),
    ("anticipated", "unanticipated"),
    ("foreseen",    "unforeseen"),
    ("commonplace", "extraordinary"),
    ("mundane",     "astonishing"),
    ("typical",     "atypical"),
    ("customary",   "unprecedented"),
    ("rote",        "startling"),
    ("habitual",    "jarring"),
    ("everyday",    "shocking"),
]


# ============================================================
# Load embeddings and build axes
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
        raise ValueError(f"No pairs survived OOV for {label}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw), len(offs)


# Build the 11 cluster input axes (matches exp40 / exp52)
print("\nBuilding PCA input axes (11-axis cluster set)...")
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
M = np.stack([axes[n] for n in axis_names])

# Build the four candidate target axes
print("\nBuilding PC1-candidate target axes...")
target_A, n_A = build_axis(TARGET_RUSSELL_DIAGONAL_PAIRS, "A_RUSSELL_DIAGONAL")
target_B, n_B = build_axis(TARGET_VALUE_PURE_PAIRS,       "B_VALUE_PURE")
target_C, n_C = build_axis(TARGET_REWARD_COMPOSITE_PAIRS, "C_REWARD_COMPOSITE")
target_D, n_D = build_axis(TARGET_SURPRISAL_PAIRS,        "D_SURPRISAL")
print(f"  A: Russell diagonal       {n_A}/{len(TARGET_RUSSELL_DIAGONAL_PAIRS)} pairs")
print(f"  B: Value pure             {n_B}/{len(TARGET_VALUE_PURE_PAIRS)} pairs")
print(f"  C: Reward composite       {n_C}/{len(TARGET_REWARD_COMPOSITE_PAIRS)} pairs")
print(f"  D: Surprisal              {n_D}/{len(TARGET_SURPRISAL_PAIRS)} pairs")

candidates = {
    "A_RUSSELL_DIAGONAL":  target_A,
    "B_VALUE_PURE":        target_B,
    "C_REWARD_COMPOSITE":  target_C,
    "D_SURPRISAL":         target_D,
}


# ============================================================
# Run the cluster PCA
# ============================================================
pca = PCA(n_components=min(len(axis_names), 10))
pca.fit(M)
print(f"\nPCA on cluster axes complete.")
print(f"Variance: " + ", ".join(
    f"PC{i+1}={pca.explained_variance_ratio_[i]*100:.1f}%"
    for i in range(6)
))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# HEADLINE COMPARATOR: which candidate best captures PC1?
# ============================================================
print("\n" + "="*72)
print("HEADLINE COMPARATOR: candidates vs PC1 (and other PCs)")
print("="*72)
print()
print("                          PC1      PC2      PC3      PC4      PC5      PC6")
print("  " + "-"*72)
for name, vec in candidates.items():
    row = f"  {name:<24}"
    for i in range(6):
        row += f" {cos(vec, pca.components_[i]):>+7.3f} "
    print(row)

# Print sorted by |cos(PC1)|
print(f"\nRanked by |cos(target, PC1)|:")
ranked = sorted(candidates.items(),
                key=lambda kv: abs(cos(kv[1], pca.components_[0])),
                reverse=True)
for name, vec in ranked:
    c1 = cos(vec, pca.components_[0])
    print(f"  {name:<24}  |cos(PC1)| = {abs(c1):.3f}  (signed: {c1:+.3f})")

winner_name, winner_vec = ranked[0]
print(f"\n→ WINNER: {winner_name}")
print(f"  |cos(PC1)| = {abs(cos(winner_vec, pca.components_[0])):.3f}")


# ============================================================
# Inter-candidate distinctness: how different are the candidates?
# ============================================================
print("\n" + "="*72)
print("Inter-candidate cosine matrix")
print("="*72)
print(f"\nIf candidates are essentially the same axis in different vocab, |cos|")
print(f"between them will be high (>0.7). If genuinely distinct, lower.\n")

cnames = list(candidates.keys())
print("  " + " "*22 + " ".join(f"{n:<22}" for n in cnames))
for n1 in cnames:
    row = f"  {n1:<22}"
    for n2 in cnames:
        if n1 == n2:
            row += f" {'  1.000':<22}"
        else:
            row += f" {cos(candidates[n1], candidates[n2]):>+7.3f}              "
    print(row)


# ============================================================
# Per-candidate deep dive
# ============================================================
print("\n" + "="*72)
print("Per-candidate deep dive: input-axis loadings, nearest-neighbor words")
print("="*72)

for name, vec in candidates.items():
    print(f"\n--- {name} ---")

    # PC alignment ranking
    pc_aligns = [(i+1, cos(vec, pca.components_[i]), pca.explained_variance_ratio_[i])
                 for i in range(min(8, pca.n_components_))]
    pc_aligns.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top PC alignments:")
    for pcnum, c, v in pc_aligns[:4]:
        print(f"    PC{pcnum} ({v*100:>4.1f}% var): cos = {c:>+.4f}")

    # Cosines with input axes
    input_cos = [(n, cos(vec, axes[n])) for n in axis_names]
    input_cos.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  Top input-axis loadings (sanity check for circularity):")
    for n, c in input_cos[:5]:
        print(f"    {n:>10}: {c:>+.4f}")

    # Nearest words
    print(f"  Positive pole:")
    for w, s in wv.similar_by_vector(vec, topn=8):
        print(f"    {s:>+.4f}  {w}")
    print(f"  Negative pole:")
    for w, s in wv.similar_by_vector(-vec, topn=8):
        print(f"    {s:>+.4f}  {w}")


# ============================================================
# Verdict + recommendation for exp55 (inversion)
# ============================================================
print("\n" + "="*72)
print("Verdict and recommendation for exp55 inversion experiment")
print("="*72)

winner_c1 = cos(winner_vec, pca.components_[0])
print(f"\nBest PC1-capturing candidate: {winner_name} at |cos(PC1)| = {abs(winner_c1):.3f}")
print(f"\nInterpretation guide:")
print(f"  |cos(PC1)| > 0.7    → strong capture; PC1 is well-described by this candidate")
print(f"  0.4 < |cos(PC1)| <= 0.7 → moderate capture; PC1 contains this content + more")
print(f"  |cos(PC1)| <= 0.4   → weak capture; PC1 is something else than this candidate")

# Specific recommendation logic
if abs(winner_c1) > 0.7:
    print(f"\n→ Use {winner_name} as the value-axis basis in exp55. PC1 is essentially this.")
elif abs(winner_c1) > 0.4:
    print(f"\n→ Use {winner_name} as the value-axis basis in exp55 with caveat: PC1")
    print(f"  has additional content this target doesn't capture. Report residual.")
else:
    print(f"\n→ No candidate strongly captures PC1. PC1 is a composite none of these")
    print(f"  individually represents. Consider building a 4-axis basis (using all")
    print(f"  three of A/B/C plus a separate composite) or reconsider the inversion design.")


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp54_results.npz",
    target_russell_diagonal=target_A,
    target_value_pure=target_B,
    target_reward_composite=target_C,
    target_surprisal=target_D,
    pca_components=pca.components_,
    pca_variance_ratio=pca.explained_variance_ratio_,
    axis_names=np.array(axis_names),
    input_axes=np.stack([axes[n] for n in axis_names]),
    winner=winner_name,
)
print("\nSaved: exp54_results.npz")
