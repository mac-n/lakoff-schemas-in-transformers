"""
exp36: Robustness check on the Phase C ordering by forcing different starting axes.

Greedy explanation-power ordering picks the most CONNECTED axis first, not
necessarily the most fundamental. UD came first in exp35 because it shares
substantial variance with many others. We can't tell whether UD's centrality
reflects its embodied primacy or just its vocabulary pervasiveness in English.

The principled robustness test: if the cluster structure is real (one entangled
cluster + a tail of independent primitives), then forcing DIFFERENT starting
axes should give the SAME tail of independent primitives, even if the cluster
internal ordering shuffles. If forcing different starts gives wildly different
tails, the greedy procedure is unreliable.

For each forced starting axis:
  1. Pick that axis first (residualize all others against it)
  2. Continue greedy from there
  3. Report the final ordering AND the explanation power trajectory
  4. Compare tails (last 4 axes — the candidates for "independent primitives")

Test starts:
  - UD (the original)
  - EXIST (the candidate yang/yin axis)
  - VALENCE (the affect primary)
  - IO_CLEAN (an axis empirically independent of the cluster)
  - FORCE (another empirically independent axis)

Stability metric: how often does each axis appear in the bottom-4 (last 4 picked)
across the 5 forced-start runs?
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML,
    PATH_MOTION_MML, LIGHT_DARK_MML, EXISTENCE_MML,
    FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
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


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


axes_init = {
    "VALENCE":  build_axis(VALENCE_PAIRS),
    "AROUSAL":  build_axis(AROUSAL_PAIRS),
    "UD":       build_axis(UP_DOWN_MML),
    "IO":       build_axis(IN_OUT_MML),
    "IO_CLEAN": build_axis(IN_OUT_MML_CLEAN),
    "FB":       build_axis(FORWARD_BACK_MML),
    "PATH":     build_axis(PATH_MOTION_MML),
    "LD":       build_axis(LIGHT_DARK_MML),
    "EXIST":    build_axis(EXISTENCE_MML),
    "FORCE":    build_axis(FORCE_MML),
    "BAL":      build_axis(BALANCE_MML),
    "DIFF":     build_axis(DIFFICULTY_BURDEN_MML),
}


def greedy_with_forced_start(start_axis, axes):
    """Run greedy explanation-power projection, but force the first pick."""
    remaining = {k: v.copy() for k, v in axes.items()}
    order = []

    # Force the first pick
    w_v = remaining.pop(start_axis)
    w_v_unit = w_v / np.linalg.norm(w_v)
    # Compute explanation power that the forced pick has
    power_forced = sum(float(other @ w_v_unit) ** 2 for other in remaining.values())
    order.append((start_axis, power_forced))
    # Subtract from all remaining
    for name in list(remaining.keys()):
        v = remaining[name]
        remaining[name] = v - float(v @ w_v_unit) * w_v_unit

    # Continue greedy
    while remaining:
        powers = {
            name: sum(
                float(other @ (v / (np.linalg.norm(v) + 1e-12))) ** 2
                for other_name, other in remaining.items() if other_name != name
            )
            for name, v in remaining.items()
        }
        winner = max(powers, key=powers.get)
        order.append((winner, powers[winner]))
        w_v = remaining.pop(winner)
        if np.linalg.norm(w_v) < 1e-12:
            # nothing to subtract; skip
            continue
        w_v_unit = w_v / np.linalg.norm(w_v)
        for name in list(remaining.keys()):
            v = remaining[name]
            remaining[name] = v - float(v @ w_v_unit) * w_v_unit
    return order


print("\n=== Running 5 forced-start orderings ===")
forced_starts = ["UD", "EXIST", "VALENCE", "IO_CLEAN", "FORCE"]
all_orderings = {}

for start in forced_starts:
    print(f"\n--- Forced start: {start} ---")
    order = greedy_with_forced_start(start, axes_init)
    all_orderings[start] = order
    print(f"  step  axis        power")
    for i, (name, power) in enumerate(order, 1):
        print(f"  {i:>2}.   {name:>10}  {power:>6.3f}")


# =================================================================
# Compare tails: which axes appear in bottom-4 across all 5 runs?
# =================================================================
print("\n\n=== Tail stability check ===")
print("If the cluster structure is real, the tail (last 4 axes picked) should")
print("be stable across forced-start choices. Independent primitives should")
print("appear in the tail regardless of which axis went first.\n")

# Bottom-4 for each forced start
tail_axes = {}
for start, order in all_orderings.items():
    bottom4 = set(name for name, _ in order[-4:])
    tail_axes[start] = bottom4
    print(f"  Forced start {start:>10} → tail: {sorted(bottom4)}")

# Frequency of each axis appearing in any tail
from collections import Counter
tail_counts = Counter()
for tail in tail_axes.values():
    for ax in tail:
        tail_counts[ax] += 1

print("\nFrequency of appearance in bottom-4 across 5 runs:")
for ax, count in tail_counts.most_common():
    print(f"  {ax:>10}: {count}/5 runs")


# =================================================================
# Compare full orderings: how much do they shuffle?
# =================================================================
print("\n=== Ordering shuffle check ===")
print("If two orderings put axes in similar positions, the structure is robust.")
print("Compute Spearman rank correlation between pairs of orderings.\n")

def to_ranks(order):
    return {name: i for i, (name, _) in enumerate(order)}

ranks = {start: to_ranks(order) for start, order in all_orderings.items()}
names = list(axes_init.keys())

print(f"  {'':>10}" + "  ".join(f"{s:>8}" for s in forced_starts))
for s1 in forced_starts:
    row = []
    for s2 in forced_starts:
        r1 = np.array([ranks[s1][n] for n in names])
        r2 = np.array([ranks[s2][n] for n in names])
        # Spearman rho = pearson on ranks
        rho = np.corrcoef(r1, r2)[0, 1]
        row.append(f"{rho:>+8.3f}")
    print(f"  {s1:>10}  " + "  ".join(row))

print("\n=== Reading guide ===")
print("  High Spearman (>0.7): orderings agree → cluster structure robust to start choice")
print("  Low Spearman (<0.3):  orderings disagree → greedy procedure unreliable")
print("  Tail stability: axes appearing 4-5/5 times in bottom-4 are robustly independent primitives")
print("  Axes appearing 0-1/5 times in bottom-4 are robustly central (high-connectedness)")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp36_results.npz",
    orderings={s: o for s, o in all_orderings.items()},
)
print("\nSaved: exp36_results.npz")
