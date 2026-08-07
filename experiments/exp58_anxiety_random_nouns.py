"""
exp58: project additional concept words and random nouns onto 6D AI-plus basis.

Niamh asked specifically for anxiety + random nouns (sausage, pyjamas, marigold).
Random nouns serve as a control — concrete physical objects have no theoretical
reason to land at any particular AI-coordinate. They should have LOW
basis-explained (most of their content lives outside agency-space).

Anxiety is the missing clinical state from exp57. Prediction based on
active-inference theory:
  - High A_affect (aroused state)
  - Negative C_reward (bad-state)
  - Negative R_percept_prec (impaired regulation / runaway worry)
  - Possibly positive G_goal_directed (committed-vigilance) OR negative
    (paralyzed-inaction)
  - Compare to fear (G=+0.19, C=-0.18, R=-0.12) and trauma (C=-0.23, D=-0.09)
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, TARGET_SALIENCE_PAIRS,
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS,
)
from exp53_residual_and_goal_directed import TARGET_GOAL_DIRECTED_PAIRS
from exp54_pc1_comparator import TARGET_REWARD_COMPOSITE_PAIRS, TARGET_SURPRISAL_PAIRS
from exp56_explore_exploit import TARGET_EXPLOIT_EXPLORE_PAIRS


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


def build_axis(pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def project_out(v, u):
    u_unit = u / np.linalg.norm(u)
    return v - (v @ u_unit) * u_unit


# Build 6D basis
A_axis = build_axis(TARGET_SALIENCE_PAIRS)
C_axis = build_axis(TARGET_REWARD_COMPOSITE_PAIRS)
D_axis = build_axis(TARGET_SURPRISAL_PAIRS)
G_axis = build_axis(TARGET_GOAL_DIRECTED_PAIRS)
EE_axis = build_axis(TARGET_EXPLOIT_EXPLORE_PAIRS)
VALENCE = build_axis(VALENCE_PAIRS)
AROUSAL = build_axis(AROUSAL_PAIRS)
EQ_raw = build_axis(TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
R_axis = project_out(EQ_raw, VALENCE)
R_axis = project_out(R_axis, AROUSAL)
R_axis = R_axis / np.linalg.norm(R_axis)

# Gram-Schmidt order: most-independent first, G last (most entangled)
gs_order = ["C_rew", "A_aff", "D_cmp", "R_per", "EE_x", "G_pol"]
basis_raw = {"C_rew": C_axis, "A_aff": A_axis, "D_cmp": D_axis,
             "R_per": R_axis, "EE_x": EE_axis, "G_pol": G_axis}

gs_basis = []
for name in gs_order:
    u = basis_raw[name].copy()
    for prev in gs_basis:
        u = u - (u @ prev) * prev
    u = u / np.linalg.norm(u)
    gs_basis.append(u)


def project(v):
    coords = {}
    v_residual = v.copy()
    for name, u_gs in zip(gs_order, gs_basis):
        c = float(v_residual @ u_gs)
        coords[name] = c
        v_residual = v_residual - c * u_gs
    residual_norm = float(np.linalg.norm(v_residual))
    explained = float(np.sqrt(max(0, 1 - residual_norm**2)))
    return coords, explained


# ============================================================
# Concept words to project
# ============================================================
# Clinical / theoretical concepts of interest
clinical = ["anxiety", "panic", "dissociation", "mania", "anhedonia",
            "alexithymia", "burnout", "shame", "guilt"]
# Random concrete nouns (Niamh's request) — should have low basis-explained
random_nouns = ["sausage", "pyjamas", "marigold",
                "stapler", "accordion", "lamppost", "casserole", "pebble"]
# Contemplative / limit states
contemplative = ["awe", "wonder", "presence", "silence", "sublime",
                 "transcendence", "void", "ineffable"]

all_words = [
    ("CLINICAL", clinical),
    ("RANDOM NOUNS (control)", random_nouns),
    ("CONTEMPLATIVE / LIMIT STATES", contemplative),
]


# ============================================================
# Project and report
# ============================================================
print("\n" + "="*82)
print("Concept words and controls projected onto 6D AI-plus basis (Gram-Schmidt)")
print("="*82)

for group_name, words in all_words:
    print(f"\n--- {group_name} ---")
    print(f"{'word':<14} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
    print("-"*82)
    for w in words:
        if w not in wv.key_to_index:
            print(f"  {w:<14}  (not in GloVe vocabulary)")
            continue
        v = wv[w] / np.linalg.norm(wv[w])
        coords, expl = project(v)
        row = f"  {w:<14}"
        for n in gs_order:
            row += f" {coords[n]:>+7.3f}"
        row += f" | {expl*100:>4.1f}%"
        print(row)


# ============================================================
# Headline: anxiety vs fear vs trauma comparison
# ============================================================
print("\n" + "="*82)
print("Comparison: anxiety / fear / trauma / depression / grief")
print("="*82)
print()
print(f"{'word':<14} " + " ".join(f"{n:>7}" for n in gs_order) + f" | {'expl':>5}")
print("-"*82)
for w in ["anxiety", "fear", "panic", "trauma", "depression", "grief", "psychosis"]:
    if w not in wv.key_to_index:
        continue
    v = wv[w] / np.linalg.norm(wv[w])
    coords, expl = project(v)
    row = f"  {w:<14}"
    for n in gs_order:
        row += f" {coords[n]:>+7.3f}"
    row += f" | {expl*100:>4.1f}%"
    print(row)
