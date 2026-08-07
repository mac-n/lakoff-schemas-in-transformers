"""
exp66 — G subdivision + target_PROGRESS construction.

PART A — G subdivision.
Sort G's 14 anchor pairs into three theoretically-distinct sub-meanings:
  COMMITMENT — policy precision proper (π in active inference)
  ORIENTATION — having priors over preferred outcomes
  PROGRESS  — state-shadow of free-energy reduction (a derivative quantity)

Build a sub-axis from each subset. Test:
  - Are they separable? (inter-sub-axis cosines)
  - Which sub-meaning carries the A-G entanglement?
  - Does PROGRESS specifically light up FB (Lakoff PROGRESS-IS-FORWARD)?
  - Which sub-meaning best captures PC2's content?

PART B — target_PROGRESS from clean non-G vocabulary.
Build a fresh PROGRESS axis from anchors that DON'T overlap with G's existing
anchors. Test inter-axis cosines, FB/PATH coupling, pole vocabulary, and
whether it's a candidate 10th basis axis (after the DV and MB additions).
"""

import numpy as np
import gensim.downloader as api

from lakoff_canonical_vocabulary import (
    FORWARD_BACK_MML, PATH_MOTION_MML,
)

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 basis + exp63 SELECTION + exp64 MB...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp63 = np.load("exp63_results.npz", allow_pickle=True)
exp64 = np.load("exp64_results.npz", allow_pickle=True)


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(pairs):
    offs, used, missing = [], [], []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c))
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


C  = unit(basis_raw["C_rew"])
W  = unit(basis_raw["W_wgt"])
A  = unit(basis_raw["A_aff"])
G  = unit(basis_raw["G_pol"])
R  = unit(basis_raw["R_per"])
D  = unit(basis_raw["D_cmp"])
IO = unit(basis_raw["IO_blk"])
DV = unit(exp63["SELECTION"])
MB = unit(exp64["MARKOV_BLANKET"])

# ===========================================================================
# PART A — G subdivision
# ===========================================================================
print("\n" + "#" * 68)
print("# PART A — G subdivided into COMMITMENT, ORIENTATION, PROGRESS")
print("#" * 68)

G_COMMIT_PAIRS = [
    ("deliberate",  "accidental"),
    ("motivated",   "unmotivated"),
    ("intentional", "unintentional"),
    ("resolute",    "hesitant"),
    ("committed",   "uncommitted"),
    ("decided",     "undecided"),
    ("ambitious",   "complacent"),
]

G_ORIENT_PAIRS = [
    ("aiming",     "wandering"),
    ("purposeful", "aimless"),
    ("oriented",   "disoriented"),
    ("targeted",   "untargeted"),
]

G_PROGRESS_SUB_PAIRS = [   # PROGRESS pairs that were already in G's anchors
    ("pursuing", "idling"),
    ("driven",   "becalmed"),
    ("chasing",  "dawdling"),
]

print("\n  Building G sub-axes...")
G_commit,  used_c, miss_c = build_axis(G_COMMIT_PAIRS)
G_orient,  used_o, miss_o = build_axis(G_ORIENT_PAIRS)
G_prog_sub, used_p, miss_p = build_axis(G_PROGRESS_SUB_PAIRS)
print(f"  G_commit:  {len(used_c)}/{len(G_COMMIT_PAIRS)} pairs in vocab")
print(f"  G_orient:  {len(used_o)}/{len(G_ORIENT_PAIRS)} pairs in vocab")
print(f"  G_prog_sub: {len(used_p)}/{len(G_PROGRESS_SUB_PAIRS)} pairs in vocab")
for label, miss in [("G_commit", miss_c), ("G_orient", miss_o), ("G_prog_sub", miss_p)]:
    if miss:
        print(f"    {label} missing: {miss}")

# A1 — separability check
print("\n" + "=" * 68)
print("A1 — Inter-sub-axis cosines (are they actually separable?)")
print("=" * 68)
print(f"  cos(G_commit, G_orient)    = {cos(G_commit, G_orient):+.4f}")
print(f"  cos(G_commit, G_prog_sub)  = {cos(G_commit, G_prog_sub):+.4f}")
print(f"  cos(G_orient, G_prog_sub)  = {cos(G_orient, G_prog_sub):+.4f}")
print(f"\n  Sanity — each sub-axis vs full G:")
print(f"  cos(G_commit, G)    = {cos(G_commit, G):+.4f}")
print(f"  cos(G_orient, G)    = {cos(G_orient, G):+.4f}")
print(f"  cos(G_prog_sub, G)  = {cos(G_prog_sub, G):+.4f}")

# A2 — which sub-axis carries the A-G entanglement?
print("\n" + "=" * 68)
print("A2 — Which sub-meaning carries the A entanglement?")
print("=" * 68)
print(f"  Baseline:   cos(A, G)           = {cos(A, G):+.4f}")
print(f"")
print(f"  cos(A, G_commit)     = {cos(A, G_commit):+.4f}")
print(f"  cos(A, G_orient)     = {cos(A, G_orient):+.4f}")
print(f"  cos(A, G_prog_sub)   = {cos(A, G_prog_sub):+.4f}")

# A3 — sub-axes vs the 7 basis axes
print("\n" + "=" * 68)
print("A3 — Sub-axes vs each basis axis")
print("=" * 68)
header = f"  {'axis':>14s}  {'G_commit':>10s}  {'G_orient':>10s}  {'G_prog_sub':>10s}"
print(header)
print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, vec in [("C_rew", C), ("W_wgt", W), ("A_aff", A), ("G_pol", G),
                  ("R_per", R), ("D_cmp", D), ("IO_blk", IO),
                  ("DV", DV), ("MB", MB)]:
    print(f"  {name:>14s}  {cos(G_commit, vec):+.4f}    "
          f"{cos(G_orient, vec):+.4f}    {cos(G_prog_sub, vec):+.4f}")

# A4 — Lakoff PROGRESS-IS-FORWARD test
print("\n" + "=" * 68)
print("A4 — Lakoff PROGRESS-IS-FORWARD test (FB and PATH)")
print("=" * 68)
FB_axis, _, _ = build_axis(FORWARD_BACK_MML)
PATH_axis, _, _ = build_axis(PATH_MOTION_MML)
print(f"  cos(G_commit,   FB)   = {cos(G_commit, FB_axis):+.4f}")
print(f"  cos(G_orient,   FB)   = {cos(G_orient, FB_axis):+.4f}")
print(f"  cos(G_prog_sub, FB)   = {cos(G_prog_sub, FB_axis):+.4f}  "
      f"← prediction: highest if PROGRESS-IS-FORWARD")
print(f"  cos(G,          FB)   = {cos(G, FB_axis):+.4f}  "
      f"(full G baseline)")
print(f"")
print(f"  cos(G_commit,   PATH) = {cos(G_commit, PATH_axis):+.4f}")
print(f"  cos(G_orient,   PATH) = {cos(G_orient, PATH_axis):+.4f}")
print(f"  cos(G_prog_sub, PATH) = {cos(G_prog_sub, PATH_axis):+.4f}")
print(f"  cos(G,          PATH) = {cos(G, PATH_axis):+.4f}")

# ===========================================================================
# PART B — target_PROGRESS from clean non-G vocabulary
# ===========================================================================
print("\n" + "#" * 68)
print("# PART B — target_PROGRESS from clean non-G vocabulary")
print("#" * 68)

PROGRESS_PAIRS = [
    ("advancing",    "regressing"),
    ("progressing",  "stalling"),
    ("gaining",      "losing"),
    ("nearing",      "distancing"),
    ("closing",      "receding"),
    ("improving",    "deteriorating"),
    ("mounting",     "dwindling"),
    ("accelerating", "decelerating"),
    ("proceeding",   "halting"),
    ("developing",   "declining"),
]

print("\n  Building target_PROGRESS...")
PROGRESS, used_pr, miss_pr = build_axis(PROGRESS_PAIRS)
print(f"  Used {len(used_pr)}/{len(PROGRESS_PAIRS)} pairs:")
for a, c in used_pr:
    print(f"    ({a}, {c})")
if miss_pr:
    print(f"  Missing:")
    for a, c in miss_pr:
        print(f"    ({a}, {c})")

# B1 — pole vocabulary
print("\n" + "=" * 68)
print("B1 — PROGRESS pole vocabulary (top-15)")
print("=" * 68)
print("\n  POSITIVE pole (advancing / progressing / gaining side):")
for w, s in wv.similar_by_vector(PROGRESS.astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")
print("\n  NEGATIVE pole (regressing / stalling / losing side):")
for w, s in wv.similar_by_vector((-PROGRESS).astype(np.float32), topn=15):
    print(f"    {w:25s}  {s:+.4f}")

# B2 — cos vs all axes
print("\n" + "=" * 68)
print("B2 — cos(PROGRESS, each axis)")
print("=" * 68)
print(f"  {'axis':>16s}  {'cos(PROGRESS,.)':>15s}  bar")
print(f"  {'-'*16}  {'-'*15}  {'-'*40}")
all_axes = [
    ("C_rew", C), ("W_wgt", W), ("A_aff", A), ("G_pol", G),
    ("R_per", R), ("D_cmp", D), ("IO_blk", IO),
    ("DV (decision-verdict)", DV), ("MB", MB),
    ("FB_lakoff", FB_axis), ("PATH_lakoff", PATH_axis),
    ("G_commit", G_commit), ("G_orient", G_orient), ("G_prog_sub", G_prog_sub),
]
for name, vec in all_axes:
    c = cos(PROGRESS, vec)
    bar = "#" * int(abs(c) * 50)
    print(f"  {name:>16s}  {c:+.4f}         {bar}")

# B3 — A-G entanglement after projecting out PROGRESS?
print("\n" + "=" * 68)
print("B3 — Does projecting out PROGRESS disentangle A and G?")
print("=" * 68)
A_res = A - (A @ PROGRESS) * PROGRESS
G_res = G - (G @ PROGRESS) * PROGRESS
print(f"  cos(A, G) before              = {cos(A, G):+.4f}")
print(f"  cos(A_res, G_res) after PROG  = {cos(A_res, G_res):+.4f}")
print(f"  Δ                             = {cos(A_res, G_res) - cos(A, G):+.4f}")

# B4 — Concept-word projections
print("\n" + "=" * 68)
print("B4 — Concept-word projections onto PROGRESS")
print("=" * 68)
PROBES = [
    # Progress-relevant
    "progress", "advancement", "achievement", "accomplishment",
    "growth", "development", "improvement",
    "stagnation", "plateau", "breakthrough", "setback",
    "momentum", "flow",
    # Process / agency
    "agency", "freedom", "hope", "ambition", "drive",
    "motivation", "purpose", "goal",
    # Affective (to test A coupling)
    "joy", "satisfaction", "fulfillment", "frustration", "disappointment",
    # Clinical (test fear/anxiety/depression on PROGRESS)
    "anxiety", "fear", "depression", "trauma", "shame",
]
print(f"  {'word':>16s}  {'cos(., PROG)':>12s}  {'cos(., A)':>10s}  "
      f"{'cos(., G)':>10s}  {'cos(., FB)':>10s}")
print(f"  {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
for w in PROBES:
    if w not in wv.key_to_index:
        print(f"  {w:>16s}  OOV")
        continue
    v = unit(wv[w])
    print(f"  {w:>16s}  {cos(v, PROGRESS):+.4f}       "
          f"{cos(v, A):+.4f}     {cos(v, G):+.4f}     "
          f"{cos(v, FB_axis):+.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp66_results.npz",
         G_commit=G_commit, G_orient=G_orient, G_prog_sub=G_prog_sub,
         PROGRESS=PROGRESS,
         FB_axis=FB_axis, PATH_axis=PATH_axis)
print("\nSaved exp66_results.npz")
