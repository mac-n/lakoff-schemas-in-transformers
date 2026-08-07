"""
exp64 — Thread 1 follow-ups: GATING extended tests + MARKOV_BLANKET construction.

After exp63 showed that what we called SELECTION is actually a GATING axis
(value-determined pass/block decision, with positive pole = "thing passed
the gate" and negative pole = "claim was blocked at the gate" in legal /
political register), this experiment has two parts:

PART A — extended tests on GATING (rename of exp63's SELECTION axis):
  A1. cos(GATING, each Lakoff schema) — where does it sit relative to
      the canonical schemas?
  A2. GATING residual after C projected out — what content does GATING
      have that's NOT just reward?
  A3. Gating-distinctive concept words (attention, threshold, filter,
      criterion, judgment, admission, veto, sanction, etc.)

PART B — MARKOV_BLANKET construction (sub-thread C2 from original Entry 25):
  B1. Build axis from abstract self/other anchors, avoiding spatial-
      containment vocab (to stay independent of IO_CLEAN).
  B2. cos(MARKOV_BLANKET, all 7 basis axes + GATING).
  B3. Does MARKOV_BLANKET pick up the abstract self/other concept words
      that IO_CLEAN missed (only `belonging` loaded on IO)?
  B4. Pole vocabulary.
"""

import numpy as np
import gensim.downloader as api

from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML,
    DIFFICULTY_BURDEN_MML,
)

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 basis + exp63 SELECTION axis...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()

exp63 = np.load("exp63_results.npz", allow_pickle=True)
GATING = exp63["SELECTION"]  # renamed in interpretation; same vector


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def build_axis(pairs):
    """exp60 convention: mean of (w_a - w_c) over in-vocab pairs, unit-normalized."""
    offs = []
    used, missing = [], []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
            used.append((a, c))
        else:
            missing.append((a, c))
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


# All 7 basis axes (unit-normalized)
C  = unit(basis_raw["C_rew"])
W  = unit(basis_raw["W_wgt"])
A  = unit(basis_raw["A_aff"])
G  = unit(basis_raw["G_pol"])
R  = unit(basis_raw["R_per"])
D  = unit(basis_raw["D_cmp"])
IO = unit(basis_raw["IO_blk"])
GATING = unit(GATING)

# ===========================================================================
# PART A — GATING extended tests
# ===========================================================================
print("\n" + "#" * 68)
print("# PART A — GATING extended tests")
print("#" * 68)

# ---------------------------------------------------------------------------
# A1. GATING vs Lakoff schemas
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("A1 — cos(GATING, each Lakoff schema)")
print("=" * 68)

schema_specs = [
    ("UD",        UP_DOWN_MML),
    ("IO_CLEAN",  IN_OUT_MML_CLEAN),
    ("FB",        FORWARD_BACK_MML),
    ("PATH",      PATH_MOTION_MML),
    ("LD",        LIGHT_DARK_MML),
    ("EXIST",     EXISTENCE_MML),
    ("FORCE",     FORCE_MML),
    ("BAL",       BALANCE_MML),
    ("DIFF",      DIFFICULTY_BURDEN_MML),
]
schemas = {}
print(f"  {'schema':>8s}  {'cos(GATING, .)':>14s}  {'cos(C, .)':>10s}  {'GATING − C':>10s}")
print(f"  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*10}")
for name, pairs in schema_specs:
    axis, used, missing = build_axis(pairs)
    schemas[name] = axis
    cg = cos(GATING, axis)
    cc = cos(C, axis)
    print(f"  {name:>8s}  {cg:+.4f}         {cc:+.4f}     {cg-cc:+.4f}")

# ---------------------------------------------------------------------------
# A2. GATING residual after C projected out — what's GATING's unique content?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("A2 — GATING residual after C projected out")
print("=" * 68)
gating_minus_C = GATING - (GATING @ C) * C
gmc_unit = unit(gating_minus_C)
gmc_norm = np.linalg.norm(gating_minus_C)
print(f"  |GATING - <GATING,C>·C| = {gmc_norm:.4f}  "
      f"({100 * gmc_norm**2:.1f}% of GATING's variance is NOT C)")
print(f"\n  Residual decomposition onto remaining basis axes:")
for name, vec in [("W_wgt", W), ("A_aff", A), ("G_pol", G),
                  ("R_per", R), ("D_cmp", D), ("IO_blk", IO)]:
    print(f"    {name:>8s}  cos = {cos(gmc_unit, vec):+.4f}")

# Top nearest-neighbor words to the GATING-minus-C residual direction
print(f"\n  Top-15 nearest-neighbor words to GATING-minus-C direction:")
pos = wv.similar_by_vector(gmc_unit.astype(np.float32), topn=15)
neg = wv.similar_by_vector((-gmc_unit).astype(np.float32), topn=15)
print(f"  POSITIVE pole (what GATING has that C doesn't):")
for w, s in pos:
    print(f"    {w:25s}  {s:+.4f}")
print(f"  NEGATIVE pole:")
for w, s in neg:
    print(f"    {w:25s}  {s:+.4f}")

# ---------------------------------------------------------------------------
# A3. Gating-distinctive concept words
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("A3 — Gating-distinctive concept words")
print("=" * 68)
GATING_PROBES = [
    # Attention / salience gating
    "attention", "focus", "salience", "vigilance", "alertness",
    # Decision / criterion / judgment
    "decision", "judgment", "judgement", "criterion", "threshold",
    "filter", "screening", "selection",
    # Permission / sanction
    "admission", "refusal", "approval", "sanction", "veto",
    "permission", "license", "ban", "taboo", "embargo",
    # Gate-related
    "gate", "gatekeeper", "gateway", "barrier", "checkpoint",
    # Recognition / acceptance
    "recognition", "acceptance", "rejection", "endorsement", "denial",
    # Attention as choice
    "noticing", "ignoring", "heeding", "dismissing",
]
print(f"  {'word':>16s}  {'cos(., GATING)':>14s}  {'cos(., C)':>10s}  {'cos(., A)':>10s}  {'cos(., G)':>10s}")
print(f"  {'-'*16}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
for w in GATING_PROBES:
    if w not in wv.key_to_index:
        print(f"  {w:>16s}  OOV")
        continue
    v = unit(wv[w])
    print(f"  {w:>16s}  {cos(v, GATING):+.4f}         "
          f"{cos(v, C):+.4f}     {cos(v, A):+.4f}     {cos(v, G):+.4f}")

# ===========================================================================
# PART B — MARKOV_BLANKET construction
# ===========================================================================
print("\n" + "#" * 68)
print("# PART B — MARKOV_BLANKET construction")
print("#" * 68)

# Abstract self/other anchors, avoiding spatial-containment vocab
# (which would just overlap IO_CLEAN's inside/outside, contained/released,
# enclosed/exposed, sealed/opened anchors)
MARKOV_BLANKET_PAIRS = [
    ("self",         "other"),
    ("agent",        "environment"),
    ("internal",     "external"),
    ("mine",         "theirs"),
    ("own",          "foreign"),
    ("private",      "public"),
    ("subjective",   "objective"),
    ("introspection", "perception"),
    ("autonomous",   "dependent"),
    ("individual",   "collective"),
    ("personal",     "impersonal"),
    ("intrinsic",    "extrinsic"),
    ("endogenous",   "exogenous"),
]

print("\n" + "=" * 68)
print("B1 — Building MARKOV_BLANKET")
print("=" * 68)
MB, used, missing = build_axis(MARKOV_BLANKET_PAIRS)
print(f"  Used {len(used)}/{len(MARKOV_BLANKET_PAIRS)} pairs:")
for a, c in used:
    print(f"    ({a}, {c})")
if missing:
    print(f"  Missing:")
    for a, c in missing:
        print(f"    ({a}, {c})")

# ---------------------------------------------------------------------------
# B2. cos(MARKOV_BLANKET, basis + GATING)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("B2 — cos(MARKOV_BLANKET, all 7 basis axes + GATING)")
print("=" * 68)
print(f"  {'axis':>10s}  {'cos(MB, .)':>10s}  {'bar'}")
print(f"  {'-'*10}  {'-'*10}  {'-'*30}")
for name, vec in [("C_rew", C), ("W_wgt", W), ("A_aff", A), ("G_pol", G),
                  ("R_per", R), ("D_cmp", D), ("IO_blk", IO),
                  ("GATING", GATING)]:
    c = cos(MB, vec)
    bar = "#" * int(abs(c) * 50)
    sign = "+" if c >= 0 else "-"
    print(f"  {name:>10s}  {c:+.4f}    {bar}")

# Key sub-thread C2 question: does MARKOV_BLANKET reframe IO_CLEAN, or is it
# independent?
print(f"\n  Sub-thread C2 verdict — cos(MARKOV_BLANKET, IO_CLEAN) = {cos(MB, IO):+.4f}")
c_mb_io = cos(MB, IO)
if abs(c_mb_io) > 0.5:
    print(f"  → SAME AXIS — MB is an IO reframe")
elif abs(c_mb_io) > 0.2:
    print(f"  → RELATED — share substantial content but distinct primitives")
else:
    print(f"  → DISTINCT — MB and IO_CLEAN are independent primitives "
          "(both candidate substrate primitives, different content)")

# ---------------------------------------------------------------------------
# B3. Concept-word coverage — does MB pick up what IO missed?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("B3 — Concept-word coverage on MARKOV_BLANKET vs IO_CLEAN")
print("=" * 68)
IO_PROBE_WORDS = [
    "self", "selfhood", "identity", "boundary", "intimacy",
    "belonging", "alienation", "isolation", "communion", "exile",
    "membership", "kinship", "loneliness", "togetherness",
    # Plus some extra abstract self/other concepts
    "ego", "soul", "individuality", "autonomy", "sovereignty",
    "interiority", "subjectivity",
]
print(f"  {'word':>16s}  {'cos(., MB)':>10s}  {'cos(., IO)':>10s}  {'cos(., GATING)':>14s}")
print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*14}")
mb_wins = 0
io_wins = 0
for w in IO_PROBE_WORDS:
    if w not in wv.key_to_index:
        print(f"  {w:>16s}  OOV")
        continue
    v = unit(wv[w])
    c_mb = cos(v, MB)
    c_io = cos(v, IO)
    c_g  = cos(v, GATING)
    if abs(c_mb) > abs(c_io):
        mb_wins += 1
        marker = "  ← MB"
    else:
        io_wins += 1
        marker = "  ← IO"
    print(f"  {w:>16s}  {c_mb:+.4f}    {c_io:+.4f}    {c_g:+.4f}{marker}")
print(f"\n  MB has larger |cos| on {mb_wins} words; IO_CLEAN on {io_wins}")

# ---------------------------------------------------------------------------
# B4. Pole vocabulary
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("B4 — MARKOV_BLANKET pole vocabulary (top-20 each pole)")
print("=" * 68)
print("\n  POSITIVE pole (self / agent / internal / mine / private / autonomous side):")
for w, s in wv.similar_by_vector(MB.astype(np.float32), topn=20):
    print(f"    {w:25s}  {s:+.4f}")
print("\n  NEGATIVE pole (other / environment / external / theirs / public side):")
for w, s in wv.similar_by_vector((-MB).astype(np.float32), topn=20):
    print(f"    {w:25s}  {s:+.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp64_results.npz",
         GATING=GATING,
         MARKOV_BLANKET=MB,
         gating_minus_C=gating_minus_C,
         gmc_norm=gmc_norm)
print("\nSaved exp64_results.npz")
