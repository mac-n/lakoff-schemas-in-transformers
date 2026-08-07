"""
exp69 — Clean reconstruction test + intentional-structure test.

Niamh's pushback (correctly): "if they were actual primitives we'd see them
separating orthogonally (to the extent that them not being well encoded in
language allows)." A and G's persistent +0.56 cosine across constructions
may reflect anchor-vocabulary cross-bleed (A includes focused/directed/
attentive — intentional; G includes oriented/targeted — attentional)
rather than substrate-real coupling.

Two tests:

PART A — clean reconstruction.
Build ATTENTION_CLEAN (perceptual-act vocab without commitment terms) and
INTENTION_CLEAN (commitment-to-action vocab without perceptual terms).
Measure cos(ATT_CLEAN, INT_CLEAN). If << 0.56, the original entanglement
was substantially anchor-bias and attention/intention ARE separable primitives.

PART B — intentional structure (subject/object).
Build target_INTENT_STRUCT from grammatical subject/object pairs of acts
of awareness/perception/action. Check whether it aligns with A (object-side),
G (subject-side), GP, COMPLEMENT, or is independent.
"""

import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 + 64 + 67 axes...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
exp64 = np.load("exp64_results.npz", allow_pickle=True)
exp67 = np.load("exp67_results.npz", allow_pickle=True)


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
            missing.append((a, c, a not in wv.key_to_index, c not in wv.key_to_index))
    if not offs:
        return None, used, missing
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


A  = unit(basis_raw["A_aff"])
G  = unit(basis_raw["G_pol"])
C  = unit(basis_raw["C_rew"])
W  = unit(basis_raw["W_wgt"])
R  = unit(basis_raw["R_per"])
D  = unit(basis_raw["D_cmp"])
IO = unit(basis_raw["IO_blk"])
MB = unit(exp64["MARKOV_BLANKET"])
GP   = unit(exp67["GATING_PROCESS"])
COMP = unit(exp67["COMPLEMENT"])

print(f"\nBaseline cos(A_orig, G_orig) = {cos(A, G):+.4f}  ← the +0.56 we're scrutinizing")

# ===========================================================================
# PART A — clean reconstruction
# ===========================================================================
print("\n" + "#" * 68)
print("# PART A — clean ATTENTION and clean INTENTION (no cross-bleed)")
print("#" * 68)

# Attention without intention-flavoring. Perceptual-act vocabulary
# specifically about NOTICING / PERCEIVING / DETECTING, avoiding terms that
# imply commitment-to-action (focused/directed/attentive all leak intention).
ATT_CLEAN_PAIRS = [
    ("noticing",    "missing"),
    ("perceiving",  "overlooking"),
    ("sensing",     "missing"),
    ("detecting",   "missing"),
    ("spotting",    "missing"),
    ("recognizing", "overlooking"),
    ("seeing",      "missing"),
    ("hearing",     "missing"),
    ("registering", "ignoring"),
    ("witnessing",  "missing"),
    ("observing",   "missing"),
    ("aware",       "unaware"),
]

# Intention without attention-flavoring. Commitment / planning / deciding
# vocabulary, avoiding terms that imply perceptual state (oriented/targeted
# leak attentional content).
INT_CLEAN_PAIRS = [
    ("intending",  "drifting"),
    ("planning",   "improvising"),
    ("deciding",   "deferring"),
    ("committing", "hedging"),
    ("choosing",   "defaulting"),
    ("designing",  "improvising"),
    ("resolving",  "postponing"),
    ("scheduling", "winging"),
    ("plotting",   "freelancing"),
    ("aiming",     "drifting"),
    ("intending",  "stumbling"),
    ("plan",       "improvise"),
]

print("\n  Building ATTENTION_CLEAN...")
ATT_CLEAN, att_used, att_miss = build_axis(ATT_CLEAN_PAIRS)
print(f"  Used {len(att_used)}/{len(ATT_CLEAN_PAIRS)} pairs")
for a, c in att_used:
    print(f"    ({a}, {c})")
if att_miss:
    print(f"  Missing: {[(a, c) for a, c, _, _ in att_miss]}")

print("\n  Building INTENTION_CLEAN...")
INT_CLEAN, int_used, int_miss = build_axis(INT_CLEAN_PAIRS)
print(f"  Used {len(int_used)}/{len(INT_CLEAN_PAIRS)} pairs")
for a, c in int_used:
    print(f"    ({a}, {c})")
if int_miss:
    print(f"  Missing: {[(a, c) for a, c, _, _ in int_miss]}")

# THE TEST
print("\n" + "=" * 68)
print("THE TEST — cos(ATTENTION_CLEAN, INTENTION_CLEAN)")
print("=" * 68)
c_clean = cos(ATT_CLEAN, INT_CLEAN)
print(f"\n  cos(ATTENTION_CLEAN, INTENTION_CLEAN) = {c_clean:+.4f}")
print(f"  Baseline cos(A_orig, G_orig)          = {cos(A, G):+.4f}")
print(f"  Δ = {c_clean - cos(A, G):+.4f}")
print()
if c_clean < 0.20:
    print(f"  → STRONG: clean reconstruction recovers near-orthogonality.")
    print(f"           Original +0.56 was substantially anchor-vocabulary bias.")
    print(f"           Attention and intention ARE separable primitives.")
elif c_clean < 0.35:
    print(f"  → MODERATE: clean reconstruction substantially reduces coupling.")
    print(f"            Partial anchor bias + some substrate coupling.")
elif c_clean < 0.50:
    print(f"  → WEAK: cleaning anchors helps a bit but coupling largely persists.")
    print(f"        Substrate-real coupling probable, modest anchor contribution.")
else:
    print(f"  → NULL: cleaning anchors doesn't help. Coupling is substrate-real.")
    print(f"        Niamh's framing should accept that attention and intention")
    print(f"        couple in language about agents, OR seek the underlying primitive")
    print(f"        of which they are joint expressions.")

# Compare each clean axis to its original
print(f"\n  How do the clean axes relate to the originals?")
print(f"  cos(ATTENTION_CLEAN, A_orig) = {cos(ATT_CLEAN, A):+.4f}")
print(f"  cos(INTENTION_CLEAN, G_orig) = {cos(INT_CLEAN, G):+.4f}")
print(f"  cos(ATTENTION_CLEAN, G_orig) = {cos(ATT_CLEAN, G):+.4f}")
print(f"  cos(INTENTION_CLEAN, A_orig) = {cos(INT_CLEAN, A):+.4f}")

# Where do the clean axes sit in the basis?
print(f"\n  Clean axes vs the rest of the basis:")
print(f"  {'axis':>10s}  {'cos(., ATT_CLEAN)':>17s}  {'cos(., INT_CLEAN)':>17s}")
for name, vec in [("C", C), ("W", W), ("R", R), ("D", D), ("IO", IO),
                  ("MB", MB), ("GP", GP), ("COMP", COMP)]:
    print(f"  {name:>10s}  {cos(vec, ATT_CLEAN):+.4f}            "
          f"{cos(vec, INT_CLEAN):+.4f}")

# Pole vocabulary sanity check
print("\n  ATTENTION_CLEAN positive pole (top 10):")
for w, s in wv.similar_by_vector(ATT_CLEAN.astype(np.float32), topn=10):
    print(f"    {w:20s}  {s:+.4f}")
print("\n  INTENTION_CLEAN positive pole (top 10):")
for w, s in wv.similar_by_vector(INT_CLEAN.astype(np.float32), topn=10):
    print(f"    {w:20s}  {s:+.4f}")

# ===========================================================================
# PART B — intentional structure (subject/object of mental acts)
# ===========================================================================
print("\n" + "#" * 68)
print("# PART B — INTENTIONAL_STRUCTURE (subject vs object of mental acts)")
print("#" * 68)

# Pairs where positive = subject-of-act, negative = object-of-act.
# Filter to likely-in-vocab.
INT_STRUCT_PAIRS = [
    ("observer",    "observed"),
    ("subject",     "object"),
    ("witness",     "witnessed"),
    ("speaker",     "spoken"),
    ("writer",      "written"),
    ("teacher",     "taught"),
    ("giver",       "given"),
    ("lover",       "loved"),
    ("helper",      "helped"),
    ("perceiver",   "perceived"),  # perceiver may be OOV
    ("interpreter", "interpreted"),
    ("knower",      "known"),       # knower may be OOV
]

print("\n  Building INTENTIONAL_STRUCTURE...")
INT_STRUCT, is_used, is_miss = build_axis(INT_STRUCT_PAIRS)
print(f"  Used {len(is_used)}/{len(INT_STRUCT_PAIRS)} pairs")
for a, c in is_used:
    print(f"    ({a}, {c})")
if is_miss:
    print(f"  Missing: {[(a, c) for a, c, _, _ in is_miss]}")

print("\n" + "=" * 68)
print("Where does INTENTIONAL_STRUCTURE sit?")
print("=" * 68)
print("  (positive = subject/agent direction; negative = object/recipient direction)")
print()
print(f"  cos(INT_STRUCT, A)          = {cos(INT_STRUCT, A):+.4f}    "
      "(neg → A is object-side)")
print(f"  cos(INT_STRUCT, G)          = {cos(INT_STRUCT, G):+.4f}    "
      "(pos → G is subject-side)")
print(f"  cos(INT_STRUCT, GP)         = {cos(INT_STRUCT, GP):+.4f}")
print(f"  cos(INT_STRUCT, COMPLEMENT) = {cos(INT_STRUCT, COMP):+.4f}    "
      "(pos → COMP+ is subject-side)")
print()
print(f"  cos(INT_STRUCT, MB)         = {cos(INT_STRUCT, MB):+.4f}    "
      "(MB is self/other, different distinction)")
print()
print(f"  cos(INT_STRUCT, C)          = {cos(INT_STRUCT, C):+.4f}")
print(f"  cos(INT_STRUCT, W)          = {cos(INT_STRUCT, W):+.4f}")
print(f"  cos(INT_STRUCT, R)          = {cos(INT_STRUCT, R):+.4f}")
print(f"  cos(INT_STRUCT, D)          = {cos(INT_STRUCT, D):+.4f}")
print(f"  cos(INT_STRUCT, IO)         = {cos(INT_STRUCT, IO):+.4f}")

# Pole vocabulary
print("\n  INTENTIONAL_STRUCTURE positive pole (subject/agent side):")
for w, s in wv.similar_by_vector(INT_STRUCT.astype(np.float32), topn=10):
    print(f"    {w:20s}  {s:+.4f}")
print("\n  INTENTIONAL_STRUCTURE negative pole (object/recipient side):")
for w, s in wv.similar_by_vector((-INT_STRUCT).astype(np.float32), topn=10):
    print(f"    {w:20s}  {s:+.4f}")

# ===========================================================================
# Save
# ===========================================================================
np.savez("exp69_results.npz",
         ATTENTION_CLEAN=ATT_CLEAN,
         INTENTION_CLEAN=INT_CLEAN,
         INT_STRUCT=INT_STRUCT,
         cos_clean=c_clean,
         cos_clean_A=cos(ATT_CLEAN, A),
         cos_clean_G=cos(INT_CLEAN, G))
print("\nSaved exp69_results.npz")
