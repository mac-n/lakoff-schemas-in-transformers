"""
exp68 — Does TIME map onto COMPLEMENT?

Niamh's prediction: if ATTENTION is present-moment selectivity (GP) and
INTENTION = ATTENTION + future-directedness (COMPLEMENT-G-side), then
COMPLEMENT should have a substantial TIME signature.

TIME convention (from exp61b): TIME = mean(first - second) over pairs like
(past, future), (yesterday, tomorrow), (before, after).
  Positive TIME direction → PAST
  Negative TIME direction → FUTURE

COMPLEMENT convention (from exp67): unit(A - G) GS-orthogonalized.
  Positive direction → A-side (attention vocab, mostly noisy)
  Negative direction → G-side (intention vocab: pursue, motivated, aiming)

Prediction: COMPLEMENT's G-side (intention) should align with TIME's
future side. Equivalently: cos(-COMPLEMENT, -TIME) > 0,
i.e., cos(COMPLEMENT, TIME) > 0.

Or: cos(G, TIME) > 0 because G's intention content points to future.
"""

import numpy as np

print("Loading saved axes from exp60, 61b, 64, 67...")
exp60  = np.load("exp60_results.npz",  allow_pickle=True)
exp61b = np.load("exp61b_results.npz", allow_pickle=True)
exp64  = np.load("exp64_results.npz",  allow_pickle=True)
exp67  = np.load("exp67_results.npz",  allow_pickle=True)
basis_raw = exp60["basis_raw"].item()


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


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
TIME = unit(exp61b["TIME_proto"])

print("\n  TIME convention: positive direction = PAST; negative direction = FUTURE")
print("  COMPLEMENT convention: positive = A-side (attention); negative = G-side (intention)")

# ---------------------------------------------------------------------------
# Niamh's specific prediction
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Niamh's prediction: cos(COMPLEMENT, TIME) > 0 (intention/future-aligned)")
print("=" * 68)
c_comp_time = cos(COMP, TIME)
print(f"  cos(COMPLEMENT, TIME) = {c_comp_time:+.4f}")
if c_comp_time > 0.1:
    print(f"  → SUPPORTED: intention has substantial future-time signature")
elif c_comp_time > 0.03:
    print(f"  → WEAK SUPPORT: small future-time alignment")
elif c_comp_time > -0.03:
    print(f"  → NULL: COMPLEMENT is essentially time-neutral")
else:
    print(f"  → COUNTER: COMPLEMENT aligns with past, not future")

# ---------------------------------------------------------------------------
# Full picture: TIME vs each candidate
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("TIME cosines across all relevant axes")
print("=" * 68)
print("  (positive cos = past-aligned; negative cos = future-aligned)")
print()
print(f"  {'axis':>16s}  {'cos with TIME':>14s}  interpretation")
print(f"  {'-'*16}  {'-'*14}  {'-'*40}")
candidates = [
    ("A (attention)",            A),
    ("G (intention)",            G),
    ("GP (shared = attn?)",      GP),
    ("COMPLEMENT (intent?)",     COMP),
    ("C (reward)",               C),
    ("W (cost)",                 W),
    ("R (precision)",            R),
    ("D (compression)",          D),
    ("IO (containment)",         IO),
    ("MB (self/other)",          MB),
]
for name, vec in candidates:
    c = cos(vec, TIME)
    if c > 0.10:    interp = "PAST-leaning"
    elif c < -0.10: interp = "FUTURE-leaning"
    elif c > 0.03:  interp = "mildly past"
    elif c < -0.03: interp = "mildly future"
    else:           interp = "time-neutral"
    print(f"  {name:>16s}  {c:+.4f}        {interp}")

# ---------------------------------------------------------------------------
# Concept-word check: future-oriented vs past-oriented words
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Future/past concept words — checking time-direction sign convention")
print("=" * 68)
import gensim.downloader as api
print("\n  Loading GloVe (for concept-word checks)...")
wv = api.load("glove-wiki-gigaword-300")
print()

CHECK_WORDS = [
    # Time-explicit
    ("past",       "should be PAST-aligned"),
    ("future",     "should be FUTURE-aligned"),
    ("yesterday",  "should be PAST-aligned"),
    ("tomorrow",   "should be FUTURE-aligned"),
    ("memory",     "should be PAST-aligned"),
    ("plan",       "should be FUTURE-aligned"),
    # Intention-related (should be FUTURE if Niamh's right)
    ("intention",  "intention — predicted FUTURE if hypothesis holds"),
    ("intend",     "intend — predicted FUTURE"),
    ("plan",       "plan — predicted FUTURE"),
    ("goal",       "goal — predicted FUTURE"),
    ("purpose",    "purpose — predicted FUTURE"),
    ("ambition",   "ambition — predicted FUTURE"),
    ("anticipate", "anticipate — predicted FUTURE"),
    # Attention-related (should be ~time-neutral if Niamh's right)
    ("attention",  "attention — predicted NEUTRAL"),
    ("focus",      "focus — predicted NEUTRAL"),
    ("notice",     "notice — predicted NEUTRAL"),
    ("aware",      "aware — predicted NEUTRAL"),
    ("salient",    "salient — predicted NEUTRAL"),
]
print(f"  {'word':>14s}  {'cos(., TIME)':>12s}  {'cos(., COMP)':>12s}  prediction")
print(f"  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*40}")
for w, note in CHECK_WORDS:
    if w not in wv.key_to_index:
        print(f"  {w:>14s}  OOV")
        continue
    v = unit(wv[w])
    print(f"  {w:>14s}  {cos(v, TIME):+.4f}       "
          f"{cos(v, COMP):+.4f}       {note}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp68_results.npz",
         cos_COMP_TIME=c_comp_time,
         cos_GP_TIME=cos(GP, TIME),
         cos_A_TIME=cos(A, TIME),
         cos_G_TIME=cos(G, TIME))
print("\nSaved exp68_results.npz")
