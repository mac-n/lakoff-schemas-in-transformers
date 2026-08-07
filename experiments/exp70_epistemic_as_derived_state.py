"""
exp70 — Test Niamh's hypothesis: is EPISTEMIC_VALUE a derived state in
(ATTENTION × INTENTION) space rather than its own primitive?

Hypothesis: low INTENTION + high ATTENTION = gathering-information / exploration.
Maps to active-inference: high γ + low π → explore mode (emergent from
precision dynamics, not a separate quantity).

Predictions:
  CURIOSITY/EXPLORATION vocab: cos(., ATT_CLEAN) > 0  AND  cos(., INT_CLEAN) ≤ 0
  PURSUIT/EXECUTION vocab:     cos(., ATT_CLEAN) > 0  AND  cos(., INT_CLEAN) > 0
  Derived axis: EXPLORATION = unit(ATT_CLEAN - INT_CLEAN)
                Curiosity vocab should load strongly POSITIVE on EXPLORATION.
                Pursuit vocab should load NEAR-ZERO or NEGATIVE on EXPLORATION.

If confirmed: drop target_EPISTEMIC_VALUE from priority queue. Epistemic
value is a derived cognitive state, not a separate basis primitive.
If refuted: proceed with target_EPISTEMIC_VALUE construction.
"""

import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading ATT_CLEAN and INT_CLEAN from exp69...")
exp69 = np.load("exp69_results.npz", allow_pickle=True)
ATT = exp69["ATTENTION_CLEAN"]
INT = exp69["INTENTION_CLEAN"]


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


ATT = unit(ATT)
INT = unit(INT)
EXPLORATION = unit(ATT - INT)   # derived axis: high attention, low intention

print(f"\n  cos(ATT_CLEAN, INT_CLEAN) = {cos(ATT, INT):+.4f}  "
      f"(near-zero — they're orthogonal primitives)")
print(f"  EXPLORATION = unit(ATT - INT)  "
      f"(the predicted gathering-information direction)")

# ---------------------------------------------------------------------------
# Concept-word batteries
# ---------------------------------------------------------------------------
CURIOSITY = [
    "curiosity", "curious", "inquisitive",
    "wondering", "wonder",
    "investigating", "investigation", "investigate",
    "exploring", "exploration", "explore",
    "intrigued", "interested", "fascinated",
    "puzzled", "questioning", "questioned",
    "probing", "probe",
    "seeking", "searching", "search",
    "inquiry", "inquiring",
    "experimenting", "experiment",
    "browsing", "scanning", "surveying",
    "studying", "examining",
]

PURSUIT = [
    "pursuing", "pursuit", "pursue",
    "committing", "commitment", "commit",
    "executing", "execution", "execute",
    "accomplishing", "accomplishment", "accomplish",
    "achieving", "achievement", "achieve",
    "finishing", "completing", "completion",
    "succeeding", "success",
    "delivering", "delivery",
    "implementing", "implementation",
    "fulfilling", "fulfillment",
]

DRIFT = [   # low ATT + low INT predicted
    "drifting", "drift",
    "daydreaming",
    "wandering", "wander",
    "spacing-out", "spaced",   # multi-word may fail
    "distracted", "distraction",
    "unfocused", "absent-minded",
    "idle", "idling",
    "lazy", "lethargic", "listless",
    "bored", "boredom",
]

# ---------------------------------------------------------------------------
# Project each concept word onto ATT, INT, and EXPLORATION
# ---------------------------------------------------------------------------
def project_battery(name, words):
    print(f"\n{'='*68}")
    print(f"  {name}")
    print(f"{'='*68}")
    print(f"  {'word':>18s}  {'cos(., ATT)':>11s}  {'cos(., INT)':>11s}  "
          f"{'cos(., EXPL)':>12s}  {'quadrant'}")
    print(f"  {'-'*18}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*15}")
    rows = []
    for w in words:
        if w not in wv.key_to_index:
            print(f"  {w:>18s}  OOV")
            continue
        v = unit(wv[w])
        c_att = cos(v, ATT)
        c_int = cos(v, INT)
        c_exp = cos(v, EXPLORATION)
        # Quadrant label
        if c_att > 0 and c_int > 0:    q = "+ATT +INT (exec)"
        elif c_att > 0 and c_int <= 0: q = "+ATT -INT (EXPL)"
        elif c_att <= 0 and c_int > 0: q = "-ATT +INT (auto)"
        else:                          q = "-ATT -INT (drift)"
        rows.append((w, c_att, c_int, c_exp))
        print(f"  {w:>18s}  {c_att:+.4f}     {c_int:+.4f}     "
              f"{c_exp:+.4f}      {q}")
    return rows


curiosity_rows = project_battery("CURIOSITY / EXPLORATION (predicted: +ATT, ≤0 INT, +EXPL)",
                                  CURIOSITY)
pursuit_rows   = project_battery("PURSUIT / EXECUTION (predicted: +ATT, +INT, ≤0 EXPL)",
                                  PURSUIT)
drift_rows     = project_battery("DRIFT / DISENGAGEMENT (predicted: -ATT, -INT)",
                                  DRIFT)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)


def stats(name, rows):
    if not rows:
        return
    c_atts = [r[1] for r in rows]
    c_ints = [r[2] for r in rows]
    c_exps = [r[3] for r in rows]
    print(f"\n  {name}  (n={len(rows)})")
    print(f"    mean cos(., ATT)  = {np.mean(c_atts):+.4f}   "
          f"({sum(1 for c in c_atts if c > 0)}/{len(rows)} positive)")
    print(f"    mean cos(., INT)  = {np.mean(c_ints):+.4f}   "
          f"({sum(1 for c in c_ints if c > 0)}/{len(rows)} positive)")
    print(f"    mean cos(., EXPL) = {np.mean(c_exps):+.4f}   "
          f"({sum(1 for c in c_exps if c > 0)}/{len(rows)} positive)")


stats("CURIOSITY", curiosity_rows)
stats("PURSUIT",   pursuit_rows)
stats("DRIFT",     drift_rows)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("VERDICT on hypothesis: EPISTEMIC_VALUE as derived state in (ATT × INT)")
print("=" * 68)

cur_atts = [r[1] for r in curiosity_rows]
cur_ints = [r[2] for r in curiosity_rows]
cur_exps = [r[3] for r in curiosity_rows]
pur_atts = [r[1] for r in pursuit_rows]
pur_ints = [r[2] for r in pursuit_rows]
pur_exps = [r[3] for r in pursuit_rows]

curiosity_in_quadrant = sum(1 for a, i in zip(cur_atts, cur_ints)
                             if a > 0 and i <= 0)
pursuit_in_quadrant   = sum(1 for a, i in zip(pur_atts, pur_ints)
                             if a > 0 and i > 0)

print(f"\n  Curiosity vocab in (+ATT, -INT) quadrant: "
      f"{curiosity_in_quadrant}/{len(curiosity_rows)}")
print(f"  Pursuit vocab in (+ATT, +INT) quadrant:   "
      f"{pursuit_in_quadrant}/{len(pursuit_rows)}")
print(f"\n  Mean cos(curiosity, EXPLORATION) = {np.mean(cur_exps):+.4f}")
print(f"  Mean cos(pursuit,   EXPLORATION) = {np.mean(pur_exps):+.4f}")
print(f"  Gap (curiosity - pursuit on EXPL) = "
      f"{np.mean(cur_exps) - np.mean(pur_exps):+.4f}")

if np.mean(cur_exps) > 0.10 and np.mean(pur_exps) < 0.05:
    verdict = ("STRONG SUPPORT: curiosity loads on EXPLORATION; pursuit doesn't.\n"
               "  → Hypothesis confirmed: epistemic value is a derived state.\n"
               "  → Drop target_EPISTEMIC_VALUE from basis queue.")
elif np.mean(cur_exps) > 0.05 and np.mean(cur_exps) > np.mean(pur_exps) + 0.05:
    verdict = ("MODERATE SUPPORT: curiosity loads more on EXPLORATION than pursuit does.\n"
               "  → Hypothesis partially supported; some epistemic content may be derived.\n"
               "  → Build target_EPISTEMIC_VALUE anyway to see if it captures residual content.")
elif abs(np.mean(cur_exps) - np.mean(pur_exps)) < 0.05:
    verdict = ("WEAK / NULL: curiosity and pursuit load similarly on EXPLORATION.\n"
               "  → EXPLORATION axis isn't differentiating exploration from execution.\n"
               "  → Build target_EPISTEMIC_VALUE — separate primitive likely needed.")
else:
    verdict = ("COUNTER: pursuit loads more on EXPLORATION than curiosity does.\n"
               "  → Hypothesis refuted as constructed; check anchor design.")
print(f"\n  {verdict}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp70_results.npz",
         EXPLORATION_axis=EXPLORATION,
         mean_curiosity_EXPL=np.mean(cur_exps),
         mean_pursuit_EXPL=np.mean(pur_exps),
         mean_drift_ATT=np.mean([r[1] for r in drift_rows]) if drift_rows else 0,
         mean_drift_INT=np.mean([r[2] for r in drift_rows]) if drift_rows else 0)
print("\nSaved exp70_results.npz")
