"""
exp61 — Thread 0 — Asymmetric-collapse → R prediction.

Project UP/DOWN/LIGHT/DARK steering directions (word lists from exp3c and
exp4b — the experiments that produced loop-collapse at all-layer p=4.0)
onto R from exp60 (V+A-residualized perceptual-precision axis).

Predictions (per BASIS_TESTS_TODO.md Thread 0):
  cos(UP, R)    < 0   — cascade pole (loops at p=4.0)
  cos(DARK, R)  < 0   — cascade pole (loops at p=4.0)
  cos(DOWN, R)  >= 0  — does not loop
  cos(LIGHT, R) >= 0  — does not loop

Equivalently in axis-form:
  cos(A_UP_DOWN,    R) < 0   (UP_DOWN axis points UP-positive; UP is cascade)
  cos(A_LIGHT_DARK, R) > 0   (LIGHT_DARK axis points LIGHT-positive; DARK is cascade)
"""

import numpy as np
import gensim.downloader as api

# ---------------------------------------------------------------------------
# 1. Load substrate + R axis
# ---------------------------------------------------------------------------
print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 basis...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
R = basis_raw["R_per"]  # V+A-residualized, unit-normalized
R = R / np.linalg.norm(R)

# ---------------------------------------------------------------------------
# 2. Steering word lists from exp3c and exp4b (verbatim)
# ---------------------------------------------------------------------------
up_words = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
            "elevating", "uplifting", "higher", "upward"]
down_words = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
              "lowering", "collapsing", "lower", "downward"]

light_words = ["light", "bright", "illuminated", "shining", "clear",
               "luminous", "radiant", "glowing", "dawn", "sunshine"]
dark_words = ["dark", "darkness", "shadow", "obscure", "murky",
              "gloomy", "dim", "shadowy", "night", "blackness"]


def mean_vec(words, wv):
    vecs = []
    missing = []
    for w in words:
        if w in wv:
            vecs.append(wv[w])
        else:
            missing.append(w)
    if missing:
        print(f"  MISSING in GloVe: {missing}")
    return np.mean(vecs, axis=0), len(vecs)


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


# ---------------------------------------------------------------------------
# 3. Build pole-mean vectors and difference axes
# ---------------------------------------------------------------------------
print("\nBuilding pole means...")
up_mean,    n_up    = mean_vec(up_words,    wv)
down_mean,  n_down  = mean_vec(down_words,  wv)
light_mean, n_light = mean_vec(light_words, wv)
dark_mean,  n_dark  = mean_vec(dark_words,  wv)
print(f"  UP: {n_up}/{len(up_words)}, DOWN: {n_down}/{len(down_words)}, "
      f"LIGHT: {n_light}/{len(light_words)}, DARK: {n_dark}/{len(dark_words)}")

A_UP_DOWN    = up_mean    - down_mean
A_LIGHT_DARK = light_mean - dark_mean

# ---------------------------------------------------------------------------
# 4. The four predictions (pole-anchored form, matching TODO phrasing)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("THREAD 0 — pole-anchored cosines with R")
print("=" * 68)

results = [
    ("UP",    up_mean,    "< 0", "cascade — loops at p=4.0"),
    ("DOWN",  down_mean,  ">= 0", "does not loop"),
    ("LIGHT", light_mean, ">= 0", "does not loop"),
    ("DARK",  dark_mean,  "< 0", "cascade — loops at p=4.0"),
]

print(f"  {'pole':6s} {'cos(pole, R)':>14s}  {'predicted':>8s}  {'verdict':6s}  notes")
print(f"  {'-'*6} {'-'*14}  {'-'*8}  {'-'*6}  {'-'*40}")
hits = 0
for name, vec, pred, note in results:
    c = cos(vec, R)
    if "< 0" in pred:
        ok = c < 0
    else:
        ok = c >= 0
    verdict = "PASS" if ok else "FAIL"
    hits += int(ok)
    print(f"  {name:6s} {c:+.4f}         {pred:>8s}  {verdict:6s}  {note}")

print(f"\n  Hits: {hits}/4")

# ---------------------------------------------------------------------------
# 5. Difference-axis cosines (anisotropy-cancelled form)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Difference-axis cosines (anisotropy-cancelled)")
print("=" * 68)
c_ud = cos(A_UP_DOWN, R)
c_ld = cos(A_LIGHT_DARK, R)
print(f"  cos(A_UP_DOWN,    R) = {c_ud:+.4f}   "
      f"(predicted NEGATIVE: UP is cascade pole)  "
      f"{'PASS' if c_ud < 0 else 'FAIL'}")
print(f"  cos(A_LIGHT_DARK, R) = {c_ld:+.4f}   "
      f"(predicted POSITIVE: LIGHT is anti-cascade, DARK is cascade)  "
      f"{'PASS' if c_ld > 0 else 'FAIL'}")

# ---------------------------------------------------------------------------
# 6. Sanity: pole-norms (for interpreting magnitudes vs anisotropy)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Sanity checks")
print("=" * 68)
print(f"  |UP_mean|    = {np.linalg.norm(up_mean):.3f}")
print(f"  |DOWN_mean|  = {np.linalg.norm(down_mean):.3f}")
print(f"  |LIGHT_mean| = {np.linalg.norm(light_mean):.3f}")
print(f"  |DARK_mean|  = {np.linalg.norm(dark_mean):.3f}")
print(f"  |A_UP_DOWN|    = {np.linalg.norm(A_UP_DOWN):.3f}  (offset cancels shared anisotropy)")
print(f"  |A_LIGHT_DARK| = {np.linalg.norm(A_LIGHT_DARK):.3f}")
print(f"  |R| = {np.linalg.norm(R):.6f}  (should be ~1.0)")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
np.savez("exp61_results.npz",
         A_UP_DOWN=A_UP_DOWN,
         A_LIGHT_DARK=A_LIGHT_DARK,
         R=R,
         cos_UP_R=cos(up_mean, R),
         cos_DOWN_R=cos(down_mean, R),
         cos_LIGHT_R=cos(light_mean, R),
         cos_DARK_R=cos(dark_mean, R),
         cos_UP_DOWN_axis_R=c_ud,
         cos_LIGHT_DARK_axis_R=c_ld)
print("\nSaved exp61_results.npz")
