"""
exp61b — Thread 0 supplementary.

Two follow-ups to exp61's null result:

(1) MML-extended UP_DOWN and LIGHT_DARK axes (broader vocab than the 10-word
    steering lists). From BASIS_REFERENCE §4 the MML versions have R-coords
    UD=-0.091 and LD=+0.144 — predicted directions both showing through.
    Does the full MML axis-vs-R cosine recover the prediction?

(2) Niamh's question: the four steering-pole means all clustered around
    cos(pole, R) ~ -0.08 to -0.12. Is that shared anisotropy direction
    something meaningful — maybe the missing TIME axis? Decompose the
    common-pole-mean onto the 7-axis basis and see what's actually there.
"""

import numpy as np
import gensim.downloader as api

from lakoff_canonical_vocabulary import UP_DOWN_MML, LIGHT_DARK_MML

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 basis...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()
basis_order = ["C_rew", "W_wgt", "A_aff", "G_pol", "R_per", "D_cmp", "IO_blk"]
B = np.stack([basis_raw[k] / np.linalg.norm(basis_raw[k]) for k in basis_order])
# B is (7, 300), each row unit-normalized

R = basis_raw["R_per"]
R = R / np.linalg.norm(R)


def build_axis_from_pairs(pairs):
    """exp60 convention: mean of (w_a - w_c) over in-vocab pairs, unit-norm."""
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    raw = np.stack(offs).mean(axis=0)
    return raw, raw / np.linalg.norm(raw), len(offs), len(pairs)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Part 1 — MML-extended UD and LD axes vs R
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("PART 1 — MML-extended axes vs R")
print("=" * 68)

UD_raw, UD, n_ud, n_ud_tot = build_axis_from_pairs(UP_DOWN_MML)
LD_raw, LD, n_ld, n_ld_tot = build_axis_from_pairs(LIGHT_DARK_MML)
print(f"  UP_DOWN_MML:    {n_ud}/{n_ud_tot} pairs in vocab")
print(f"  LIGHT_DARK_MML: {n_ld}/{n_ld_tot} pairs in vocab")

c_ud_R = cos(UD, R)
c_ld_R = cos(LD, R)

print(f"\n  cos(UP_DOWN_MML,    R) = {c_ud_R:+.4f}   "
      f"(prediction: negative if UP-pole is cascade)  "
      f"{'PASS' if c_ud_R < 0 else 'FAIL'}")
print(f"  cos(LIGHT_DARK_MML, R) = {c_ld_R:+.4f}   "
      f"(prediction: positive if DARK-pole is cascade)  "
      f"{'PASS' if c_ld_R > 0 else 'FAIL'}")

print(f"\n  Comparison vs narrow steering vocab (exp61):")
print(f"    UD: narrow {-0.0186:+.4f}  vs MML {c_ud_R:+.4f}")
print(f"    LD: narrow {-0.0024:+.4f}  vs MML {c_ld_R:+.4f}")
print(f"  (BASIS_REFERENCE §4 reported UD R-coord -0.091, LD R-coord +0.144 — "
      "those are post-GS-orthogonalized coordinates, this is raw cos.)")

# ---------------------------------------------------------------------------
# Part 2 — Diagnostic: what IS the shared anisotropy direction?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("PART 2 — Shared anisotropy diagnostic")
print("=" * 68)

# Rebuild exp61's pole means
up_words    = ["up", "rising", "lifting", "ascending", "climbing", "soaring",
               "elevating", "uplifting", "higher", "upward"]
down_words  = ["down", "falling", "sinking", "descending", "dropping", "plummeting",
               "lowering", "collapsing", "lower", "downward"]
light_words = ["light", "bright", "illuminated", "shining", "clear",
               "luminous", "radiant", "glowing", "dawn", "sunshine"]
dark_words  = ["dark", "darkness", "shadow", "obscure", "murky",
               "gloomy", "dim", "shadowy", "night", "blackness"]


def pole_mean(words):
    return np.mean([wv[w] for w in words if w in wv], axis=0)


up_m, down_m, light_m, dark_m = (pole_mean(up_words), pole_mean(down_words),
                                  pole_mean(light_words), pole_mean(dark_words))

# Shared direction = mean of all four pole-means. This is the component
# common to "concrete spatial/visual happening" vocabulary regardless of polarity.
shared = (up_m + down_m + light_m + dark_m) / 4.0
shared_unit = shared / np.linalg.norm(shared)
print(f"  |shared pole-mean direction| = {np.linalg.norm(shared):.3f}")
print(f"  cos(shared, R) = {cos(shared, R):+.4f}   "
      "(this is what dragged all four poles into the -0.1 zone)")

print(f"\n  Decomposition onto 7-axis basis (unit-normalized shared direction):")
print(f"  {'axis':>8s}  {'cos':>8s}  {'meaning'}")
print(f"  {'-'*8}  {'-'*8}  {'-'*40}")
meanings = {
    "C_rew":  "integrated reward / wellbeing",
    "W_wgt":  "weight / cost / effort burden",
    "A_aff":  "affect (Russell V×A diagonal)",
    "G_pol":  "policy precision / goal-directedness",
    "R_per":  "perceptual precision / regulability",
    "D_cmp":  "compression / surprisal / predictability",
    "IO_blk": "container-topology",
}
proj_on_basis = {}
for axis_name in basis_order:
    axis = basis_raw[axis_name] / np.linalg.norm(basis_raw[axis_name])
    c = cos(shared_unit, axis)
    proj_on_basis[axis_name] = c
    print(f"  {axis_name:>8s}  {c:+.4f}   {meanings[axis_name]}")

# How much of the shared direction lives in the 7-axis basis?
proj_vec = np.zeros_like(shared_unit)
# Naive sum (NOT Gram-Schmidt — basis isn't orthogonal):
shared_in_basis = 0.0
for axis_name in basis_order:
    axis = basis_raw[axis_name] / np.linalg.norm(basis_raw[axis_name])
    proj_vec += (shared_unit @ axis) * axis
explained = np.linalg.norm(proj_vec)**2  # rough: not GS so not strictly variance-explained
print(f"\n  ||sum of axis-projections||^2 = {explained:.4f}  "
      f"(rough basis-coverage indicator, not GS-corrected)")

# Test Niamh's TIME hypothesis directly: build a quick proto-TIME axis
# from the Thread 2.5 anchor candidates that are in-vocab.
print("\n" + "-" * 68)
print("  TIME hypothesis check (quick proto-TIME from Thread 2.5 anchors):")
print("-" * 68)

TIME_PAIRS = [
    ("past", "future"),
    ("yesterday", "tomorrow"),
    ("before", "after"),
    ("earlier", "later"),
    ("ancient", "modern"),
    ("old", "new"),
    ("remembered", "anticipated"),
    ("begun", "pending"),
    ("completed", "planned"),
    ("precedes", "follows"),
    ("historical", "upcoming"),
]
TIME_raw, TIME, n_t, n_t_tot = build_axis_from_pairs(TIME_PAIRS)
print(f"  TIME axis built from {n_t}/{n_t_tot} in-vocab pairs.")
print(f"  cos(shared, TIME) = {cos(shared_unit, TIME):+.4f}")
print(f"  cos(TIME, R)      = {cos(TIME, R):+.4f}")
print(f"  |TIME projection onto shared direction| / |shared| = "
      f"{abs(cos(shared_unit, TIME)):.4f}")

# Also test: how time-loaded are the individual steering words?
# Compare TIME with each pole separately
print(f"\n  cos(pole, TIME) for each steering pole:")
for name, vec in [("UP", up_m), ("DOWN", down_m),
                  ("LIGHT", light_m), ("DARK", dark_m)]:
    print(f"    {name:6s}  cos(pole, TIME) = {cos(vec, TIME):+.4f}")

# And: how much does the shared direction look like TIME compared to other axes?
print(f"\n  Ranked: shared-direction cosines, |cos| descending")
ranked = sorted([("R_per",  cos(shared_unit, basis_raw["R_per"]/np.linalg.norm(basis_raw["R_per"]))),
                 ("C_rew",  cos(shared_unit, basis_raw["C_rew"]/np.linalg.norm(basis_raw["C_rew"]))),
                 ("W_wgt",  cos(shared_unit, basis_raw["W_wgt"]/np.linalg.norm(basis_raw["W_wgt"]))),
                 ("A_aff",  cos(shared_unit, basis_raw["A_aff"]/np.linalg.norm(basis_raw["A_aff"]))),
                 ("G_pol",  cos(shared_unit, basis_raw["G_pol"]/np.linalg.norm(basis_raw["G_pol"]))),
                 ("D_cmp",  cos(shared_unit, basis_raw["D_cmp"]/np.linalg.norm(basis_raw["D_cmp"]))),
                 ("IO_blk", cos(shared_unit, basis_raw["IO_blk"]/np.linalg.norm(basis_raw["IO_blk"]))),
                 ("TIME?",  cos(shared_unit, TIME))],
                key=lambda x: -abs(x[1]))
for axis_name, c in ranked:
    bar = "#" * int(abs(c) * 50)
    print(f"    {axis_name:>8s}  {c:+.4f}  {bar}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez("exp61b_results.npz",
         UD_MML=UD, LD_MML=LD,
         cos_UD_MML_R=c_ud_R,
         cos_LD_MML_R=c_ld_R,
         shared_pole_direction=shared_unit,
         TIME_proto=TIME,
         cos_shared_R=cos(shared, R),
         cos_shared_TIME=cos(shared_unit, TIME),
         cos_TIME_R=cos(TIME, R),
         shared_basis_decomp=np.array([proj_on_basis[k] for k in basis_order]))
print("\nSaved exp61b_results.npz")
