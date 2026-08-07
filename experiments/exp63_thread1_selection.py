"""
exp63 — Thread 1 — target_SELECTION construction.

THE HINGE for the basis-collapse hypothesis.

Build SELECTION from verb-form active-inference primitive vocabulary
(selected/rejected, chose/refused, etc.) and run the full test battery:

1. Orthogonality / overlap with all 7 basis axes.
2. cos(SELECTION, IO_CLEAN) specifically — does SELECTION reframe IO?
3. cos(SELECTION, G) — does SELECTION subsume G? (Thread 2.7c)
4. A onto SELECTION alone — does SELECTION cover more of A than W+R's 6.6% bar?
5. G onto SELECTION — Thread 2.7c verdict on whether G drops to SELECTION
6. A onto (G, SELECTION) — Thread 2.7b repurposed, full A decomposition
7. A onto (W, R, G, SELECTION) — bonus four-axis check
8. Concept-word projections onto SELECTION — does it pick up
   self/identity/intimacy/alienation that IO_CLEAN missed?
9. Pole vocabulary — top nearest-neighbour words at each pole for face-validity

Substrate: glove-wiki-gigaword-300. Basis: exp60_results.npz.
"""

import numpy as np
import gensim.downloader as api

# ---------------------------------------------------------------------------
# Anchor pairs (from BASIS_TESTS_TODO.md Thread 1)
# ---------------------------------------------------------------------------
SELECTION_PAIRS = [
    ("selected",   "rejected"),
    ("chose",      "refused"),
    ("picked",     "discarded"),
    ("admitted",   "denied"),
    ("accepted",   "declined"),
    ("kept",       "removed"),
    ("chosen",     "eliminated"),
    ("preferred",  "overlooked"),
    ("favored",    "excluded"),
    ("designated", "omitted"),
    # "singled-out" is hyphenated — try variants
    ("highlighted", "neglected"),
]
# Note: "singled-out / ignored" dropped — hyphenated form unlikely in GloVe.
# 11 pairs remain. May be reduced further by in-vocab check below.

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")

print("Loading exp60 basis...")
exp60 = np.load("exp60_results.npz", allow_pickle=True)
basis_raw = exp60["basis_raw"].item()


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
            missing.append((a, c, a not in wv.key_to_index, c not in wv.key_to_index))
    raw = np.stack(offs).mean(axis=0)
    return unit(raw), used, missing


# ---------------------------------------------------------------------------
# Build SELECTION axis
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Building SELECTION axis")
print("=" * 68)
SELECTION, used_pairs, missing_pairs = build_axis(SELECTION_PAIRS)
print(f"  Used: {len(used_pairs)}/{len(SELECTION_PAIRS)} pairs")
for a, c in used_pairs:
    print(f"    ({a}, {c})")
if missing_pairs:
    print(f"  Missing:")
    for a, c, a_miss, c_miss in missing_pairs:
        print(f"    ({a}, {c}) — "
              f"{'a OOV' if a_miss else ''}{', ' if a_miss and c_miss else ''}"
              f"{'c OOV' if c_miss else ''}")

# ---------------------------------------------------------------------------
# (1) + (4) Orthogonality with all 7 basis axes
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 1+4 — cos(SELECTION, each basis axis)")
print("=" * 68)
basis_order = ["C_rew", "W_wgt", "A_aff", "G_pol", "R_per", "D_cmp", "IO_blk"]
meanings = {
    "C_rew":  "integrated reward",
    "W_wgt":  "weight / cost",
    "A_aff":  "affect",
    "G_pol":  "policy precision",
    "R_per":  "perceptual precision",
    "D_cmp":  "compression",
    "IO_blk": "container-topology",
}
sel_cos = {}
print(f"  {'axis':>8s}  {'cos':>8s}  bar                                              meaning")
print(f"  {'-'*8}  {'-'*8}  {'-'*40}  {'-'*30}")
for name in basis_order:
    ax = unit(basis_raw[name])
    c = cos(SELECTION, ax)
    sel_cos[name] = c
    bar = "#" * int(abs(c) * 50)
    sign = "+" if c >= 0 else "-"
    print(f"  {name:>8s}  {c:+.4f}  {bar:<40s}  {meanings[name]}")

# ---------------------------------------------------------------------------
# (2) SELECTION vs IO_CLEAN — the IO-reframe question
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 2 — SELECTION vs IO_CLEAN (the IO-reframe question)")
print("=" * 68)
c_io = sel_cos["IO_blk"]
print(f"  cos(SELECTION, IO_CLEAN) = {c_io:+.4f}")
if abs(c_io) > 0.5:
    print(f"  → SAME AXIS — SELECTION is essentially IO in verb-form")
elif abs(c_io) > 0.2:
    print(f"  → RELATED — they share substantial content but are distinct")
else:
    print(f"  → DISTINCT — SELECTION is not an IO reframe; they're independent primitives")

# ---------------------------------------------------------------------------
# (3) SELECTION vs G — the G-subsumption question (Thread 2.7c)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 3 — SELECTION vs G (Thread 2.7c — does SELECTION subsume G?)")
print("=" * 68)
c_g = sel_cos["G_pol"]
print(f"  cos(SELECTION, G_pol) = {c_g:+.4f}")
if abs(c_g) > 0.7:
    print(f"  → STRONG OVERLAP — G likely reduces to SELECTION")
elif abs(c_g) > 0.4:
    print(f"  → MODERATE OVERLAP — SELECTION captures part of G; G has additional content")
else:
    print(f"  → WEAK OVERLAP — SELECTION and G are largely independent")

# Quantitative: project G onto SELECTION and compute residual
A = unit(basis_raw["A_aff"])
W = unit(basis_raw["W_wgt"])
R = unit(basis_raw["R_per"])
G = unit(basis_raw["G_pol"])

gamma_g = G @ SELECTION
G_hat = gamma_g * SELECTION
G_residual = G - G_hat
G_res_mag = np.linalg.norm(G_residual)
G_explained_by_SEL = 1.0 - G_res_mag**2  # |G|=1 so this is variance frac

print(f"\n  Quantitative — G onto SELECTION:")
print(f"    γ (SELECTION coeff)              = {gamma_g:+.4f}")
print(f"    |G - γ·SELECTION| / |G|          = {G_res_mag:.4f}")
print(f"    variance of G explained by SEL   = {G_explained_by_SEL:.4f} "
      f"({100*G_explained_by_SEL:.1f}%)")
if G_res_mag < 0.2:
    print(f"    VERDICT: G DROPS FROM BASIS — it was always SELECTION-aligned")
elif G_res_mag < 0.5:
    print(f"    VERDICT: G partially reduces to SELECTION; "
          f"residual has independent content")
else:
    print(f"    VERDICT: G keeps substantial independent content beyond SELECTION")

# ---------------------------------------------------------------------------
# (4) A onto SELECTION alone — does SELECTION beat the W+R 6.6% bar from exp62?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 4 — A onto SELECTION alone (vs exp62's W+R bar of 6.6%)")
print("=" * 68)
gamma_a = A @ SELECTION
A_hat_sel = gamma_a * SELECTION
A_res_sel = A - A_hat_sel
A_res_sel_mag = np.linalg.norm(A_res_sel)
A_explained_by_SEL = 1.0 - A_res_sel_mag**2

print(f"  γ (SELECTION coeff)              = {gamma_a:+.4f}")
print(f"  |A - γ·SELECTION| / |A|          = {A_res_sel_mag:.4f}")
print(f"  variance of A explained by SEL   = {A_explained_by_SEL:.4f} "
      f"({100*A_explained_by_SEL:.1f}%)")
print(f"  ── for comparison ──")
print(f"  variance of A explained by (W, R)   = 6.6% (exp62)")
print(f"  variance of A explained by G alone  = {(A @ G)**2:.4f} "
      f"({100*(A @ G)**2:.1f}%)")

# ---------------------------------------------------------------------------
# (5/6) A onto (G, SELECTION) and (W, R, G, SELECTION)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 5+6 — A onto multi-axis subspaces")
print("=" * 68)


def regress(target, basis_vecs, names):
    M = np.stack(basis_vecs, axis=1)  # (300, k)
    gram = M.T @ M
    rhs = M.T @ target
    coeffs = np.linalg.solve(gram, rhs)
    target_hat = M @ coeffs
    res = target - target_hat
    res_mag = np.linalg.norm(res)
    explained = 1.0 - res_mag**2 / np.linalg.norm(target)**2
    return coeffs, res_mag, explained, res


# A onto (G, SELECTION)
coeffs, res_mag, explained, _ = regress(A, [G, SELECTION], ["G", "SELECTION"])
print(f"\n  A onto (G, SELECTION):")
print(f"    coeffs: G={coeffs[0]:+.4f}, SELECTION={coeffs[1]:+.4f}")
print(f"    residual / |A| = {res_mag:.4f}")
print(f"    variance explained = {explained:.4f} ({100*explained:.1f}%)")

# A onto (W, R, SELECTION) — original Thread 2.7(b)
coeffs, res_mag, explained, _ = regress(A, [W, R, SELECTION], ["W", "R", "SELECTION"])
print(f"\n  A onto (W, R, SELECTION) — original Thread 2.7(b) framing:")
print(f"    coeffs: W={coeffs[0]:+.4f}, R={coeffs[1]:+.4f}, SELECTION={coeffs[2]:+.4f}")
print(f"    residual / |A| = {res_mag:.4f}")
print(f"    variance explained = {explained:.4f} ({100*explained:.1f}%)")

# A onto (W, R, G, SELECTION) — full four-axis
coeffs, res_mag, explained, _ = regress(A, [W, R, G, SELECTION], ["W", "R", "G", "SELECTION"])
print(f"\n  A onto (W, R, G, SELECTION) — full four-axis decomposition:")
print(f"    coeffs: W={coeffs[0]:+.4f}, R={coeffs[1]:+.4f}, "
      f"G={coeffs[2]:+.4f}, SELECTION={coeffs[3]:+.4f}")
print(f"    residual / |A| = {res_mag:.4f}")
print(f"    variance explained = {explained:.4f} ({100*explained:.1f}%)")

# ---------------------------------------------------------------------------
# (7) IO_CLEAN onto SELECTION — symmetric Thread 1 question
# ---------------------------------------------------------------------------
IO = unit(basis_raw["IO_blk"])
gamma_io = IO @ SELECTION
IO_hat = gamma_io * SELECTION
IO_res = IO - IO_hat
IO_res_mag = np.linalg.norm(IO_res)
print(f"\n  IO_CLEAN onto SELECTION:")
print(f"    γ = {gamma_io:+.4f}")
print(f"    |IO - γ·SELECTION| / |IO| = {IO_res_mag:.4f}")
print(f"    variance of IO explained by SEL = {1-IO_res_mag**2:.4f} "
      f"({100*(1-IO_res_mag**2):.1f}%)")

# ---------------------------------------------------------------------------
# (8) Concept-word projections — does SELECTION pick up what IO missed?
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 8 — Concept-word projections onto SELECTION")
print("=" * 68)

CONCEPT_GROUPS = {
    "IO-probe (self/other/boundary)": [
        "self", "selfhood", "identity", "boundary", "intimacy",
        "belonging", "alienation", "isolation", "communion", "exile",
        "membership", "kinship", "loneliness", "togetherness",
    ],
    "Agency / action": [
        "agency", "freedom", "hope", "growth", "play", "creativity",
        "decision", "choice", "judgment", "will", "preference",
    ],
    "Clinical / affective": [
        "anxiety", "fear", "panic", "trauma", "depression",
        "shame", "guilt", "rage", "joy", "love",
    ],
    "Contemplative": [
        "awe", "sublime", "transcendence", "ineffable", "presence",
        "ritual", "meditation", "flow",
    ],
    "Random (control)": [
        "sausage", "pyjamas", "marigold", "stapler", "pebble",
    ],
}

# For comparison: IO_CLEAN projections (from BASIS_REFERENCE §5 IO-probe table)
# we'll just report SELECTION projections here, and pull IO from basis directly
print(f"\n  {'word':>16s}  {'cos(word, SEL)':>14s}  {'cos(word, IO)':>13s}  "
      f"{'cos(word, G)':>12s}")
print(f"  {'-'*16}  {'-'*14}  {'-'*13}  {'-'*12}")

projections_by_group = {}
for group_name, words in CONCEPT_GROUPS.items():
    print(f"\n  -- {group_name} --")
    rows = []
    for w in words:
        if w not in wv.key_to_index:
            print(f"  {w:>16s}  OOV")
            continue
        v = unit(wv[w])
        cs = cos(v, SELECTION)
        ci = cos(v, IO)
        cg = cos(v, G)
        rows.append((w, cs, ci, cg))
        print(f"  {w:>16s}  {cs:+.4f}         {ci:+.4f}         {cg:+.4f}")
    projections_by_group[group_name] = rows

# ---------------------------------------------------------------------------
# (9) Pole vocabulary — face-validity
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("Test 9 — Pole vocabulary (top-20 nearest-neighbour words each pole)")
print("=" * 68)
# wv is a KeyedVectors. To find words closest to SELECTION direction:
# similar_by_vector returns top-N by cosine similarity.

print("\n  POSITIVE pole (selected / chose / kept side):")
positive_neighbors = wv.similar_by_vector(SELECTION.astype(np.float32), topn=20)
for w, s in positive_neighbors:
    print(f"    {w:25s}  {s:+.4f}")

print("\n  NEGATIVE pole (rejected / refused / removed side):")
negative_neighbors = wv.similar_by_vector((-SELECTION).astype(np.float32), topn=20)
for w, s in negative_neighbors:
    print(f"    {w:25s}  {s:+.4f}")

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
np.savez("exp63_results.npz",
         SELECTION=SELECTION,
         cos_with_basis=np.array([sel_cos[k] for k in basis_order]),
         basis_order=np.array(basis_order),
         G_explained_by_SEL=G_explained_by_SEL,
         A_explained_by_SEL=A_explained_by_SEL,
         IO_explained_by_SEL=1 - IO_res_mag**2)
print("\nSaved exp63_results.npz")
