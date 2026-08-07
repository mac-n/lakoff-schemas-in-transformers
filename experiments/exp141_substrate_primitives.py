"""
exp141_substrate_primitives.py — four tests of substrate-primitive
hypotheses raised after exp140.

TEST 0 — Is VALENCE riding the anisotropy direction?
    Build raw VALENCE axis (no stripping). Compute cos(VAL_raw, aniso)
    per layer. Compare to raw MAGNITUDE, raw DIRECTIONAL, and Lakoff
    schemas. exp123 found Lakoff schemas have |cos with aniso| < 0.03.
    exp137 found morphology pair-diff has ~0.55-0.59. Where does
    valence land? If high → valence IS partly anisotropy.

TEST 1 — Does LAKOFF_UP shift from DIRECTIONAL to MAGNITUDE across layers?
    Plot cos(LAKOFF_clean[L], {MAG_clean[L], DIR_clean[L], VAL_clean[L]})
    across all 24 layers. Prediction: DIR-alignment dominates early
    layers, MAG (and maybe VAL) take over late.

TEST 2 — Is BALANCE a norm-deviation primitive?
    For each suffix pair (base, inflected), compute ||residual(base)||
    and ||residual(inflected)|| at each layer. Check (a) whether
    inflected forms systematically differ in norm and (b) whether the
    norm-deviation correlates with the pair's projection on BALANCE.
    If yes, BALANCE captures what LayerNorm's variance-normalisation
    pushes against.

TEST 3 — Does VALENCE behave like a substrate-primitive?
    Cross-layer stability of clean VAL axis. Compare to MAG (+0.79),
    DIR (+0.74), LAKOFF (+0.74 per exp140). If VAL is in the same
    range or higher, it's playing a substrate-level role. If notably
    lower, it's a learned content direction.

TEST 4 — Does LAKOFF_UP decompose into {MAG, DIR, VAL} across layers?
    Per layer: orthonormalise MAG, DIR, VAL via Gram-Schmidt. Project
    LAKOFF_UP_clean onto each. Plot coefficients across L. Prediction
    (embodied-metacognition synthesis): DIR coefficient dominates
    early, MAG and VAL grow late.
    CONTROL: same decomposition with 3 random orthonormal axes built
    from token activations — to confirm trajectory isn't an artifact
    of having 3 free directions.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
LAYERS = list(range(N_LAYERS))
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]


# ============================================================================
# Word lists — curated for purity
# ============================================================================

MAGNITUDE_BIG   = ["huge","big","large","enormous","vast","massive","gigantic"]
MAGNITUDE_SMALL = ["tiny","small","little","minute","microscopic","miniature"]

DIRECTIONAL_UP   = ["above","over","atop","upward","overhead","upper"]
DIRECTIONAL_DOWN = ["below","under","underneath","downward","beneath","lower"]

# Valence — DELIBERATELY excludes magnitude-loaded words (great, huge),
# directional-loaded words (high, low), and light-loaded words (bright, dark).
VALENCE_POS = ["good","joy","love","beautiful","happy","kind","gentle","pleasant"]
VALENCE_NEG = ["bad","sorrow","hate","ugly","sad","cruel","harsh","unpleasant"]

LAKOFF_UP   = ["up","rise","rose","rising","ascend","raise","climb","lift",
               "above","over","top","high","higher","upward"]
LAKOFF_DOWN = ["down","fall","fell","falling","descend","drop","sink",
               "below","under","bottom","low","lower","downward"]

# Suffix pairs for BALANCE-as-norm test (subset of exp138 list)
SUFFIX_PAIRS = {
    "ER_comparative": [("big","bigger"),("small","smaller"),("tall","taller"),
        ("high","higher"),("low","lower"),("deep","deeper"),("wide","wider"),
        ("fast","faster"),("slow","slower"),("old","older"),("new","newer"),
        ("hot","hotter"),("cold","colder")],
    "EST_superlative": [("big","biggest"),("small","smallest"),("tall","tallest"),
        ("high","highest"),("low","lowest"),("deep","deepest"),("old","oldest"),
        ("hot","hottest"),("cold","coldest")],
    "ING_progressive": [("walk","walking"),("run","running"),("jump","jumping"),
        ("sit","sitting"),("stand","standing"),("swim","swimming"),
        ("talk","talking"),("sing","singing"),("dance","dancing"),
        ("play","playing"),("work","working"),("read","reading")],
    "ED_past": [("walk","walked"),("jump","jumped"),("look","looked"),
        ("talk","talked"),("play","played"),("work","worked"),("ask","asked"),
        ("call","called"),("move","moved"),("stop","stopped"),("start","started")],
    "UN_negation": [("happy","unhappy"),("kind","unkind"),("healthy","unhealthy"),
        ("safe","unsafe"),("clear","unclear"),("clean","unclean"),("fair","unfair"),
        ("certain","uncertain"),("known","unknown"),("seen","unseen")],
    "RE_repetition": [("do","redo"),("make","remake"),("build","rebuild"),
        ("write","rewrite"),("read","reread"),("start","restart"),
        ("create","recreate"),("paint","repaint")],
}

SCHEMA_NAMES = ["UP-DOWN","IN-OUT_CLEAN","FORWARD-BACK","PATH-MOTION",
                "LIGHT-DARK","FORCE","BALANCE","DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# ============================================================================
# Collect residuals
# ============================================================================

all_words = set(COMMON + RARE)
for words in [MAGNITUDE_BIG, MAGNITUDE_SMALL, DIRECTIONAL_UP, DIRECTIONAL_DOWN,
              VALENCE_POS, VALENCE_NEG, LAKOFF_UP, LAKOFF_DOWN]:
    all_words.update(words)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)
print(f"\nCollecting residuals for {len(all_words)} words at all {N_LAYERS} layers...")

residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )
    if (k+1) % 50 == 0:
        print(f"  {k+1}/{len(all_words)}")


# Anisotropy direction per layer
anisotropy_dirs = []
for L in LAYERS:
    all_r = np.stack([residuals[w][L] for w in all_words], axis=0)
    m = all_r.mean(axis=0)
    anisotropy_dirs.append(m / np.linalg.norm(m))


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def build_axis_raw(pos, neg, layer):
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    return raw / np.linalg.norm(raw)


def strip_aniso_freq(direction, layer):
    freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    aniso = anisotropy_dirs[layer]
    direction = direction - (direction @ aniso) * aniso
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)
    direction = direction - (direction @ freq_orth) * freq_orth
    return direction / np.linalg.norm(direction)


def build_axis_clean(pos, neg, layer):
    return strip_aniso_freq(build_axis_raw(pos, neg, layer), layer)


def build_schema_raw(name, layer):
    pairs = LAKOFF_SCHEMAS_MML[name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    return build_axis_raw(pos, neg, layer)


def build_schema_clean(name, layer):
    return strip_aniso_freq(build_schema_raw(name, layer), layer)


# ============================================================================
# TEST 0 — does VALENCE ride the anisotropy direction?
# ============================================================================

print("\n" + "=" * 78)
print("TEST 0 — |cos with anisotropy| per layer (raw axes, NO STRIPPING)")
print("=" * 78)
print(f"\n  layer    MAG_raw    DIR_raw    VAL_raw    LAKOFF_raw   BALANCE_raw")
for L in LAYERS:
    if L not in [0, 4, 8, 12, 16, 20, 23]:
        continue
    aniso = anisotropy_dirs[L]
    m = build_axis_raw(MAGNITUDE_BIG, MAGNITUDE_SMALL, L) @ aniso
    d = build_axis_raw(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L) @ aniso
    v = build_axis_raw(VALENCE_POS, VALENCE_NEG, L) @ aniso
    l = build_axis_raw(LAKOFF_UP, LAKOFF_DOWN, L) @ aniso
    b = build_schema_raw("BALANCE", L) @ aniso
    print(f"  L{L:>3}   {float(m):>+8.3f}   {float(d):>+8.3f}   {float(v):>+8.3f}   "
          f"{float(l):>+8.3f}      {float(b):>+8.3f}")

print("\n  Interpretation:")
print("  - exp123 baseline: Lakoff schemas have |cos with aniso| < 0.03 (clean)")
print("  - exp137 baseline: morphology pair-diffs have ~0.55-0.59 (contaminated)")
print("  - if VAL_raw is in the morphology range or higher, valence rides anisotropy")
print("  - if VAL_raw is in the Lakoff range (<0.1), valence is orthogonal to anisotropy")


# ============================================================================
# Build all clean axes for all layers (used by tests 1, 3, 4)
# ============================================================================

mag_clean   = [build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L) for L in LAYERS]
dir_clean   = [build_axis_clean(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L) for L in LAYERS]
val_clean   = [build_axis_clean(VALENCE_POS, VALENCE_NEG, L) for L in LAYERS]
lakoff_clean = [build_axis_clean(LAKOFF_UP, LAKOFF_DOWN, L) for L in LAYERS]
balance_clean = [build_schema_clean("BALANCE", L) for L in LAYERS]


# ============================================================================
# TEST 1 — LAKOFF_UP layer trajectory: cos with MAG vs DIR vs VAL
# ============================================================================

print("\n" + "=" * 78)
print("TEST 1 — cos(LAKOFF_clean[L], MAG/DIR/VAL_clean[L]) across all layers")
print("=" * 78)
print(f"\n  layer    LAK·MAG    LAK·DIR    LAK·VAL    winner")
cos_lm = np.zeros(N_LAYERS)
cos_ld = np.zeros(N_LAYERS)
cos_lv = np.zeros(N_LAYERS)
for L in LAYERS:
    cos_lm[L] = float(lakoff_clean[L] @ mag_clean[L])
    cos_ld[L] = float(lakoff_clean[L] @ dir_clean[L])
    cos_lv[L] = float(lakoff_clean[L] @ val_clean[L])
    abs_vals = {"MAG": abs(cos_lm[L]), "DIR": abs(cos_ld[L]), "VAL": abs(cos_lv[L])}
    winner = max(abs_vals, key=abs_vals.get)
    if L % 2 == 0 or L == N_LAYERS - 1:
        print(f"  L{L:>3}   {cos_lm[L]:>+8.3f}   {cos_ld[L]:>+8.3f}   "
              f"{cos_lv[L]:>+8.3f}    {winner}")


# ============================================================================
# TEST 2 — BALANCE as norm-deviation primitive
# ============================================================================

print("\n" + "=" * 78)
print("TEST 2 — BALANCE as norm-deviation primitive")
print("=" * 78)
print("\n2a. Norm difference per suffix family (||inflected|| - ||base||)")
print(f"\n  layer    {'suffix':<18}  mean Δ||r||     SE")
LAYERS_BAL = [4, 8, 12, 16, 20]
for L in LAYERS_BAL:
    print(f"\n  L{L}:")
    for sn, pairs in SUFFIX_PAIRS.items():
        deltas = []
        for b, i in pairs:
            nb = np.linalg.norm(residuals[b][L])
            ni = np.linalg.norm(residuals[i][L])
            deltas.append(ni - nb)
        m = np.mean(deltas); se = np.std(deltas, ddof=1) / np.sqrt(len(deltas))
        print(f"         {sn:<18}  {m:>+10.3f}    {se:>+6.3f}")

print("\n2b. Correlation: per-pair norm-delta vs per-pair projection on BALANCE")
print(f"\n  layer    suffix family            r(Δ||r||, BALANCE proj)")
for L in LAYERS_BAL:
    bal = balance_clean[L]
    for sn, pairs in SUFFIX_PAIRS.items():
        deltas = []
        bal_projs = []
        for b, i in pairs:
            rb = residuals[b][L]; ri = residuals[i][L]
            deltas.append(np.linalg.norm(ri) - np.linalg.norm(rb))
            diff = (ri / np.linalg.norm(ri)) - (rb / np.linalg.norm(rb))
            bal_projs.append(float(diff @ bal))
        if len(deltas) >= 4 and np.std(deltas) > 1e-8 and np.std(bal_projs) > 1e-8:
            r = np.corrcoef(deltas, bal_projs)[0, 1]
        else:
            r = float("nan")
        print(f"  L{L:>3}    {sn:<22}    {r:>+8.3f}")

print("\n  Interpretation:")
print("  - If Δ||r|| is systematically nonzero AND correlates with BALANCE")
print("    projection per pair, BALANCE encodes 'how far from typical norm' —")
print("    consistent with the LayerNorm-variance-normalisation hypothesis.")


# ============================================================================
# TEST 3 — VALENCE cross-layer stability vs other axes
# ============================================================================

print("\n" + "=" * 78)
print("TEST 3 — cross-layer stability (mean off-diag cos, L4-L22)")
print("=" * 78)

WORK = list(range(4, 23))
def cross_layer_mean(dirs, work):
    arr = np.array([dirs[L] for L in work])
    cm = arr @ arr.T
    return cm[~np.eye(len(work), dtype=bool)].mean()

mag_stab = cross_layer_mean(mag_clean, WORK)
dir_stab = cross_layer_mean(dir_clean, WORK)
val_stab = cross_layer_mean(val_clean, WORK)
lak_stab = cross_layer_mean(lakoff_clean, WORK)
bal_stab = cross_layer_mean(balance_clean, WORK)

print(f"\n  MAGNITUDE    {mag_stab:+.4f}")
print(f"  DIRECTIONAL  {dir_stab:+.4f}")
print(f"  VALENCE      {val_stab:+.4f}")
print(f"  LAKOFF UP    {lak_stab:+.4f}")
print(f"  BALANCE      {bal_stab:+.4f}")
print("\n  Interpretation:")
print("  - exp140 baseline: MAG=+0.79, DIR=+0.74")
print("  - if VAL >= MAG: substrate-primitive case")
print("  - if VAL ~ LAKOFF: learned-content case")
print("  - if BAL very high: supports BALANCE as substrate-rooted (LayerNorm?)")


# ============================================================================
# TEST 4 — Gram-Schmidt decomposition of LAKOFF_UP onto {MAG, DIR, VAL}
# ============================================================================

def gram_schmidt(*axes):
    """Orthonormalise a sequence of unit vectors."""
    out = []
    for a in axes:
        v = a.copy()
        for u in out:
            v = v - (v @ u) * u
        n = np.linalg.norm(v)
        if n > 1e-8:
            out.append(v / n)
        else:
            out.append(np.zeros_like(v))
    return out


print("\n" + "=" * 78)
print("TEST 4 — Gram-Schmidt decomposition of LAKOFF_UP onto {MAG, DIR, VAL}")
print("=" * 78)

# First report axis-axis cosines so we know how orthogonal the basis is
print("\n4a. Axis-axis cosines (after stripping) — how non-orthogonal are MAG/DIR/VAL?")
print(f"\n  layer    cos(MAG,DIR)    cos(MAG,VAL)    cos(DIR,VAL)")
for L in LAYERS:
    if L not in [0, 4, 8, 12, 16, 20, 23]:
        continue
    cmd = float(mag_clean[L] @ dir_clean[L])
    cmv = float(mag_clean[L] @ val_clean[L])
    cdv = float(dir_clean[L] @ val_clean[L])
    print(f"  L{L:>3}      {cmd:>+8.3f}        {cmv:>+8.3f}        {cdv:>+8.3f}")
print("\n  → If these are near 0, ordering doesn't matter (basis effectively orthogonal).")
print("  → If non-trivial, the first-stripped axis 'claims' the shared variance.")

# Run decomposition under three orderings
ORDERINGS = [
    ("MAG_first", ["MAG", "DIR", "VAL"]),
    ("DIR_first", ["DIR", "MAG", "VAL"]),
    ("VAL_first", ["VAL", "MAG", "DIR"]),
]
axis_lookup = {"MAG": mag_clean, "DIR": dir_clean, "VAL": val_clean}

decomp_all = {}  # name -> (N_LAYERS, 4) array of [coefMAG, coefDIR, coefVAL, resid]
for name, order in ORDERINGS:
    print(f"\n4b. Decomposition under ordering {order} — Gram-Schmidt strips left-to-right")
    print(f"\n  layer    coef_MAG    coef_DIR    coef_VAL    resid")
    arr = np.zeros((N_LAYERS, 4))
    for L in LAYERS:
        # Build orthonormal basis in the requested order
        ax_list = [axis_lookup[k][L] for k in order]
        u_first, u_second, u_third = gram_schmidt(*ax_list)
        u_by_name = dict(zip(order, [u_first, u_second, u_third]))
        lk = lakoff_clean[L]
        cM = float(lk @ u_by_name["MAG"])
        cD = float(lk @ u_by_name["DIR"])
        cV = float(lk @ u_by_name["VAL"])
        resid = lk - cM * u_by_name["MAG"] - cD * u_by_name["DIR"] - cV * u_by_name["VAL"]
        arr[L] = [cM, cD, cV, np.linalg.norm(resid)]
        if L % 4 == 0 or L == N_LAYERS - 1:
            print(f"  L{L:>3}   {cM:>+8.3f}   {cD:>+8.3f}   {cV:>+8.3f}   {arr[L,3]:>+6.3f}")
    decomp_all[name] = arr

# Keep the MAG-first decomp as the canonical one for downstream uses
decomp = decomp_all["MAG_first"]


# ============================================================================
# TEST 4b — random-axis control
# ============================================================================

print("\n" + "=" * 78)
print("TEST 4b — random-axis control")
print("=" * 78)
print("  Pick 3 random unit directions in residual space, orthonormalise,")
print("  decompose LAKOFF_UP. Repeat 50 times. If real coefficients sit")
print("  inside the null band, the structure isn't specific to {MAG,DIR,VAL}.")

D = lakoff_clean[0].shape[0]
N_NULL = 50
null_coefs = np.zeros((N_LAYERS, N_NULL, 3))  # 3 abs coefficients
rng = np.random.default_rng(42)
for L in LAYERS:
    for n in range(N_NULL):
        # 3 random gaussian directions, orthonormalise
        raw = [rng.standard_normal(D) for _ in range(3)]
        raw = [r / np.linalg.norm(r) for r in raw]
        u1, u2, u3 = gram_schmidt(*raw)
        lk = lakoff_clean[L]
        null_coefs[L, n] = [abs(float(lk @ u1)),
                            abs(float(lk @ u2)),
                            abs(float(lk @ u3))]

print(f"\n  layer    real |MAG|  null mean  real |DIR|  null mean  real |VAL|  null mean")
for L in LAYERS:
    if L not in [0, 4, 8, 12, 16, 20, 23]:
        continue
    real_m = abs(decomp[L, 0]); null_m = null_coefs[L].mean(axis=0)[0]
    real_d = abs(decomp[L, 1]); null_d = null_coefs[L].mean(axis=0)[1]
    real_v = abs(decomp[L, 2]); null_v = null_coefs[L].mean(axis=0)[2]
    print(f"  L{L:>3}   {real_m:>+8.3f}   {null_m:>+8.3f}    "
          f"{real_d:>+8.3f}   {null_d:>+8.3f}    "
          f"{real_v:>+8.3f}   {null_v:>+8.3f}")


# ============================================================================
# Plots
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
xs = np.arange(N_LAYERS)

# Plot 1: TEST 1 — LAKOFF_UP cos trajectory
ax = axes[0, 0]
ax.plot(xs, cos_lm, "-o", label="cos(LAKOFF, MAG)", color="tab:orange")
ax.plot(xs, cos_ld, "-s", label="cos(LAKOFF, DIR)", color="tab:blue")
ax.plot(xs, cos_lv, "-^", label="cos(LAKOFF, VAL)", color="tab:green")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("cosine")
ax.set_title("TEST 1 — LAKOFF_UP alignment with substrate axes across layers")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

# Plots 2-4: TEST 4 — decomposition coefficients under each ordering
ordering_axes = {"MAG_first": axes[0, 1], "DIR_first": axes[1, 0], "VAL_first": axes[1, 1]}
null_mean_arr = null_coefs.mean(axis=1)
null_std_arr  = null_coefs.std(axis=1)
for name, ax in ordering_axes.items():
    arr = decomp_all[name]
    ax.plot(xs, arr[:, 0], "-o", label="MAG coef", color="tab:orange")
    ax.plot(xs, arr[:, 1], "-s", label="DIR coef", color="tab:blue")
    ax.plot(xs, arr[:, 2], "-^", label="VAL coef", color="tab:green")
    ax.plot(xs, arr[:, 3], "--", label="residual ||...||", color="gray")
    for k, color in zip([0, 1, 2], ["tab:orange", "tab:blue", "tab:green"]):
        ax.fill_between(xs, -null_mean_arr[:, k] - null_std_arr[:, k],
                        null_mean_arr[:, k] + null_std_arr[:, k],
                        color=color, alpha=0.08)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("coefficient")
    ax.set_title(f"TEST 4 ({name}) — Gram-Schmidt order: " +
                 " → ".join({"MAG_first":["MAG","DIR","VAL"],
                             "DIR_first":["DIR","MAG","VAL"],
                             "VAL_first":["VAL","MAG","DIR"]}[name]))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp141_substrate_primitives.png", dpi=120)
print("\nSaved exp141_substrate_primitives.png")


# ============================================================================
# Summary block
# ============================================================================

print("\n" + "=" * 78)
print("SUMMARY — substrate-primitive hypotheses")
print("=" * 78)
print(f"\n  TEST 0  (valence-as-anisotropy):")
val_aniso_L12 = float(build_axis_raw(VALENCE_POS, VALENCE_NEG, 12) @ anisotropy_dirs[12])
mag_aniso_L12 = float(build_axis_raw(MAGNITUDE_BIG, MAGNITUDE_SMALL, 12) @ anisotropy_dirs[12])
print(f"    VAL_raw·aniso @ L12 = {val_aniso_L12:+.3f}")
print(f"    MAG_raw·aniso @ L12 = {mag_aniso_L12:+.3f}")
print(f"    → valence rides anisotropy IF |VAL_raw·aniso| > 0.3 at any layer")

print(f"\n  TEST 1  (LAKOFF shift across layers):")
print(f"    L0  : LAK·DIR = {cos_ld[0]:+.3f}, LAK·MAG = {cos_lm[0]:+.3f}")
print(f"    L12 : LAK·DIR = {cos_ld[12]:+.3f}, LAK·MAG = {cos_lm[12]:+.3f}")
print(f"    L23 : LAK·DIR = {cos_ld[23]:+.3f}, LAK·MAG = {cos_lm[23]:+.3f}")

print(f"\n  TEST 3  (cross-layer stability):")
print(f"    MAG={mag_stab:+.3f}  DIR={dir_stab:+.3f}  VAL={val_stab:+.3f}  "
      f"LAK={lak_stab:+.3f}  BAL={bal_stab:+.3f}")

print(f"\n  TEST 4  (decomposition L12):")
print(f"    MAG coef = {decomp[12, 0]:+.3f}, DIR coef = {decomp[12, 1]:+.3f}, "
      f"VAL coef = {decomp[12, 2]:+.3f}, residual = {decomp[12, 3]:+.3f}")
print(f"    null mean abs coef @ L12 = {null_coefs[12].mean():+.3f}")

print("\nDone.")
