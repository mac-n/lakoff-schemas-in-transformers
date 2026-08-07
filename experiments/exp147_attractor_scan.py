"""
exp147_attractor_scan.py — is the late-layer VERTICAL attractor robust,
and what other axes are attractors?

In exp146 with seed 42, RAND-steering (α=4 at L12) produced
ΔVERTICAL = +0.530 at L20. Two possibilities:

A) Late-layer geometric attractor for VERTICAL specifically. Random
   perturbations propagate to align with VERTICAL_clean[L20] by late
   layers as a structural feature of Pythia's residual-stream geometry.
B) Coincidence with seed 42. Different random seeds would scatter
   around zero on VERTICAL.

Test: run N=20 different random seeds, all orthogonalized against
UP_clean[L12], steer at α=4, read multiple content axes at L20.

Axes scanned (does each show attractor behavior?):
- VERTICAL (above/below) — the exp146 finding
- MAGNITUDE (huge/tiny) — substrate-primitive from exp140
- VALENCE (good/bad) — substrate-attractor from exp145
- DIRECTIONAL (above/below) — same as VERTICAL actually
- HORIZONTAL (left/right)
- CONTAINMENT (inside/outside)
- TEMPORAL (before/after)

For each axis, compute mean Δ projection across N random seeds.
- If mean is consistently positive (> +0.3) with low variance,
  → axis is a robust late-layer attractor
- If mean is ~zero with high variance, → no specific attractor;
  any RAND result is noise.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
LAYERS = list(range(N_LAYERS))
resid_hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]


# ============================================================================
# Word lists
# ============================================================================

LAKOFF_UP   = ["up","rise","rose","rising","ascend","raise","climb","lift",
               "above","over","top","high","higher","upward"]
LAKOFF_DOWN = ["down","fall","fell","falling","descend","drop","sink",
               "below","under","bottom","low","lower","downward"]

VERTICAL_UP   = ["above","over","atop","upward","overhead","upper"]
VERTICAL_DOWN = ["below","under","underneath","downward","beneath","lower"]

MAGNITUDE_BIG   = ["huge","big","large","enormous","vast","massive","gigantic"]
MAGNITUDE_SMALL = ["tiny","small","little","minute","microscopic","miniature"]

VALENCE_POS = ["good","joy","love","beautiful","happy","kind","gentle","pleasant"]
VALENCE_NEG = ["bad","sorrow","hate","ugly","sad","cruel","harsh","unpleasant"]

HORIZONTAL_RIGHT = ["right","rightward","east","eastward"]
HORIZONTAL_LEFT  = ["left","leftward","west","westward"]

CONTAINMENT_IN  = ["inside","within","interior","internal","inner"]
CONTAINMENT_OUT = ["outside","exterior","external","outer","outwards"]

PROXIMITY_NEAR = ["near","close","nearby","adjacent","proximate"]
PROXIMITY_FAR  = ["far","distant","remote","faraway","afar"]

TEMPORAL_AFTER  = ["after","later","subsequent","following","next"]
TEMPORAL_BEFORE = ["before","earlier","previous","prior","preceding"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]

PROMPTS = [
    "The pool was",
    "She walked into the room and noticed the ceiling was",
    "His mood that morning was",
    "The mountain in the distance looked",
    "After climbing for hours they finally saw",
]


# ============================================================================
# Collect single-word residuals
# ============================================================================

all_words = set(COMMON + RARE)
for words in [LAKOFF_UP, LAKOFF_DOWN, VERTICAL_UP, VERTICAL_DOWN,
              MAGNITUDE_BIG, MAGNITUDE_SMALL, VALENCE_POS, VALENCE_NEG,
              HORIZONTAL_RIGHT, HORIZONTAL_LEFT,
              CONTAINMENT_IN, CONTAINMENT_OUT,
              PROXIMITY_NEAR, PROXIMITY_FAR,
              TEMPORAL_AFTER, TEMPORAL_BEFORE]:
    all_words.update(words)
all_words = sorted(all_words)
print(f"\nCollecting single-word residuals for {len(all_words)} words at all layers...")

word_residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=resid_hooks)
    word_residuals[w] = np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )

anisotropy_dirs = []
for L in LAYERS:
    all_r = np.stack([word_residuals[w][L] for w in all_words], axis=0)
    m = all_r.mean(axis=0)
    anisotropy_dirs.append(m / np.linalg.norm(m))


def mean_word_acts(words, layer):
    return np.mean([word_residuals[w][layer] for w in words], axis=0)


def strip_aniso_freq(direction, layer):
    freq_raw = mean_word_acts(COMMON, layer) - mean_word_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    aniso = anisotropy_dirs[layer]
    direction = direction - (direction @ aniso) * aniso
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)
    direction = direction - (direction @ freq_orth) * freq_orth
    return direction / np.linalg.norm(direction)


def build_axis_clean(pos, neg, layer):
    raw = mean_word_acts(pos, layer) - mean_word_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


print("\nBuilding readout axes at all layers...")
up_clean = [build_axis_clean(LAKOFF_UP, LAKOFF_DOWN, L) for L in LAYERS]
READOUT_AXES = {
    "VERTICAL":     [build_axis_clean(VERTICAL_UP, VERTICAL_DOWN, L) for L in LAYERS],
    "MAGNITUDE":    [build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L) for L in LAYERS],
    "VALENCE":      [build_axis_clean(VALENCE_POS, VALENCE_NEG, L) for L in LAYERS],
    "HORIZONTAL":   [build_axis_clean(HORIZONTAL_RIGHT, HORIZONTAL_LEFT, L) for L in LAYERS],
    "CONTAINMENT":  [build_axis_clean(CONTAINMENT_IN, CONTAINMENT_OUT, L) for L in LAYERS],
    "PROXIMITY":    [build_axis_clean(PROXIMITY_NEAR, PROXIMITY_FAR, L) for L in LAYERS],
    "TEMPORAL":     [build_axis_clean(TEMPORAL_AFTER, TEMPORAL_BEFORE, L) for L in LAYERS],
}


# ============================================================================
# Steering protocol
# ============================================================================

L_INJECT = 12
ALPHA = 4.0
N_SEEDS = 20

D = up_clean[L_INJECT].shape[0]


def random_dir_for_seed(seed):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(D)
    # Orthogonalise against UP_clean at L_INJECT
    raw = raw - (raw @ up_clean[L_INJECT]) * up_clean[L_INJECT]
    return raw / np.linalg.norm(raw)


def run_with_steering(prompt, steer_direction, alpha):
    toks = model.to_tokens(prompt)
    if alpha == 0.0:
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=resid_hooks)
    else:
        steer_tensor = torch.tensor(steer_direction, dtype=torch.float32,
                                    device=device) * alpha
        hook_name = f"blocks.{L_INJECT}.hook_resid_post"
        def hook_fn(activation, hook):
            activation[:, -1, :] = activation[:, -1, :] + steer_tensor
            return activation
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                _, cache = model.run_with_cache(toks, names_filter=resid_hooks)
    return np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )


# ============================================================================
# Main experiment
# ============================================================================

print(f"\nComputing baselines for {len(PROMPTS)} prompts...")
baseline_resids = []
for prompt in PROMPTS:
    baseline = run_with_steering(prompt, up_clean[L_INJECT], 0.0)
    baseline_resids.append(baseline)

# shifts[seed][axis][layer] = mean Δ proj across prompts
print(f"\nRunning {N_SEEDS} random-direction steerings...")
all_shifts = np.zeros((N_SEEDS, len(READOUT_AXES), N_LAYERS))  # [seed, axis, layer]
axis_names = list(READOUT_AXES.keys())

for s_idx, seed in enumerate(range(N_SEEDS)):
    rd = random_dir_for_seed(seed)
    for p_idx, prompt in enumerate(PROMPTS):
        steered = run_with_steering(prompt, rd, ALPHA)
        diff = steered - baseline_resids[p_idx]
        for a_idx, a_name in enumerate(axis_names):
            for L in LAYERS:
                all_shifts[s_idx, a_idx, L] += float(diff[L] @ READOUT_AXES[a_name][L])
    all_shifts[s_idx] /= len(PROMPTS)
    if (s_idx + 1) % 5 == 0:
        print(f"  {s_idx+1}/{N_SEEDS} seeds done")


# ============================================================================
# Aggregate — per-axis distribution across seeds at L20 (key)
# ============================================================================

print("\n" + "=" * 78)
print(f"Distribution across {N_SEEDS} random seeds, at L=20")
print("=" * 78)
print(f"\n  {'axis':<14}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  "
      f"{'attractor?':>12}")
L_KEY = 20
for a_idx, a_name in enumerate(axis_names):
    vals = all_shifts[:, a_idx, L_KEY]
    m = vals.mean(); s = vals.std(); mn = vals.min(); mx = vals.max()
    is_attractor = "YES" if m > 0.3 and m / max(s, 1e-6) > 1.5 else \
                   "(weak)" if m > 0.15 and m / max(s, 1e-6) > 1.0 else "no"
    print(f"  {a_name:<14}  {m:>+8.3f}  {s:>8.3f}  {mn:>+8.3f}  {mx:>+8.3f}  "
          f"{is_attractor:>12}")

# Also print at L=16 and L=23 for comparison
for L_KEY in [16, 23]:
    print(f"\n  --- At L={L_KEY} ---")
    print(f"  {'axis':<14}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    for a_idx, a_name in enumerate(axis_names):
        vals = all_shifts[:, a_idx, L_KEY]
        print(f"  {a_name:<14}  {vals.mean():>+8.3f}  {vals.std():>8.3f}  "
              f"{vals.min():>+8.3f}  {vals.max():>+8.3f}")


# ============================================================================
# Per-layer trajectory of mean Δ across seeds, per axis
# ============================================================================

print("\n" + "=" * 78)
print("Per-layer mean Δ across seeds — which axes attract late?")
print("=" * 78)
print(f"\n  {'layer':<6}  " + "  ".join(f"{a[:9]:>9}" for a in axis_names))
for L in [4, 8, 12, 14, 16, 18, 20, 22, 23]:
    row = f"  L{L:<5}  "
    for a_idx, a_name in enumerate(axis_names):
        row += f"{all_shifts[:, a_idx, L].mean():>+9.3f}  "
    print(row)


# ============================================================================
# Plot — mean Δ per layer per axis across seeds, with ±1 std band
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))
xs = np.arange(N_LAYERS)
colors = {"VERTICAL": "tab:red", "MAGNITUDE": "tab:orange", "VALENCE": "tab:green",
          "HORIZONTAL": "tab:blue", "CONTAINMENT": "tab:purple",
          "PROXIMITY": "tab:brown", "TEMPORAL": "tab:cyan"}

for a_idx, a_name in enumerate(axis_names):
    means = all_shifts[:, a_idx, :].mean(axis=0)
    stds  = all_shifts[:, a_idx, :].std(axis=0)
    color = colors.get(a_name, "black")
    ax.plot(xs, means, "-o", label=a_name, color=color, markersize=4)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)

ax.axvline(L_INJECT, color="red", linestyle=":", alpha=0.5, label="L_inject")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("readout layer")
ax.set_ylabel(f"mean Δ projection (across {N_SEEDS} random seeds, ±1 std)")
ax.set_title(f"exp147 — late-layer attractor scan\n"
             f"Random-direction steering, α={ALPHA} at L{L_INJECT}, {N_SEEDS} seeds")
ax.legend(fontsize=9, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp147_attractor_scan.png", dpi=120)
print("\nSaved exp147_attractor_scan.png")


# ============================================================================
# Verdict
# ============================================================================

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)

L_KEY = 20
print(f"\nAt L={L_KEY}, mean ± std across {N_SEEDS} seeds:")
for a_idx, a_name in enumerate(axis_names):
    vals = all_shifts[:, a_idx, L_KEY]
    m = vals.mean(); s = vals.std()
    sign_consistent = (vals > 0).sum() if m > 0 else (vals < 0).sum()
    print(f"  {a_name:<14}  {m:>+7.3f} ± {s:>5.3f}  "
          f"sign-consistent {sign_consistent}/{N_SEEDS}")

print("""
INTERPRETATION:
- An axis is a robust late-layer attractor if mean > 0.3 with low std
  (mean/std > 1.5) AND sign-consistent across most seeds.
- VERTICAL was the exp146 surprise (+0.530 with seed 42).
  If mean across seeds at L20 is in the +0.3 to +0.6 range with low
  variance, the late-VERTICAL-attractor is confirmed robust.
- VALENCE was the exp145 finding (NOUN_VERB/CONC_ABS routed there
  above noise). Whether VAL is also an attractor here clarifies if
  the late-VAL-routing claim is substrate or specific.
- Other axes (HORIZONTAL, CONTAINMENT, PROXIMITY, TEMPORAL) provide
  contrast — if no/few of these are attractors, the finding is
  specific to certain axes.
""")

print("Done.")
