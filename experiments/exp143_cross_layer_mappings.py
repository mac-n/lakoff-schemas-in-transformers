"""
exp143_cross_layer_mappings.py — does UP-steering at L_inject propagate to
MAG/VAL/DIR projections at downstream layers?

Background:
- exp116 showed clean-UP steering causes magnitude effects in output
  (pool depth shifts up with patient height under steering).
- exp141 found cos(UP_clean[L], MAG_clean[L]) ~ 0 at every layer.
- Apparent contradiction: how can UP-steering increase magnitude effects
  if UP and MAG don't share geometric structure within-layer?

Resolution (hypothesis): the cos-at-single-layer test measures within-layer
co-activation. Lakoff's metaphorical mappings are CROSS-LAYER transformations
implemented by attention/MLP weights. Steering on UP at L_inject can produce
MAG content at L_inject + k via the learned UP→MORE mapping, even though
UP_clean[L] and MAG_clean[L] are orthogonal at every individual L.

Method:
- Build clean UP, MAG, VAL, DIR axes at all layers (anisotropy + freq stripped).
- For each prompt, run forward pass and record per-layer last-token residuals
  BASELINE (no steering).
- Repeat with steering: add α · UP_clean[L_inject] at the L_inject residual
  hook. Record per-layer last-token residuals POST-STEERING.
- Compute projection of (post − baseline) residual at downstream layers onto
  MAG_clean[L_readout], VAL_clean[L_readout], DIR_clean[L_readout].
- Repeat with α in {0, 1, 2, 4, 8} for dose-response.
- Control: same protocol with a RANDOM unit direction at L_inject (same norm
  as α · UP_clean). Should produce no systematic shift on MAG/VAL/DIR.

If UP-steering propagates to MAG projection at L_readout > L_inject, the
model has the UP→MORE mapping as a cross-layer transformation. Same logic
for VAL (UP→GOOD) and DIR (UP→UP-as-direction).

Predicted result:
- MAG projection at L_readout > L_inject increases with α (UP→MORE mapping)
- VAL projection at L_readout > L_inject increases with α (UP→GOOD mapping)
- DIR projection mostly preserved (within-layer alignment already)
- Random-direction control: flat
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
# Word lists — same as exp141 for consistency
# ============================================================================

MAGNITUDE_BIG   = ["huge","big","large","enormous","vast","massive","gigantic"]
MAGNITUDE_SMALL = ["tiny","small","little","minute","microscopic","miniature"]

DIRECTIONAL_UP   = ["above","over","atop","upward","overhead","upper"]
DIRECTIONAL_DOWN = ["below","under","underneath","downward","beneath","lower"]

VALENCE_POS = ["good","joy","love","beautiful","happy","kind","gentle","pleasant"]
VALENCE_NEG = ["bad","sorrow","hate","ugly","sad","cruel","harsh","unpleasant"]

LAKOFF_UP   = ["up","rise","rose","rising","ascend","raise","climb","lift",
               "above","over","top","high","higher","upward"]
LAKOFF_DOWN = ["down","fall","fell","falling","descend","drop","sink",
               "below","under","bottom","low","lower","downward"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# ============================================================================
# Prompts — natural English, last-token position is where we read residuals
# ============================================================================

PROMPTS = [
    "The pool was",
    "She walked into the room and noticed the ceiling was",
    "His mood that morning was",
    "The mountain in the distance looked",
    "After climbing for hours they finally saw",
    "The well in the garden seemed surprisingly",
    "When the elevator finally stopped he stepped out feeling",
    "The library shelves reached up to a height that was",
    "The child looked at the cake and said it was",
    "Standing at the edge they realised the drop was",
]

print(f"Using {len(PROMPTS)} prompts for steering test.")


# ============================================================================
# Collect single-word residuals (for building axes)
# ============================================================================

all_words = set(COMMON + RARE)
for words in [MAGNITUDE_BIG, MAGNITUDE_SMALL, DIRECTIONAL_UP, DIRECTIONAL_DOWN,
              VALENCE_POS, VALENCE_NEG, LAKOFF_UP, LAKOFF_DOWN]:
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
    if (k+1) % 30 == 0:
        print(f"  {k+1}/{len(all_words)}")


# Anisotropy from single-word residuals
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


# Build clean axes at all layers
print("\nBuilding clean axes at all layers...")
up_clean   = [build_axis_clean(LAKOFF_UP, LAKOFF_DOWN, L) for L in LAYERS]
mag_clean  = [build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L) for L in LAYERS]
val_clean  = [build_axis_clean(VALENCE_POS, VALENCE_NEG, L) for L in LAYERS]
dir_clean  = [build_axis_clean(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L) for L in LAYERS]

# Sanity check: confirm exp141's finding that cos(UP, MAG) ~ 0 at L12
print(f"\nSanity check at L12 (should be near zero per exp141):")
print(f"  cos(UP_clean[12], MAG_clean[12]) = {float(up_clean[12] @ mag_clean[12]):+.4f}")
print(f"  cos(UP_clean[12], VAL_clean[12]) = {float(up_clean[12] @ val_clean[12]):+.4f}")
print(f"  cos(UP_clean[12], DIR_clean[12]) = {float(up_clean[12] @ dir_clean[12]):+.4f}")


# ============================================================================
# Steering protocol
# ============================================================================

L_INJECT_LIST = [8, 12, 16]   # try multiple injection layers
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]

# We need a random control direction PER injection layer (orthogonal to UP)
rng = np.random.default_rng(42)
random_dirs = {}
for L_inj in L_INJECT_LIST:
    raw = rng.standard_normal(up_clean[L_inj].shape[0])
    # Orthogonalise against UP_clean at that layer
    raw = raw - (raw @ up_clean[L_inj]) * up_clean[L_inj]
    random_dirs[L_inj] = raw / np.linalg.norm(raw)


def run_with_steering(prompt, L_inject, steer_direction, alpha):
    """Run forward pass with α·steer_direction added at L_inject hook.
    Returns last-token residuals at all layers."""
    toks = model.to_tokens(prompt)
    if alpha == 0.0:
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=resid_hooks)
    else:
        steer_tensor = torch.tensor(steer_direction, dtype=torch.float32,
                                    device=device) * alpha
        hook_name = f"blocks.{L_inject}.hook_resid_post"
        def hook_fn(activation, hook):
            activation[:, -1, :] = activation[:, -1, :] + steer_tensor
            return activation
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                _, cache = model.run_with_cache(toks, names_filter=resid_hooks)
    return np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy() for L in LAYERS],
        axis=0
    )  # (N_LAYERS, D)


# ============================================================================
# Main experiment — for each L_inject, alpha, direction (UP vs RAND),
# record per-layer residuals across all prompts and compute projection
# changes on MAG / VAL / DIR axes.
# ============================================================================

print("\n" + "=" * 78)
print("MAIN EXPERIMENT — UP-steering propagation to MAG/VAL/DIR downstream")
print("=" * 78)


# Storage: results[direction_name][L_inject][alpha] -> mean projection shift per readout layer
# shift = projection_under_steering - projection_baseline, averaged over prompts.
results = {"UP": {}, "RAND": {}}

# Also store raw projections to inspect baselines
baselines = {}  # baselines[L_inject][prompt_idx] = (N_LAYERS, 3) for [MAG, VAL, DIR]

for L_inject in L_INJECT_LIST:
    print(f"\n--- L_inject = {L_inject} ---")
    results["UP"][L_inject] = {}
    results["RAND"][L_inject] = {}

    # Compute baselines (alpha=0) per prompt
    baseline_resids = []  # list of (N_LAYERS, D) per prompt
    for p_idx, prompt in enumerate(PROMPTS):
        baseline = run_with_steering(prompt, L_inject, up_clean[L_inject], 0.0)
        baseline_resids.append(baseline)
    print(f"  baselines computed for {len(PROMPTS)} prompts.")

    # For each direction (UP vs RAND) and alpha, compute projection shifts
    for dir_name, dir_per_layer in [("UP", up_clean), ("RAND", None)]:
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue  # baseline already covered
            shifts = np.zeros((N_LAYERS, 3))  # [MAG, VAL, DIR]
            for p_idx, prompt in enumerate(PROMPTS):
                if dir_name == "UP":
                    steer_dir = up_clean[L_inject]
                else:
                    steer_dir = random_dirs[L_inject]
                steered = run_with_steering(prompt, L_inject, steer_dir, alpha)
                diff = steered - baseline_resids[p_idx]  # (N_LAYERS, D)
                for L in LAYERS:
                    shifts[L, 0] += float(diff[L] @ mag_clean[L])
                    shifts[L, 1] += float(diff[L] @ val_clean[L])
                    shifts[L, 2] += float(diff[L] @ dir_clean[L])
            shifts /= len(PROMPTS)
            results[dir_name][L_inject][alpha] = shifts
        print(f"  {dir_name} steering complete (α in {ALPHAS[1:]})")


# ============================================================================
# Print key results
# ============================================================================

print("\n" + "=" * 78)
print("RESULTS — projection shifts under UP-steering vs RAND-steering")
print("=" * 78)

for L_inject in L_INJECT_LIST:
    print(f"\n  L_inject = {L_inject}, showing readout shifts at L_inject+2, +4, +6, last")
    readout_layers = [L_inject + 2, L_inject + 4, L_inject + 6, N_LAYERS - 1]
    readout_layers = [L for L in readout_layers if L < N_LAYERS]
    for alpha in [1.0, 2.0, 4.0, 8.0]:
        print(f"\n    α = {alpha}")
        print(f"      {'L_readout':<10}  {'ΔMAG (UP)':>10}  {'ΔMAG (RND)':>10}  "
              f"{'ΔVAL (UP)':>10}  {'ΔVAL (RND)':>10}  {'ΔDIR (UP)':>10}  {'ΔDIR (RND)':>10}")
        up_shifts = results["UP"][L_inject][alpha]
        rd_shifts = results["RAND"][L_inject][alpha]
        for L_ro in readout_layers:
            print(f"      L{L_ro:<8}  {up_shifts[L_ro, 0]:>+10.3f}  {rd_shifts[L_ro, 0]:>+10.3f}  "
                  f"{up_shifts[L_ro, 1]:>+10.3f}  {rd_shifts[L_ro, 1]:>+10.3f}  "
                  f"{up_shifts[L_ro, 2]:>+10.3f}  {rd_shifts[L_ro, 2]:>+10.3f}")


# ============================================================================
# Plot: per L_inject, dose-response curves of (ΔMAG, ΔVAL, ΔDIR) at L_inject+4
# ============================================================================

fig, axes = plt.subplots(len(L_INJECT_LIST), 3, figsize=(15, 4 * len(L_INJECT_LIST)),
                          sharey=False)
if len(L_INJECT_LIST) == 1:
    axes = axes[None, :]

for i, L_inject in enumerate(L_INJECT_LIST):
    L_ro = min(L_inject + 4, N_LAYERS - 1)
    alphas_arr = np.array([0.0] + ALPHAS[1:])
    for j, (axis_name, axis_idx, color) in enumerate(
        [("MAG", 0, "tab:orange"), ("VAL", 1, "tab:green"), ("DIR", 2, "tab:blue")]
    ):
        ax = axes[i, j]
        up_vals = [0.0] + [results["UP"][L_inject][a][L_ro, axis_idx] for a in ALPHAS[1:]]
        rd_vals = [0.0] + [results["RAND"][L_inject][a][L_ro, axis_idx] for a in ALPHAS[1:]]
        ax.plot(alphas_arr, up_vals, "-o", label="UP-steering", color=color, linewidth=2)
        ax.plot(alphas_arr, rd_vals, "--s", label="RAND-steering", color="gray")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("α (steering strength)")
        ax.set_ylabel(f"Δ projection on {axis_name}_clean[L{L_ro}]")
        ax.set_title(f"L_inject={L_inject}, L_readout={L_ro}, axis={axis_name}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

fig.suptitle("exp143 — does UP-steering propagate to MAG/VAL/DIR projections downstream?\n"
             "(UP solid line = real, RAND dashed = control; gap = cross-layer mapping)",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp143_cross_layer_mappings.png", dpi=120)
print("\nSaved exp143_cross_layer_mappings.png")


# ============================================================================
# Also plot: full per-layer ΔMAG, ΔVAL, ΔDIR trajectories under α=4 UP-steering
# ============================================================================

fig, axes = plt.subplots(1, len(L_INJECT_LIST), figsize=(5 * len(L_INJECT_LIST), 5),
                          sharey=True)
if len(L_INJECT_LIST) == 1:
    axes = [axes]
xs = np.arange(N_LAYERS)
for i, L_inject in enumerate(L_INJECT_LIST):
    ax = axes[i]
    alpha = 4.0
    up_shifts = results["UP"][L_inject][alpha]
    rd_shifts = results["RAND"][L_inject][alpha]
    ax.plot(xs, up_shifts[:, 0], "-o", label="ΔMAG (UP)", color="tab:orange", markersize=3)
    ax.plot(xs, up_shifts[:, 1], "-^", label="ΔVAL (UP)", color="tab:green", markersize=3)
    ax.plot(xs, up_shifts[:, 2], "-s", label="ΔDIR (UP)", color="tab:blue", markersize=3)
    ax.plot(xs, rd_shifts[:, 0], "--", label="ΔMAG (RND)", color="tab:orange", alpha=0.4)
    ax.plot(xs, rd_shifts[:, 1], "--", label="ΔVAL (RND)", color="tab:green", alpha=0.4)
    ax.plot(xs, rd_shifts[:, 2], "--", label="ΔDIR (RND)", color="tab:blue", alpha=0.4)
    ax.axvline(L_inject, color="red", linestyle=":", alpha=0.5, label="L_inject")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("readout layer")
    ax.set_ylabel("Δ projection")
    ax.set_title(f"L_inject={L_inject}, α=4")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

fig.suptitle("exp143 — per-layer projection shifts under α=4 UP-steering vs RAND-steering",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp143_per_layer_trajectory.png", dpi=120)
print("Saved exp143_per_layer_trajectory.png")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print("""
Hypothesis: UP-steering at L_inject produces increases in MAG/VAL projections
at L_readout > L_inject, even though cos(UP_clean, MAG_clean) ~ 0 within-layer.
This would resolve the exp116-vs-exp141 apparent contradiction by showing the
UP→MORE mapping is a cross-layer transformation, not within-layer alignment.

KEY CHECK: at α=4 with L_inject=12, compare:
  - ΔMAG (UP) at L_readout=16 or 20 — should be substantially positive
  - ΔMAG (RAND) at same — should be ~zero
If the gap is large, UP→MORE cross-layer mapping is confirmed.
Same logic for ΔVAL (UP→GOOD).
""")

for L_inject in L_INJECT_LIST:
    print(f"  L_inject={L_inject}, α=4, readout L={min(L_inject+4, N_LAYERS-1)}:")
    L_ro = min(L_inject + 4, N_LAYERS - 1)
    s = results["UP"][L_inject][4.0]
    r = results["RAND"][L_inject][4.0]
    print(f"    ΔMAG  UP={s[L_ro,0]:+.3f}  RAND={r[L_ro,0]:+.3f}  "
          f"gap={s[L_ro,0]-r[L_ro,0]:+.3f}")
    print(f"    ΔVAL  UP={s[L_ro,1]:+.3f}  RAND={r[L_ro,1]:+.3f}  "
          f"gap={s[L_ro,1]-r[L_ro,1]:+.3f}")
    print(f"    ΔDIR  UP={s[L_ro,2]:+.3f}  RAND={r[L_ro,2]:+.3f}  "
          f"gap={s[L_ro,2]-r[L_ro,2]:+.3f}")

print("\nDone.")
