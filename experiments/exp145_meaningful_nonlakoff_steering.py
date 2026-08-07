"""
exp145_meaningful_nonlakoff_steering.py — does ANY meaningful steering
direction produce ΔVAL downstream, or only UP specifically?

Two interpretations exp143 didn't distinguish:

A) Selective UP→VAL learned mapping. UP-steering specifically maps to VAL
   downstream via learned HAPPY-IS-UP transformation. Other meaningful
   steering directions would produce different downstream feature shifts
   (or no specific shift).

B) General late-VAL-routing. Pythia's late layers organise any meaningful
   input around affective dimensions as a structural feature of being a
   language model. Any meaningful steering direction produces large ΔVAL
   downstream because the final substrate is valence-shaped.

Test: steer on directions that are:
- meaningful (not random noise)
- non-Lakoffian (don't correspond to image schemas)
- different in kind (syntactic, semantic, etc.)

Candidate steering axes:
1. NOUN_VS_VERB axis: pure syntactic distinction, no Lakoffian content
2. CONCRETE_VS_ABSTRACT axis: semantic distinction, no Lakoffian content
3. ANIMATE_VS_INANIMATE axis: semantic distinction, possibly some VAL bias
   but not Lakoffian

Compare to:
- UP-steering (replicated from exp143 for comparison)
- RAND-direction (from exp143-style protocol)

Read ΔMAG, ΔVAL, ΔDIR projections at downstream layers.

Result interpretation:
- If all meaningful directions produce ~equal ΔVAL: late-VAL-routing
  is global (Interpretation B). Substantive finding about transformer
  late-layer geometry.
- If only UP produces ΔVAL, others produce different shifts: selective
  Lakoff mappings (Interpretation A). Classical metaphor execution.
- If non-UP directions also produce ΔVAL but smaller than UP: mixed,
  partial late-VAL-routing plus specific UP→VAL on top.
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

# Lakoff reference axes (same as exp141, exp143)
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

# Non-Lakoffian directions
# NOUN vs VERB — syntactic distinction
NOUNS = ["dog","cat","table","chair","house","tree","book","car","river","mountain",
         "stone","window","apple","fish","road"]
VERBS = ["run","jump","eat","sleep","walk","write","sing","dance","swim","read",
         "drive","build","speak","listen","play"]

# CONCRETE vs ABSTRACT — semantic, no Lakoffian mapping
CONCRETE = ["dog","table","water","stone","apple","book","car","chair","window",
            "tree","fish","road","river","house","brick"]
ABSTRACT = ["idea","freedom","truth","justice","democracy","theory","concept",
            "principle","meaning","wisdom","reason","essence","virtue","logic","ethics"]

# ANIMATE vs INANIMATE — semantic
ANIMATE = ["dog","cat","person","child","bird","fish","horse","wolf","baby","mother",
           "father","sister","friend","teacher","worker"]
INANIMATE = ["stone","table","window","brick","cup","spoon","wall","floor","ceiling",
             "rock","glass","metal","wood","plastic","paper"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]

# Test prompts (same as exp143)
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


# ============================================================================
# Collect single-word residuals
# ============================================================================

all_words = set(COMMON + RARE)
for words in [MAGNITUDE_BIG, MAGNITUDE_SMALL, DIRECTIONAL_UP, DIRECTIONAL_DOWN,
              VALENCE_POS, VALENCE_NEG, LAKOFF_UP, LAKOFF_DOWN,
              NOUNS, VERBS, CONCRETE, ABSTRACT, ANIMATE, INANIMATE]:
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
    if (k+1) % 50 == 0:
        print(f"  {k+1}/{len(all_words)}")

# Anisotropy per layer
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
up_clean    = [build_axis_clean(LAKOFF_UP, LAKOFF_DOWN, L) for L in LAYERS]
mag_clean   = [build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L) for L in LAYERS]
val_clean   = [build_axis_clean(VALENCE_POS, VALENCE_NEG, L) for L in LAYERS]
dir_clean   = [build_axis_clean(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L) for L in LAYERS]

# Non-Lakoffian steering directions
noun_verb_clean = [build_axis_clean(NOUNS, VERBS, L) for L in LAYERS]
conc_abs_clean  = [build_axis_clean(CONCRETE, ABSTRACT, L) for L in LAYERS]
anim_inan_clean = [build_axis_clean(ANIMATE, INANIMATE, L) for L in LAYERS]

# Verify the non-Lakoffian axes are reasonably orthogonal to UP/VAL
print(f"\nSanity check at L12 — non-Lakoffian axes' alignment with Lakoff content:")
L = 12
print(f"  cos(NOUN_VERB, UP)  = {float(noun_verb_clean[L] @ up_clean[L]):+.3f}")
print(f"  cos(NOUN_VERB, VAL) = {float(noun_verb_clean[L] @ val_clean[L]):+.3f}")
print(f"  cos(CONC_ABS, UP)   = {float(conc_abs_clean[L] @ up_clean[L]):+.3f}")
print(f"  cos(CONC_ABS, VAL)  = {float(conc_abs_clean[L] @ val_clean[L]):+.3f}")
print(f"  cos(ANIM_INAN, UP)  = {float(anim_inan_clean[L] @ up_clean[L]):+.3f}")
print(f"  cos(ANIM_INAN, VAL) = {float(anim_inan_clean[L] @ val_clean[L]):+.3f}")


# ============================================================================
# Steering protocol (same as exp143)
# ============================================================================

L_INJECT = 12
ALPHAS = [1.0, 2.0, 4.0]

rng = np.random.default_rng(42)
random_dir_raw = rng.standard_normal(up_clean[L_INJECT].shape[0])
random_dir_raw = random_dir_raw - (random_dir_raw @ up_clean[L_INJECT]) * up_clean[L_INJECT]
random_dir = random_dir_raw / np.linalg.norm(random_dir_raw)


def run_with_steering(prompt, L_inject, steer_direction, alpha):
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
    )


# ============================================================================
# Main experiment
# ============================================================================

print("\n" + "=" * 78)
print("MAIN EXPERIMENT — does ΔVAL appear under non-Lakoffian meaningful steering?")
print("=" * 78)

STEER_DIRS = {
    "UP":         (up_clean[L_INJECT],         "Lakoffian (reference)"),
    "NOUN_VERB":  (noun_verb_clean[L_INJECT],  "syntactic, non-Lakoffian"),
    "CONC_ABS":   (conc_abs_clean[L_INJECT],   "semantic, non-Lakoffian"),
    "ANIM_INAN":  (anim_inan_clean[L_INJECT],  "semantic, possible VAL bias"),
    "RAND":       (random_dir,                  "noise control"),
}

# Compute baseline residuals
print(f"\nComputing baseline residuals for {len(PROMPTS)} prompts (no steering)...")
baseline_resids = []
for prompt in PROMPTS:
    baseline = run_with_steering(prompt, L_INJECT, up_clean[L_INJECT], 0.0)
    baseline_resids.append(baseline)

# results[dir_name][alpha] -> (N_LAYERS, 3) mean projection shift [MAG, VAL, DIR]
results = {}
for name, (steer_dir, description) in STEER_DIRS.items():
    print(f"\n--- Steering on {name} ({description}) ---")
    results[name] = {}
    for alpha in ALPHAS:
        shifts = np.zeros((N_LAYERS, 3))
        for p_idx, prompt in enumerate(PROMPTS):
            steered = run_with_steering(prompt, L_INJECT, steer_dir, alpha)
            diff = steered - baseline_resids[p_idx]
            for L in LAYERS:
                shifts[L, 0] += float(diff[L] @ mag_clean[L])
                shifts[L, 1] += float(diff[L] @ val_clean[L])
                shifts[L, 2] += float(diff[L] @ dir_clean[L])
        shifts /= len(PROMPTS)
        results[name][alpha] = shifts
    print(f"  α=4 readout at L16: ΔMAG={results[name][4.0][16,0]:+.3f}  "
          f"ΔVAL={results[name][4.0][16,1]:+.3f}  ΔDIR={results[name][4.0][16,2]:+.3f}")


# ============================================================================
# Comparison table — ΔVAL under each steering direction at α=4, various layers
# ============================================================================

print("\n" + "=" * 78)
print("ΔVAL under each steering direction at α=4 (key result)")
print("=" * 78)
print(f"\n  L_readout    " + "  ".join(f"{n:>10}" for n in STEER_DIRS.keys()))
for L_ro in [14, 16, 18, 20, 22, 23]:
    if L_ro >= N_LAYERS:
        continue
    row = f"  L{L_ro:<10}  "
    for name in STEER_DIRS.keys():
        row += f"{results[name][4.0][L_ro, 1]:>+10.3f}  "
    print(row)

print("\nΔMAG under each steering direction at α=4")
print(f"\n  L_readout    " + "  ".join(f"{n:>10}" for n in STEER_DIRS.keys()))
for L_ro in [14, 16, 18, 20, 22, 23]:
    if L_ro >= N_LAYERS:
        continue
    row = f"  L{L_ro:<10}  "
    for name in STEER_DIRS.keys():
        row += f"{results[name][4.0][L_ro, 0]:>+10.3f}  "
    print(row)

print("\nΔDIR under each steering direction at α=4")
print(f"\n  L_readout    " + "  ".join(f"{n:>10}" for n in STEER_DIRS.keys()))
for L_ro in [14, 16, 18, 20, 22, 23]:
    if L_ro >= N_LAYERS:
        continue
    row = f"  L{L_ro:<10}  "
    for name in STEER_DIRS.keys():
        row += f"{results[name][4.0][L_ro, 2]:>+10.3f}  "
    print(row)


# ============================================================================
# Plot — ΔVAL per layer per steering direction at α=4
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
xs = np.arange(N_LAYERS)
colors = {"UP": "tab:purple", "NOUN_VERB": "tab:orange",
          "CONC_ABS": "tab:green", "ANIM_INAN": "tab:blue", "RAND": "gray"}

for ax, axis_name, axis_idx in zip(axes, ["MAG", "VAL", "DIR"], [0, 1, 2]):
    for name in STEER_DIRS.keys():
        style = "-" if name == "UP" else "--" if name == "RAND" else "-"
        lw = 2.5 if name == "UP" else 1.5
        ax.plot(xs, results[name][4.0][:, axis_idx], style,
                label=name, color=colors[name], linewidth=lw)
    ax.axvline(L_INJECT, color="red", linestyle=":", alpha=0.5, label="L_inject")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("readout layer")
    ax.set_ylabel(f"Δ projection on {axis_name}_clean[L]")
    ax.set_title(f"Δ{axis_name} under α=4 steering, by direction")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

fig.suptitle("exp145 — does ΔVAL appear under non-Lakoffian steering?\n"
             "(UP solid bold = Lakoffian reference; others = non-Lakoffian; "
             "RAND dashed = noise)", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp145_meaningful_nonlakoff.png", dpi=120)
print("\nSaved exp145_meaningful_nonlakoff.png")


# ============================================================================
# Verdict
# ============================================================================

print("\n" + "=" * 78)
print("VERDICT — which interpretation does the data support?")
print("=" * 78)

L_ro = 16
print(f"\n  ΔVAL at L_readout={L_ro}, α=4:")
for name in STEER_DIRS.keys():
    print(f"    {name:<12}  {results[name][4.0][L_ro, 1]:+.3f}")

up_dval = results["UP"][4.0][L_ro, 1]
nonlakoff_dvals = [results[n][4.0][L_ro, 1] for n in ["NOUN_VERB", "CONC_ABS", "ANIM_INAN"]]
rand_dval = results["RAND"][4.0][L_ro, 1]

mean_nonlakoff = np.mean(nonlakoff_dvals)
print(f"\n  UP ΔVAL:                 {up_dval:+.3f}")
print(f"  Non-Lakoffian mean ΔVAL: {mean_nonlakoff:+.3f}")
print(f"  RAND ΔVAL:               {rand_dval:+.3f}")

if up_dval > 0.3 and mean_nonlakoff > 0.3 and abs(mean_nonlakoff / max(up_dval, 1e-6)) > 0.6:
    print("\n  -> SUPPORTS INTERPRETATION B (general late-VAL-routing):")
    print("     non-Lakoffian meaningful steering produces comparable ΔVAL to UP.")
    print("     Late layers route meaningful content to VAL as a structural feature.")
elif up_dval > 0.3 and mean_nonlakoff < 0.15:
    print("\n  -> SUPPORTS INTERPRETATION A (selective UP→VAL learned mapping):")
    print("     only UP-steering produces large ΔVAL; non-Lakoffian directions don't.")
    print("     UP→VAL is a specifically learned cross-domain transformation.")
else:
    print("\n  -> MIXED: partial late-VAL-routing + specific UP→VAL on top.")
    print("     Some general routing but UP has a specific additional effect.")

print("\nDone.")
