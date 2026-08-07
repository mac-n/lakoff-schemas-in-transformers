"""
exp146_concrete_directional_disambiguation.py — what IS the CONC_ABS → DIR
late-layer routing we found in exp145?

In exp145, steering on CONC_ABS at L12 produced ΔDIR (vertical
above/below axis) at L20 = +0.896 — larger than UP-steering at the
same readout point. Three competing interpretations:

H1 (spatial modelling): the model represents concrete content with
   internal spatial structure. Concrete content gets routed to ALL
   spatial axes (vertical, horizontal, containment, proximity), not
   just vertical.
H2 (physicality / hardness): the operative variable is physicality,
   not concreteness. HARD_SOFT steering should produce equivalent
   DIR routing. CONC_ABS effect is hardness leaking through.
H3 (language co-occurrence pattern): concrete content co-occurs with
   spatial language in text. The model routes concrete-content to
   spatial-language preparation. Predicts CONC_ABS routes to spatial
   axes but not to temporal axes (which DON'T systematically
   co-occur with concrete vs abstract content).
H4 (general categorial routing): late layers route any meaningful
   categorial distinction to DIR. Predicts ALL meaningful steering
   directions (including RATIONAL_EMOTIONAL) produce similar DIR
   routing. Would dampen any spatial-specific claim.

Design:
- Steering directions: CONC_ABS, HARD_SOFT, RATIONAL_EMOTIONAL,
  UP (Lakoffian reference), RAND (noise control).
- Readout axes: VERTICAL (above/below), HORIZONTAL (left/right),
  CONTAINMENT (inside/outside), PROXIMITY (near/far), TEMPORAL
  (before/after) — temporal is the non-spatial control.
- L_inject = 12, α = 4. Read at L14, 16, 18, 20, 22, 23.

Result interpretation:
- If CONC_ABS routes to all spatial but not temporal → H1 (spatial)
- If HARD_SOFT routes equivalently to CONC_ABS → H2 (physicality)
- If RATIONAL_EMOTIONAL also routes to spatial → H4 (general
  categorial routing)
- If CONC_ABS only routes to vertical specifically → narrower
  vertical-bias finding (maybe FACTS-ARE-UPRIGHT or similar)
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
# Steering direction word lists
# ============================================================================

# UP (Lakoffian reference)
LAKOFF_UP   = ["up","rise","rose","rising","ascend","raise","climb","lift",
               "above","over","top","high","higher","upward"]
LAKOFF_DOWN = ["down","fall","fell","falling","descend","drop","sink",
               "below","under","bottom","low","lower","downward"]

# Concrete vs abstract (replicate from exp145)
CONCRETE = ["dog","table","water","stone","apple","book","car","chair","window",
            "tree","fish","road","river","house","brick"]
ABSTRACT = ["idea","freedom","truth","justice","democracy","theory","concept",
            "principle","meaning","wisdom","reason","essence","virtue","logic","ethics"]

# Hard vs soft (physical property within concrete)
HARD = ["stone","metal","brick","ice","bone","rock","steel","granite","iron",
        "crystal","glass","concrete","hardwood"]
SOFT = ["feather","cotton","silk","foam","jelly","cloud","pillow","fur","velvet",
        "sponge","wool","fabric","fluff"]

# Rational vs emotional (within abstract domain)
RATIONAL = ["logic","reason","analysis","thought","calculation","deduction",
            "intellect","argument","proof","inference"]
EMOTIONAL = ["emotion","feeling","passion","instinct","affect","sentiment","mood",
             "heart","tenderness","yearning"]


# ============================================================================
# Readout axis word lists
# ============================================================================

# VERTICAL (above/below) — same as exp145 DIR axis
VERTICAL_UP   = ["above","over","atop","upward","overhead","upper"]
VERTICAL_DOWN = ["below","under","underneath","downward","beneath","lower"]

# HORIZONTAL (left/right) — limited English vocabulary
HORIZONTAL_RIGHT = ["right","rightward","east","eastward"]
HORIZONTAL_LEFT  = ["left","leftward","west","westward"]

# CONTAINMENT (inside/outside)
CONTAINMENT_IN  = ["inside","within","interior","internal","inner"]
CONTAINMENT_OUT = ["outside","exterior","external","outer","outwards"]

# PROXIMITY (near/far)
PROXIMITY_NEAR = ["near","close","nearby","adjacent","proximate"]
PROXIMITY_FAR  = ["far","distant","remote","faraway","afar"]

# TEMPORAL (before/after) — non-spatial control
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
for words in [LAKOFF_UP, LAKOFF_DOWN, CONCRETE, ABSTRACT, HARD, SOFT,
              RATIONAL, EMOTIONAL,
              VERTICAL_UP, VERTICAL_DOWN, HORIZONTAL_RIGHT, HORIZONTAL_LEFT,
              CONTAINMENT_IN, CONTAINMENT_OUT, PROXIMITY_NEAR, PROXIMITY_FAR,
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


# Build all axes at all layers
print("\nBuilding all axes at all layers...")
STEER_AXES = {
    "UP":               (LAKOFF_UP,  LAKOFF_DOWN),
    "CONC_ABS":         (CONCRETE,   ABSTRACT),
    "HARD_SOFT":        (HARD,       SOFT),
    "RATIONAL_EMOT":    (RATIONAL,   EMOTIONAL),
}
READOUT_AXES = {
    "VERTICAL":     (VERTICAL_UP,    VERTICAL_DOWN),
    "HORIZONTAL":   (HORIZONTAL_RIGHT, HORIZONTAL_LEFT),
    "CONTAINMENT":  (CONTAINMENT_IN, CONTAINMENT_OUT),
    "PROXIMITY":    (PROXIMITY_NEAR, PROXIMITY_FAR),
    "TEMPORAL":     (TEMPORAL_AFTER, TEMPORAL_BEFORE),
}

steer_axes_clean = {}
for name, (pos, neg) in STEER_AXES.items():
    steer_axes_clean[name] = [build_axis_clean(pos, neg, L) for L in LAYERS]

readout_axes_clean = {}
for name, (pos, neg) in READOUT_AXES.items():
    readout_axes_clean[name] = [build_axis_clean(pos, neg, L) for L in LAYERS]


# ============================================================================
# Sanity check — alignment of steering axes with readout axes at L12
# ============================================================================

print("\nSanity check at L12 — steering axes vs readout axes (cos at injection layer):")
print(f"  {'steer\\readout':<16}  " + "  ".join(f"{r:>11}" for r in READOUT_AXES.keys()))
L = 12
for s_name in STEER_AXES.keys():
    row = f"  {s_name:<16}  "
    for r_name in READOUT_AXES.keys():
        c = float(steer_axes_clean[s_name][L] @ readout_axes_clean[r_name][L])
        row += f"{c:>+11.3f}  "
    print(row)


# ============================================================================
# Steering protocol (same as exp143 / exp145)
# ============================================================================

L_INJECT = 12
ALPHA = 4.0

rng = np.random.default_rng(42)
random_dir_raw = rng.standard_normal(steer_axes_clean["UP"][L_INJECT].shape[0])
random_dir_raw = random_dir_raw - (random_dir_raw @ steer_axes_clean["UP"][L_INJECT]) * steer_axes_clean["UP"][L_INJECT]
random_dir = random_dir_raw / np.linalg.norm(random_dir_raw)


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

print("\n" + "=" * 78)
print("MAIN EXPERIMENT — steering × readout disambiguation")
print("=" * 78)

# Baselines (no steering)
print(f"\nBaselines for {len(PROMPTS)} prompts...")
baseline_resids = []
for prompt in PROMPTS:
    baseline = run_with_steering(prompt, steer_axes_clean["UP"][L_INJECT], 0.0)
    baseline_resids.append(baseline)

# Steering for each direction
ALL_STEER = list(STEER_AXES.keys()) + ["RAND"]
shifts = {}  # shifts[steer_name][L_readout] = dict of Δprojection per readout axis

for s_name in ALL_STEER:
    print(f"\n--- Steering on {s_name} ---")
    if s_name == "RAND":
        steer_dir = random_dir
    else:
        steer_dir = steer_axes_clean[s_name][L_INJECT]

    accum = {L: {r: 0.0 for r in READOUT_AXES.keys()} for L in LAYERS}
    for prompt in PROMPTS:
        steered = run_with_steering(prompt, steer_dir, ALPHA)
        for L in LAYERS:
            diff = steered[L] - baseline_resids[PROMPTS.index(prompt)][L]
            for r_name in READOUT_AXES.keys():
                accum[L][r_name] += float(diff @ readout_axes_clean[r_name][L])
    for L in LAYERS:
        for r_name in READOUT_AXES.keys():
            accum[L][r_name] /= len(PROMPTS)
    shifts[s_name] = accum

    # Print snapshot at L=16 and L=20
    print(f"  L16: " + "  ".join(f"{r}={shifts[s_name][16][r]:+.3f}" for r in READOUT_AXES.keys()))
    print(f"  L20: " + "  ".join(f"{r}={shifts[s_name][20][r]:+.3f}" for r in READOUT_AXES.keys()))


# ============================================================================
# Headline tables
# ============================================================================

print("\n" + "=" * 78)
print("HEADLINE — Δ projection at L=20 (where exp145 effect was largest)")
print("=" * 78)
print(f"\n  {'steer':<16}  " + "  ".join(f"{r:>11}" for r in READOUT_AXES.keys()))
for s_name in ALL_STEER:
    row = f"  {s_name:<16}  "
    for r_name in READOUT_AXES.keys():
        row += f"{shifts[s_name][20][r_name]:>+11.3f}  "
    print(row)

print("\n" + "=" * 78)
print("Δ projection at L=16 (within-network mid-late processing)")
print("=" * 78)
print(f"\n  {'steer':<16}  " + "  ".join(f"{r:>11}" for r in READOUT_AXES.keys()))
for s_name in ALL_STEER:
    row = f"  {s_name:<16}  "
    for r_name in READOUT_AXES.keys():
        row += f"{shifts[s_name][16][r_name]:>+11.3f}  "
    print(row)


# ============================================================================
# Plot — Δ projection per layer per readout axis, per steering direction
# ============================================================================

fig, axes = plt.subplots(len(READOUT_AXES), 1, figsize=(12, 3 * len(READOUT_AXES)),
                         sharex=True)
xs = np.arange(N_LAYERS)
colors = {"UP": "tab:purple", "CONC_ABS": "tab:orange",
          "HARD_SOFT": "tab:brown", "RATIONAL_EMOT": "tab:green",
          "RAND": "gray"}

for ax, r_name in zip(axes, READOUT_AXES.keys()):
    for s_name in ALL_STEER:
        ys = [shifts[s_name][L][r_name] for L in LAYERS]
        style = "-" if s_name not in ("RAND",) else "--"
        lw = 2.5 if s_name == "UP" else 1.5
        ax.plot(xs, ys, style, label=s_name, color=colors.get(s_name, "black"),
                linewidth=lw)
    ax.axvline(L_INJECT, color="red", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Δ on {r_name}")
    ax.set_title(f"Readout axis: {r_name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("readout layer")
fig.suptitle("exp146 — disambiguating CONC_ABS → DIR routing\n"
             "(across 5 readout axes; UP solid bold = Lakoffian reference; RAND dashed)",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp146_disambiguation.png", dpi=120)
print("\nSaved exp146_disambiguation.png")


# ============================================================================
# Verdict logic
# ============================================================================

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)

# Get key numbers
def get(s, L, r):
    return shifts[s][L][r]

L_KEY = 20
print(f"\n  Δ at L={L_KEY}:\n")
ca_vert  = get("CONC_ABS", L_KEY, "VERTICAL")
hs_vert  = get("HARD_SOFT", L_KEY, "VERTICAL")
re_vert  = get("RATIONAL_EMOT", L_KEY, "VERTICAL")
ca_temp  = get("CONC_ABS", L_KEY, "TEMPORAL")
ca_horiz = get("CONC_ABS", L_KEY, "HORIZONTAL")
ca_cont  = get("CONC_ABS", L_KEY, "CONTAINMENT")
ca_prox  = get("CONC_ABS", L_KEY, "PROXIMITY")
up_vert  = get("UP", L_KEY, "VERTICAL")
rand_vert = get("RAND", L_KEY, "VERTICAL")

print(f"  CONC_ABS → VERTICAL:   {ca_vert:+.3f}")
print(f"  CONC_ABS → HORIZONTAL: {ca_horiz:+.3f}")
print(f"  CONC_ABS → CONTAINMENT: {ca_cont:+.3f}")
print(f"  CONC_ABS → PROXIMITY:  {ca_prox:+.3f}")
print(f"  CONC_ABS → TEMPORAL:   {ca_temp:+.3f}  (non-spatial control)")
print(f"  HARD_SOFT → VERTICAL:  {hs_vert:+.3f}")
print(f"  RATIONAL_EMOT → VERT:  {re_vert:+.3f}")
print(f"  UP → VERTICAL:         {up_vert:+.3f}  (Lakoffian reference)")
print(f"  RAND → VERTICAL:       {rand_vert:+.3f}  (noise floor)")

print("\n  --- Hypothesis evaluation ---")

# Check CONC_ABS routes to multiple spatial but not temporal
spatial_dvals = [ca_vert, ca_horiz, ca_cont, ca_prox]
n_spatial_positive = sum(1 for x in spatial_dvals if x > 0.2)
print(f"\n  H1 (spatial modelling): CONC_ABS routes to multiple spatial axes?")
print(f"    {n_spatial_positive}/4 spatial axes with Δ > 0.2.")
print(f"    Temporal Δ = {ca_temp:+.3f} (should be small for H1)")
if n_spatial_positive >= 3 and abs(ca_temp) < 0.2:
    print(f"    → SUPPORTS H1: spatial modelling")
elif n_spatial_positive == 1:
    print(f"    → CONTRA H1: only one spatial axis affected (narrow vertical-bias finding)")
else:
    print(f"    → partial: routes to some spatial axes")

print(f"\n  H2 (physicality): HARD_SOFT routes to VERTICAL similarly to CONC_ABS?")
print(f"    HARD_SOFT → VERT = {hs_vert:+.3f} vs CONC_ABS → VERT = {ca_vert:+.3f}")
print(f"    ratio = {hs_vert/max(abs(ca_vert), 1e-6):.2f}")
if abs(hs_vert) > 0.5 * abs(ca_vert):
    print(f"    → SUPPORTS H2: physicality is operative")
else:
    print(f"    → CONTRA H2: hardness doesn't explain the effect")

print(f"\n  H4 (general categorial routing): RATIONAL_EMOT → VERT?")
print(f"    RATIONAL_EMOT → VERT = {re_vert:+.3f}")
if abs(re_vert) > 0.4 * abs(ca_vert):
    print(f"    → SUPPORTS H4: even abstract-internal distinctions route to VERT")
    print(f"       (could also be RATIONAL-IS-UP Lakoff metaphor)")
else:
    print(f"    → CONTRA H4: RATIONAL_EMOT doesn't route to vertical")

print("\nDone.")
