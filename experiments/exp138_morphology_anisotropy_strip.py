"""
exp138_morphology_anisotropy_strip.py — redo morphology × Lakoff schema
heatmap with proper anisotropy stripping.

exp137 revealed morphology directions have |cos with anisotropy| ~ 0.55-0.59.
That contamination drove the "all suffixes converge to one direction" finding.

Here:
1. Build suffix directions with anisotropy + freq strip
2. Build schema directions with anisotropy + freq strip (consistency)
3. Compute suffix × schema cosine matrix per layer
4. Plot heatmap — does morphology still look like one direction, or do
   individual suffix → schema mappings emerge?

If suffixes now show distinct projections (different schemas for different
suffixes), the original prediction (-ING → PATH, etc.) gets a second chance.
If still uniform → morphology genuinely has unified inflectedness even after
cleaning.
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

LAYERS_OF_INTEREST = [4, 8, 12, 16, 20]
hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS_OF_INTEREST]

SUFFIX_PAIRS = {
    "ER_comparative": [("big","bigger"),("small","smaller"),("tall","taller"),
        ("high","higher"),("low","lower"),("deep","deeper"),("wide","wider"),
        ("fast","faster"),("slow","slower"),("old","older"),("new","newer"),
        ("hot","hotter"),("cold","colder"),("hard","harder"),("soft","softer")],
    "EST_superlative": [("big","biggest"),("small","smallest"),("tall","tallest"),
        ("high","highest"),("low","lowest"),("deep","deepest"),("old","oldest"),
        ("new","newest"),("hot","hottest"),("cold","coldest"),("hard","hardest")],
    "ING_progressive": [("walk","walking"),("run","running"),("jump","jumping"),
        ("sit","sitting"),("stand","standing"),("swim","swimming"),("think","thinking"),
        ("talk","talking"),("sing","singing"),("dance","dancing"),("play","playing"),
        ("work","working"),("read","reading"),("write","writing"),("eat","eating")],
    "ED_past": [("walk","walked"),("jump","jumped"),("look","looked"),
        ("talk","talked"),("play","played"),("work","worked"),("ask","asked"),
        ("call","called"),("learn","learned"),("move","moved"),("stop","stopped"),
        ("start","started")],
    "S_plural": [("cat","cats"),("dog","dogs"),("book","books"),("house","houses"),
        ("car","cars"),("tree","trees"),("bird","birds"),("hand","hands"),
        ("eye","eyes"),("girl","girls"),("boy","boys"),("year","years")],
    "UN_negation": [("happy","unhappy"),("kind","unkind"),("healthy","unhealthy"),
        ("safe","unsafe"),("clear","unclear"),("clean","unclean"),("fair","unfair"),
        ("certain","uncertain"),("known","unknown"),("seen","unseen")],
    "RE_repetition": [("do","redo"),("make","remake"),("build","rebuild"),
        ("write","rewrite"),("read","reread"),("start","restart"),
        ("create","recreate"),("paint","repaint")],
}

SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# Collect all words for anisotropy computation
all_words = set(COMMON + RARE)
for pairs in SUFFIX_PAIRS.values():
    for b, i in pairs:
        all_words.add(b); all_words.add(i)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)
print(f"\nCollecting residuals for {len(all_words)} words at {LAYERS_OF_INTEREST}...")

residuals = {}
for k, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                    for L in LAYERS_OF_INTEREST}
    if (k+1) % 100 == 0:
        print(f"  {k+1}/{len(all_words)}")


# Anisotropy direction per layer
anisotropy_dirs = {}
for L in LAYERS_OF_INTEREST:
    all_r = np.stack([residuals[w][L] for w in all_words], axis=0)
    m = all_r.mean(axis=0)
    anisotropy_dirs[L] = m / np.linalg.norm(m)


def mean_acts(words, layer):
    return np.mean([residuals[w][layer] for w in words], axis=0)


def strip_aniso_freq(direction, layer):
    """Project out per-layer anisotropy AND freq axis, renormalise."""
    # Build freq axis
    freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
    freq = freq_raw / np.linalg.norm(freq_raw)
    # Orthonormalise the two stripping axes (Gram-Schmidt)
    aniso = anisotropy_dirs[layer]
    # aniso first
    direction = direction - (direction @ aniso) * aniso
    # then freq (orthogonalised against aniso first)
    freq_orth = freq - (freq @ aniso) * aniso
    freq_orth = freq_orth / np.linalg.norm(freq_orth)
    direction = direction - (direction @ freq_orth) * freq_orth
    return direction / np.linalg.norm(direction)


def build_schema_direction(schema_name, layer):
    pairs = LAKOFF_SCHEMAS_MML[schema_name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    raw = mean_acts(pos, layer) - mean_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


def build_suffix_direction(suffix_name, layer):
    pairs = SUFFIX_PAIRS[suffix_name]
    diffs = []
    for base, infl in pairs:
        b = residuals[base][layer]; i = residuals[infl][layer]
        b_unit = b / np.linalg.norm(b); i_unit = i / np.linalg.norm(i)
        diffs.append(i_unit - b_unit)
    raw = np.mean(diffs, axis=0)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


# Verify anisotropy now stripped
print("\nVerifying anisotropy strip...")
for L in [4, 12, 20]:
    for sn in list(SUFFIX_PAIRS.keys())[:3]:
        d = build_suffix_direction(sn, L)
        cos_aniso = float(d @ anisotropy_dirs[L])
        print(f"  L{L} {sn}: cos(direction, anisotropy) = {cos_aniso:+.4f}")


# Compute suffix × schema heatmap per layer
matrices = {L: np.zeros((len(SUFFIX_PAIRS), len(SCHEMA_NAMES)))
            for L in LAYERS_OF_INTEREST}

suffix_dirs_per_layer = {}
for L in LAYERS_OF_INTEREST:
    schema_dirs = {sn: build_schema_direction(sn, L) for sn in SCHEMA_NAMES}
    suffix_dirs = {sn: build_suffix_direction(sn, L) for sn in SUFFIX_PAIRS}
    suffix_dirs_per_layer[L] = suffix_dirs
    for i, suffix in enumerate(SUFFIX_PAIRS):
        for j, schema in enumerate(SCHEMA_NAMES):
            matrices[L][i, j] = float(suffix_dirs[suffix] @ schema_dirs[schema])


# Also: per-layer suffix-suffix cosine (do they still all share one direction?)
print("\n" + "=" * 78)
print("Suffix-suffix cosine mean (after anisotropy strip)")
print("=" * 78)
for L in LAYERS_OF_INTEREST:
    sd_arr = np.array([suffix_dirs_per_layer[L][sn] for sn in SUFFIX_PAIRS])
    cos_mat = sd_arr @ sd_arr.T
    off = ~np.eye(len(SUFFIX_PAIRS), dtype=bool)
    print(f"  L{L}: mean off-diag suffix-suffix cos = {cos_mat[off].mean():+.3f}  "
          f"(was ~+0.77 before anisotropy strip)")


# Print matrices
for L in LAYERS_OF_INTEREST:
    print(f"\nLayer {L} — suffix × schema cosines (anisotropy + freq stripped)")
    print(f"  {'suffix':<18}  " + "  ".join(f"{s[:9]:>9}" for s in SCHEMA_NAMES))
    for i, suffix in enumerate(SUFFIX_PAIRS):
        row = f"  {suffix:<18}  "
        for j in range(len(SCHEMA_NAMES)):
            row += f"{matrices[L][i, j]:>+9.3f}  "
        print(row)


# Plot heatmap
fig, axes = plt.subplots(1, len(LAYERS_OF_INTEREST), figsize=(20, 5), sharey=True)
for ax, L in zip(axes, LAYERS_OF_INTEREST):
    im = ax.imshow(matrices[L], cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(SCHEMA_NAMES)))
    ax.set_xticklabels([s[:8] for s in SCHEMA_NAMES], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(SUFFIX_PAIRS)))
    ax.set_yticklabels(list(SUFFIX_PAIRS.keys()), fontsize=8)
    for i in range(len(SUFFIX_PAIRS)):
        for j in range(len(SCHEMA_NAMES)):
            v = matrices[L][i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.25 else "black", fontsize=7)
    ax.set_title(f"Layer {L}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04)

fig.suptitle("exp138 — suffix × Lakoff schema cosines (PROPERLY anisotropy-stripped)\n"
             "Does morphology still collapse, or do suffix-specific schema mappings emerge?",
             fontsize=12)
plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp138_suffix_schema_clean.png", dpi=120)
print("\nSaved exp138_suffix_schema_clean.png")
