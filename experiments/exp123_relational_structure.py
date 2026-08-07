"""
exp123_relational_structure.py — do Lakoff image schemas form a coherent
relational system across layers of Pythia 410M?

v4 lab notebook Entry 1.

The shift in framing from v3: instead of asking "is UP encoded at L12?",
ask "does the N×N inter-schema cosine matrix have stable structure across
all layers, in a way random word-set partitions would not?"

Per the v4 design doc:

1. 8 schemas from LAKOFF_SCHEMAS_MML (Lakoff & Espenson 1991 MML):
   UP-DOWN, IN-OUT_CLEAN, FORWARD-BACK, LIGHT-DARK, FORCE, BALANCE,
   DIFFICULTY-BURDEN, PATH-MOTION.  (EXISTENCE dropped — composed, not
   primary.)

2. Per layer 0..23 of Pythia 410M:
     - Extract residuals for all unique anchor words + frequency words
     - Build freq axis at that layer
     - Build raw direction for each schema, freq-strip
     - Compute 8×8 schema cosine matrix

3. Strong null: K=100 trials of random partitions of the pooled anchor
   vocabulary into 8 pseudo-schemas of matched sizes, run same procedure.

4. Weak null: K=100 trials of random unit vectors at each layer.

5. Metrics:
   M1 — per-pair std of cos across layers (lower = more stable)
   M2 — across-layer configuration similarity matrix
   M3 — predicted-coupling specificity test
   M4 — antonymy sanity (cos(schema, flipped(schema)) ≈ -1)
"""

import json
from itertools import combinations

import numpy as np
import torch
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

# ============================================================================
# Setup
# ============================================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("\nLoading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
N_LAYERS = model.cfg.n_layers
D_MODEL = model.cfg.d_model
print(f"  {N_LAYERS} layers, d_model={D_MODEL}")


# ============================================================================
# Anchor preparation
# ============================================================================

# Drop EXISTENCE (composed) and the unclean IN-OUT (use the clean variant).
SCHEMA_NAMES = [
    "UP-DOWN",
    "IN-OUT_CLEAN",
    "FORWARD-BACK",
    "PATH-MOTION",
    "LIGHT-DARK",
    "FORCE",
    "BALANCE",
    "DIFFICULTY-BURDEN",
]
SCHEMAS = {name: LAKOFF_SCHEMAS_MML[name] for name in SCHEMA_NAMES}


def n_toks(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


# NOTE: do NOT filter to single-token only. Pythia BPE splits a lot of
# rare/morphologically-marked words into subwords. v3 (exp114) used
# multi-token words directly and just took the last-position residual —
# that's the standard probing approach. Keep that here for comparability.
print("\nUsing all pairs (no single-token filter; last-position residual used)")
SCHEMAS_FILTERED = SCHEMAS
for name, pairs in SCHEMAS.items():
    print(f"  {name:>20}: {len(pairs)} pairs (all kept)")

# Pool of all unique anchor words (used for strong null shuffling)
all_anchor_words = set()
for name, pairs in SCHEMAS_FILTERED.items():
    for a, b in pairs:
        all_anchor_words.add(a)
        all_anchor_words.add(b)
all_anchor_words = sorted(all_anchor_words)
print(f"\nUnique anchor words across all schemas: {len(all_anchor_words)}")


# ============================================================================
# Frequency axis vocabulary (same as v3)
# ============================================================================

COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
# Don't filter — same as v3 exp114
print(f"Frequency axis: {len(COMMON)} common, {len(RARE)} rare (no single-tok filter)")


# ============================================================================
# Get residuals at all layers for all needed words (cached)
# ============================================================================

ALL_WORDS = sorted(set(all_anchor_words) | set(COMMON) | set(RARE))
print(f"\nExtracting residuals at all {N_LAYERS} layers for {len(ALL_WORDS)} words...")

# Cache: residuals[word] = tensor of shape [N_LAYERS, D_MODEL]
residuals = {}

# Hook setup: collect all resid_post at all layers
hook_names = [f"blocks.{L}.hook_resid_post" for L in range(N_LAYERS)]

for i, w in enumerate(ALL_WORDS):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
    # Each cache[hook_name] is [1, T, D]; we want last position
    per_layer = torch.stack(
        [cache[hn][0, -1, :].clone() for hn in hook_names], dim=0
    )  # [N_LAYERS, D_MODEL]
    residuals[w] = per_layer.cpu()
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(ALL_WORDS)} words cached")

print("  done caching residuals")


# ============================================================================
# Direction builders
# ============================================================================


def mean_acts(words, layer):
    """Mean residual at layer for the given list of words."""
    acts = torch.stack([residuals[w][layer] for w in words], dim=0)
    return acts.mean(dim=0)


def build_direction(pos_words, neg_words, layer):
    """Mean(pos) - mean(neg) at layer, normalised."""
    pos_mean = mean_acts(pos_words, layer)
    neg_mean = mean_acts(neg_words, layer)
    raw = pos_mean - neg_mean
    norm = raw.norm()
    if norm < 1e-9:
        return None
    return raw / norm


def build_freq_axis(layer):
    return build_direction(COMMON, RARE, layer)


def strip_freq(direction, freq_axis):
    """Project freq out of direction, renormalise."""
    if direction is None or freq_axis is None:
        return None
    proj = (direction @ freq_axis) * freq_axis
    stripped = direction - proj
    n = stripped.norm()
    if n < 1e-9:
        return None
    return stripped / n


# ============================================================================
# Build all real-schema directions at all layers (clean = freq-stripped)
# ============================================================================

print("\nBuilding real schema directions at all layers...")

# Schema pos/neg word lists (deduped per schema)
schema_pos = {}
schema_neg = {}
for name, pairs in SCHEMAS_FILTERED.items():
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    schema_pos[name] = pos
    schema_neg[name] = neg
    print(f"  {name:>20}: {len(pos)} pos words, {len(neg)} neg words")

# directions[name] = tensor [N_LAYERS, D_MODEL]
real_directions = {name: torch.zeros(N_LAYERS, D_MODEL) for name in SCHEMA_NAMES}

for layer in range(N_LAYERS):
    freq_axis = build_freq_axis(layer)
    for name in SCHEMA_NAMES:
        raw = build_direction(schema_pos[name], schema_neg[name], layer)
        clean = strip_freq(raw, freq_axis)
        if clean is not None:
            real_directions[name][layer] = clean


# ============================================================================
# Compute real-schema NxN cosine matrices per layer
# ============================================================================

N_SCHEMAS = len(SCHEMA_NAMES)
real_cos_matrices = np.zeros((N_LAYERS, N_SCHEMAS, N_SCHEMAS))

for layer in range(N_LAYERS):
    for i, ni in enumerate(SCHEMA_NAMES):
        for j, nj in enumerate(SCHEMA_NAMES):
            ci = real_directions[ni][layer]
            cj = real_directions[nj][layer]
            real_cos_matrices[layer, i, j] = (ci @ cj).item()

print("\nReal schema cosine matrices computed for all layers.")


# ============================================================================
# Strong null: random anchor partitions
# ============================================================================

K_NULL = 100
print(f"\nRunning K={K_NULL} strong-null trials (random anchor partitions)...")

# Sizes match real schemas
schema_sizes = [(len(schema_pos[name]), len(schema_neg[name])) for name in SCHEMA_NAMES]

rng = np.random.default_rng(seed=42)

# null_cos_matrices[trial] = [N_LAYERS, N, N]
null_cos_matrices = np.zeros((K_NULL, N_LAYERS, N_SCHEMAS, N_SCHEMAS))

for trial in range(K_NULL):
    # Shuffle the anchor pool
    pool = list(all_anchor_words)
    rng.shuffle(pool)
    # Partition into pseudo-schemas
    pseudo_pos = []
    pseudo_neg = []
    cursor = 0
    for n_pos, n_neg in schema_sizes:
        # Take n_pos + n_neg from pool (cycle if needed)
        if cursor + n_pos + n_neg > len(pool):
            # Re-shuffle and continue
            pool = list(all_anchor_words)
            rng.shuffle(pool)
            cursor = 0
        pseudo_pos.append(pool[cursor:cursor + n_pos])
        pseudo_neg.append(pool[cursor + n_pos:cursor + n_pos + n_neg])
        cursor += n_pos + n_neg

    # Build pseudo-schema directions per layer
    for layer in range(N_LAYERS):
        freq_axis = build_freq_axis(layer)
        layer_dirs = []
        for k in range(N_SCHEMAS):
            raw = build_direction(pseudo_pos[k], pseudo_neg[k], layer)
            clean = strip_freq(raw, freq_axis)
            if clean is None:
                clean = torch.zeros(D_MODEL)
            layer_dirs.append(clean)
        # Cosine matrix
        for i in range(N_SCHEMAS):
            for j in range(N_SCHEMAS):
                null_cos_matrices[trial, layer, i, j] = (layer_dirs[i] @ layer_dirs[j]).item()

    if (trial + 1) % 10 == 0:
        print(f"  null trial {trial+1}/{K_NULL}")

print("  null trials complete")


# ============================================================================
# Weak null: random unit vectors (sanity)
# ============================================================================

print(f"\nRunning K={K_NULL} weak-null trials (random unit vectors)...")
weak_null_cos_matrices = np.zeros((K_NULL, N_LAYERS, N_SCHEMAS, N_SCHEMAS))

torch.manual_seed(13)
for trial in range(K_NULL):
    # Random unit vectors at each layer
    for layer in range(N_LAYERS):
        vecs = torch.randn(N_SCHEMAS, D_MODEL)
        vecs = vecs / vecs.norm(dim=1, keepdim=True)
        cm = (vecs @ vecs.T).numpy()
        weak_null_cos_matrices[trial, layer] = cm

print("  weak null done")


# ============================================================================
# METRIC M1 — per-pair std of cos across layers
# ============================================================================

print("\n" + "=" * 78)
print("METRIC M1 — per-pair std of cos across layers")
print("=" * 78)

# Real
real_pair_stds = np.zeros((N_SCHEMAS, N_SCHEMAS))
real_pair_means = np.zeros((N_SCHEMAS, N_SCHEMAS))
for i in range(N_SCHEMAS):
    for j in range(N_SCHEMAS):
        real_pair_stds[i, j] = real_cos_matrices[:, i, j].std()
        real_pair_means[i, j] = real_cos_matrices[:, i, j].mean()

# Null (strong): mean across trials of per-pair std
null_pair_stds = null_cos_matrices.std(axis=1).mean(axis=0)  # [N, N]
null_pair_means = null_cos_matrices.mean(axis=1).mean(axis=0)  # [N, N]
weak_null_pair_stds = weak_null_cos_matrices.std(axis=1).mean(axis=0)
weak_null_pair_means = weak_null_cos_matrices.mean(axis=1).mean(axis=0)

print(f"\nMean across all schema pairs (upper triangle, i<j):")
ut_mask = np.triu(np.ones((N_SCHEMAS, N_SCHEMAS), dtype=bool), k=1)

real_std_ut = real_pair_stds[ut_mask]
null_std_ut = null_pair_stds[ut_mask]
weak_std_ut = weak_null_pair_stds[ut_mask]
real_mean_ut = real_pair_means[ut_mask]
null_mean_ut = null_pair_means[ut_mask]

print(f"  REAL schemas:     mean(cos)={real_mean_ut.mean():+.4f}  "
      f"mean(std-across-layers)={real_std_ut.mean():.4f}")
print(f"  STRONG null:      mean(cos)={null_mean_ut.mean():+.4f}  "
      f"mean(std-across-layers)={null_std_ut.mean():.4f}")
print(f"  WEAK null (rand): mean(cos)={weak_null_pair_means[ut_mask].mean():+.4f}  "
      f"mean(std-across-layers)={weak_std_ut.mean():.4f}")

# Per-pair ratio
print(f"\nPer schema-pair STABILITY: std(real) vs std(strong-null)")
print(f"  ratio = std(real_pair) / mean(std(null_pair))")
print(f"  ratio < 1 → real more stable than null")
print()
print(f"  {'pair':<35}  {'mean(cos)':>10}  {'std(real)':>10}  {'std(null)':>10}  {'ratio':>7}")
for i in range(N_SCHEMAS):
    for j in range(i + 1, N_SCHEMAS):
        rs = real_pair_stds[i, j]
        ns = null_pair_stds[i, j]
        rm = real_pair_means[i, j]
        ratio = rs / ns if ns > 1e-9 else float('nan')
        print(f"  {SCHEMA_NAMES[i]+' ↔ '+SCHEMA_NAMES[j]:<35}  "
              f"{rm:>+10.3f}  {rs:>10.4f}  {ns:>10.4f}  {ratio:>7.2f}")


# ============================================================================
# METRIC M2 — across-layer configuration similarity
# ============================================================================

print("\n" + "=" * 78)
print("METRIC M2 — across-layer configuration similarity")
print("=" * 78)

# Vectorise upper triangle of each layer's cos matrix
def vec_ut(M):
    return M[np.triu_indices(M.shape[0], k=1)]


# Per-layer signature: vectorised upper triangle of cos matrix
real_sigs = np.array([vec_ut(real_cos_matrices[L]) for L in range(N_LAYERS)])  # [L, P]
# Cross-layer similarity (cosine of signatures)
real_layer_sim = np.zeros((N_LAYERS, N_LAYERS))
for a in range(N_LAYERS):
    for b in range(N_LAYERS):
        na = np.linalg.norm(real_sigs[a])
        nb = np.linalg.norm(real_sigs[b])
        if na > 1e-9 and nb > 1e-9:
            real_layer_sim[a, b] = np.dot(real_sigs[a], real_sigs[b]) / (na * nb)

# Null: average across trials
null_layer_sims = np.zeros((K_NULL, N_LAYERS, N_LAYERS))
for trial in range(K_NULL):
    sigs = np.array([vec_ut(null_cos_matrices[trial, L]) for L in range(N_LAYERS)])
    for a in range(N_LAYERS):
        for b in range(N_LAYERS):
            na = np.linalg.norm(sigs[a])
            nb = np.linalg.norm(sigs[b])
            if na > 1e-9 and nb > 1e-9:
                null_layer_sims[trial, a, b] = np.dot(sigs[a], sigs[b]) / (na * nb)
mean_null_layer_sim = null_layer_sims.mean(axis=0)

# Summarise: mean off-diagonal of real vs null
off = ~np.eye(N_LAYERS, dtype=bool)
print(f"\n  Mean off-diagonal of layer-similarity matrix:")
print(f"    REAL schemas:  {real_layer_sim[off].mean():+.4f}")
print(f"    STRONG null:   {mean_null_layer_sim[off].mean():+.4f}")
print(f"  (high = similar relational structure preserved across layer pairs)")

# Print the real layer-similarity matrix as a heatmap-ish text
print(f"\n  Real layer-similarity matrix (layer-by-layer, abbreviated):")
print(f"    rows/cols are layers 0..{N_LAYERS-1}")
for a in range(N_LAYERS):
    row = ""
    for b in range(N_LAYERS):
        v = real_layer_sim[a, b]
        row += f"{v:+.2f} "
    print(f"  L{a:>2}  {row}")


# ============================================================================
# METRIC M3 — Lakoff-predicted couplings
# ============================================================================

print("\n" + "=" * 78)
print("METRIC M3 — Lakoff-predicted couplings vs unpredicted couplings")
print("=" * 78)

# Predicted positive couplings, declared BEFORE running:
PREDICTED_POSITIVE = [
    ("UP-DOWN", "LIGHT-DARK"),        # GOOD-IS-UP, GOOD-IS-LIGHT both positively-oriented
    ("UP-DOWN", "BALANCE"),            # MORE-IS-UP / equilibrium-as-good
    ("LIGHT-DARK", "BALANCE"),         # ORDER-IS-LIGHT, BALANCE-IS-LIGHT
    ("FORCE", "DIFFICULTY-BURDEN"),    # FORCE bears on DIFFICULTY (HARD-IS-FORCEFUL)
    ("UP-DOWN", "FORCE"),              # CONTROL-IS-UP / FORCEFUL-IS-DOMINANT
    ("FORWARD-BACK", "PATH-MOTION"),   # canonical PATH structure
]

print(f"\n  Predicted-positive couplings: their cos across layers")
print(f"  {'pair':<40}  {'mean(cos)':>10}  {'sign-consistent?':>16}")
predicted_means = []
for (a, b) in PREDICTED_POSITIVE:
    i = SCHEMA_NAMES.index(a)
    j = SCHEMA_NAMES.index(b)
    cos_per_layer = real_cos_matrices[:, i, j]
    m = cos_per_layer.mean()
    n_pos = (cos_per_layer > 0).sum()
    consistent = f"{n_pos}/{N_LAYERS} layers > 0"
    predicted_means.append(m)
    print(f"  {a+' ↔ '+b:<40}  {m:>+10.3f}  {consistent:>16}")

unpredicted_means = []
for (i, j) in combinations(range(N_SCHEMAS), 2):
    pair = (SCHEMA_NAMES[i], SCHEMA_NAMES[j])
    if pair in PREDICTED_POSITIVE or (pair[1], pair[0]) in PREDICTED_POSITIVE:
        continue
    unpredicted_means.append(real_cos_matrices[:, i, j].mean())

print(f"\n  Summary:")
print(f"    Predicted-positive couplings:  mean={np.mean(predicted_means):+.4f}, "
      f"n={len(predicted_means)}")
print(f"    Unpredicted couplings:         mean={np.mean(unpredicted_means):+.4f}, "
      f"n={len(unpredicted_means)}")


# ============================================================================
# METRIC M4 — antonymy sanity
# ============================================================================

print("\n" + "=" * 78)
print("METRIC M4 — antonymy sanity: cos(schema, flipped(schema)) should ≈ -1")
print("=" * 78)

# Flip each schema and measure self-cos
print(f"\n  {'schema':<22}  {'cos(L=0)':>10}  {'cos(L=12)':>10}  {'cos(L=23)':>10}")
for name in SCHEMA_NAMES:
    pos = schema_pos[name]
    neg = schema_neg[name]
    for layer in [0, N_LAYERS // 2, N_LAYERS - 1]:
        freq_axis = build_freq_axis(layer)
        d_real = strip_freq(build_direction(pos, neg, layer), freq_axis)
        d_flip = strip_freq(build_direction(neg, pos, layer), freq_axis)
        if d_real is not None and d_flip is not None:
            c = (d_real @ d_flip).item()
        else:
            c = float('nan')
        if layer == 0:
            row = f"  {name:<22}  {c:>+10.3f}"
        else:
            row += f"  {c:>+10.3f}"
    print(row)


# ============================================================================
# Save
# ============================================================================

np.savez(
    "/Users/macn/Documents/embeddingexp/exp123_results.npz",
    schema_names=np.array(SCHEMA_NAMES),
    real_cos_matrices=real_cos_matrices,
    null_cos_matrices=null_cos_matrices,
    weak_null_cos_matrices=weak_null_cos_matrices,
    real_layer_sim=real_layer_sim,
    mean_null_layer_sim=mean_null_layer_sim,
    real_pair_stds=real_pair_stds,
    real_pair_means=real_pair_means,
    null_pair_stds=null_pair_stds,
    null_pair_means=null_pair_means,
    weak_null_pair_stds=weak_null_pair_stds,
    weak_null_pair_means=weak_null_pair_means,
)

with open("/Users/macn/Documents/embeddingexp/exp123_config.json", "w") as f:
    json.dump({
        "model": "pythia-410m",
        "n_layers": int(N_LAYERS),
        "d_model": int(D_MODEL),
        "schemas": {name: {
            "pos": schema_pos[name],
            "neg": schema_neg[name],
            "n_pairs_kept": len(SCHEMAS_FILTERED[name]),
            "n_pairs_original": len(SCHEMAS[name]),
        } for name in SCHEMA_NAMES},
        "common_words": COMMON,
        "rare_words": RARE,
        "n_anchor_pool": len(all_anchor_words),
        "K_null": K_NULL,
        "predicted_positive_couplings": PREDICTED_POSITIVE,
    }, f, indent=2)

print("\n" + "=" * 78)
print("Saved exp123_results.npz + exp123_config.json")
print("=" * 78)
