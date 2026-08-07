"""
exp124_salience_attention_test.py — Niamh's hypothesis:
"the anisotropy direction in transformer residual streams encodes
attention-routing / salience information."

TEST 1 (observational): for each token in a corpus sample, at multiple
layers, compute:
  - projection of its residual onto the "common direction" at that layer
    (three operationalisations: mean direction, PC1, freq axis)
  - residual norm (control)
  - position index (control)
  - log token frequency (control; SUBTLEX via Brysbaert)
  - total attention RECEIVED from downstream positions (across all heads
    at the same layer), normalised by number of downstream queries

If salience-via-common-direction is the routing signal, the projection
onto the common direction should correlate with attention received,
above and beyond the position/norm/frequency baselines.

Test at layers 4 (early), 12 (mid), 20 (late) of Pythia 410M.
"""

import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()
print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")


# ============================================================================
# Corpus: a variety of natural text
# ============================================================================

PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then disappears into the forest.",
    "Researchers studying the climate have found that ocean temperatures are rising at an unprecedented rate, with serious consequences for marine ecosystems.",
    "She walked into the room carrying a cup of tea and a manuscript that she had been editing for the past three weeks.",
    "The algorithm was designed to identify patterns in large datasets, but its developers warned that it should not be used to make decisions about individual people without human oversight.",
    "When the storm finally cleared, the villagers emerged from their homes to assess the damage that the night had brought to their fields and roads.",
    "His argument was that the modern conception of the self emerged in the seventeenth century with the philosophical writings of Descartes and his contemporaries.",
    "On Tuesday morning the central bank raised interest rates by half a percentage point, citing persistent inflation in the housing and energy sectors.",
    "The child looked up at the painted ceiling, her mouth open in astonishment at the elaborate scenes of angels and clouds and golden light.",
    "After the surgery, the patient was monitored carefully for several days to ensure that there were no complications related to the anaesthesia or the incision.",
    "Linguists have long debated whether the structure of a language influences the way its speakers perceive the world around them.",
    "In the depths of the cave the temperature stayed remarkably constant throughout the year, varying by less than two degrees Celsius between summer and winter.",
    "When she sang, the audience grew quiet and leaned forward, every face turned toward the small figure on the stage with the bright red dress.",
]


# ============================================================================
# Frequency lookup (SUBTLEX via Brysbaert)
# ============================================================================

brys = pd.read_csv("/Users/macn/Documents/embeddingexp/norms/Brysbaert_concreteness.txt",
                   sep="\t")
brys["word_lc"] = brys["Word"].str.lower()
freq_lookup = dict(zip(brys["word_lc"], brys["SUBTLEX"]))


# ============================================================================
# Frequency axis (same as v3)
# ============================================================================

COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]


def get_resid_at_layer(text, layer):
    """Get residual at the last token of `text` for a given layer."""
    hook = f"blocks.{layer}.hook_resid_post"
    toks = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook)
    return cache[hook][0, -1, :].clone()


# ============================================================================
# Per-layer measurement loop
# ============================================================================

LAYERS = [4, 12, 20]

# Per-layer collected data
records = {L: [] for L in LAYERS}

# Frequency axes per layer
print("\nBuilding freq axis at each layer...")
freq_axes = {}
for L in LAYERS:
    common_acts = torch.stack([get_resid_at_layer(w, L) for w in COMMON]).mean(0)
    rare_acts = torch.stack([get_resid_at_layer(w, L) for w in RARE]).mean(0)
    raw = common_acts - rare_acts
    freq_axes[L] = (raw / raw.norm()).cpu().numpy()
    print(f"  L{L}: freq axis built")


print(f"\nProcessing {len(PROMPTS)} prompts...")
hooks_needed = ([f"blocks.{L}.hook_resid_post" for L in LAYERS] +
                [f"blocks.{L}.attn.hook_pattern" for L in LAYERS])

for p_idx, prompt in enumerate(PROMPTS):
    toks = model.to_tokens(prompt)
    seq_len = toks.shape[1]
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hooks_needed)

    token_strs = [model.tokenizer.decode([toks[0, i].item()]).strip().lower()
                  for i in range(seq_len)]

    for L in LAYERS:
        resids = cache[f"blocks.{L}.hook_resid_post"][0]   # [seq, d]
        attns = cache[f"blocks.{L}.attn.hook_pattern"][0]  # [n_heads, q, k]

        for j in range(1, seq_len - 1):  # skip BOS and last
            # Total attention received by position j from all downstream queries
            # at this layer, all heads
            n_downstream = seq_len - 1 - j  # queries i > j
            if n_downstream <= 0:
                continue
            attn_received_total = attns[:, j+1:, j].sum().item()
            attn_received_per_query = attn_received_total / (
                n_downstream * model.cfg.n_heads)

            r = resids[j].cpu().numpy()
            records[L].append({
                "prompt_idx": p_idx,
                "position": j,
                "token": token_strs[j],
                "seq_len": seq_len,
                "residual_norm": float(np.linalg.norm(r)),
                "attn_received_total": attn_received_total,
                "attn_received_per_query": attn_received_per_query,
                "n_downstream": n_downstream,
                "log_freq": float(np.log1p(freq_lookup.get(token_strs[j], 0.0))),
                "residual": r,
            })

    if (p_idx + 1) % 3 == 0:
        print(f"  {p_idx+1}/{len(PROMPTS)} prompts processed")


# ============================================================================
# Per-layer: compute common direction (multiple operationalisations)
#            and correlate projections with attention received
# ============================================================================

print("\n" + "=" * 76)
print("RESULTS")
print("=" * 76)


def safe_corr(x, y, kind="pearson"):
    fn = pearsonr if kind == "pearson" else spearmanr
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan"), float("nan")
    r, p = fn(x, y)
    return r, p


for L in LAYERS:
    rows = records[L]
    resids = np.stack([r["residual"] for r in rows], axis=0)  # [N, d]
    attn_per_q = np.array([r["attn_received_per_query"] for r in rows])
    attn_total = np.array([r["attn_received_total"] for r in rows])
    norms = np.array([r["residual_norm"] for r in rows])
    positions = np.array([r["position"] for r in rows])
    log_freq = np.array([r["log_freq"] for r in rows])
    seq_lens = np.array([r["seq_len"] for r in rows])
    # relative position fraction (controls for sequence length)
    rel_pos = positions / seq_lens

    # Three operationalisations of "common direction":
    #   (a) mean of all residuals at this layer
    #   (b) PC1 of residuals
    #   (c) freq axis (COMMON-RARE)
    mean_dir = resids.mean(axis=0)
    mean_dir_unit = mean_dir / np.linalg.norm(mean_dir)
    pc1 = np.linalg.svd(resids - resids.mean(axis=0), full_matrices=False)[2][0]
    pc1_unit = pc1 / np.linalg.norm(pc1)
    freq_unit = freq_axes[L]

    proj_mean = resids @ mean_dir_unit
    proj_pc1 = resids @ pc1_unit
    proj_freq = resids @ freq_unit

    # Compare directions
    cos_mean_pc1 = float(mean_dir_unit @ pc1_unit)
    cos_mean_freq = float(mean_dir_unit @ freq_unit)
    cos_pc1_freq = float(pc1_unit @ freq_unit)

    print(f"\n----- Layer {L} -----  (N = {len(rows)} tokens)")
    print(f"  ‖mean_dir‖={np.linalg.norm(mean_dir):.2f}  PC1-vs-mean cos={cos_mean_pc1:+.3f}  "
          f"freq-vs-mean cos={cos_mean_freq:+.3f}  PC1-vs-freq cos={cos_pc1_freq:+.3f}")

    print(f"\n  Pearson correlations with attention received PER QUERY:")
    print(f"  {'predictor':<28}  {'pearson r':>10}  {'p-value':>10}")
    for name, x in [
        ("projection onto mean_dir",  proj_mean),
        ("projection onto PC1",       proj_pc1),
        ("projection onto freq axis", proj_freq),
        ("residual norm",             norms),
        ("absolute position",         positions),
        ("relative position",         rel_pos),
        ("log token frequency",       log_freq),
    ]:
        r, p = safe_corr(x, attn_per_q)
        print(f"  {name:<28}  {r:>+10.3f}  {p:>10.2e}")

    print(f"\n  Spearman correlations (rank-based):")
    print(f"  {'predictor':<28}  {'spearman r':>10}  {'p-value':>10}")
    for name, x in [
        ("projection onto mean_dir",  proj_mean),
        ("projection onto PC1",       proj_pc1),
        ("projection onto freq axis", proj_freq),
        ("residual norm",             norms),
        ("absolute position",         positions),
        ("relative position",         rel_pos),
        ("log token frequency",       log_freq),
    ]:
        r, p = safe_corr(x, attn_per_q, kind="spearman")
        print(f"  {name:<28}  {r:>+10.3f}  {p:>10.2e}")

    # Multiple-regression-style partial check:
    # if we residualise attn vs (position + norm + log_freq), does common-dir
    # projection still explain residual variance?
    from sklearn.linear_model import LinearRegression
    X_base = np.column_stack([positions, rel_pos, norms, log_freq])
    base_model = LinearRegression().fit(X_base, attn_per_q)
    resid_attn = attn_per_q - base_model.predict(X_base)
    print(f"\n  After controlling for [position, rel_pos, norm, log_freq]:")
    for name, x in [
        ("projection onto mean_dir",  proj_mean),
        ("projection onto PC1",       proj_pc1),
        ("projection onto freq axis", proj_freq),
    ]:
        r, p = safe_corr(x, resid_attn)
        print(f"    partial r({name}, attn_per_q) = {r:+.3f}  p={p:.2e}")


# Save
np.savez("/Users/macn/Documents/embeddingexp/exp124_results.npz",
         **{f"layer_{L}_data": np.array(
            [(r["position"], r["seq_len"], r["residual_norm"],
              r["attn_received_per_query"], r["attn_received_total"],
              r["log_freq"]) for r in records[L]],
            dtype=[("position", "i4"), ("seq_len", "i4"), ("norm", "f4"),
                   ("attn_per_q", "f4"), ("attn_total", "f4"), ("log_freq", "f4")]
         ) for L in LAYERS})
print("\nSaved exp124_results.npz")
