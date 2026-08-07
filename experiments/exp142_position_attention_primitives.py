"""
exp142_position_attention_primitives.py — do FORWARD-BACK and LIGHT-DARK
ride substrate-architectural axes?

Hypothesis (extending exp141 BALANCE-as-norm finding):
- FORWARD-BACK rides POSITION-IN-SEQUENCE. Causal attention + positional
  encoding give Pythia a hardcoded directional asymmetry. "Past tokens"
  are earlier in residual stream; "future tokens" are later. Lakoff's
  PAST-IS-BEHIND maps onto this directly.
- LIGHT-DARK rides ATTENTION. KNOWING-IS-SEEING / UNDERSTANDING-IS-
  ILLUMINATION in Lakoff arises because visual attention is the substrate-
  given epistemic primitive for embodied beings. Transformer's substrate-
  given epistemic primitive is literal attention. Tokens receiving high
  attention = illuminated; low attention = in the dark.

Tests:
1. Build POSITION axis from natural-prompt residuals (later-position minus
   earlier-position, per layer, anisotropy-stripped). Measure cos with
   all 8 Lakoff schemas.
2. Build ATTENTION axis (high-attention-received minus low-attention-
   received residual mean, per layer, anisotropy-stripped). Measure cos
   with all 8 Lakoff schemas.
3. Specific predictions: FB rides POSITION strongly; LD rides ATTENTION
   strongly; BALANCE rides norm (already from exp141, included as cross-
   check); other schemas show no specific substrate alignment.
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
N_HEADS = model.cfg.n_heads
LAYERS = list(range(N_LAYERS))
resid_hooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
attn_hooks  = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]


# ============================================================================
# Word lists for Lakoff schemas (single-word residuals, as before)
# ============================================================================

SCHEMA_NAMES = ["UP-DOWN", "IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION",
                "LIGHT-DARK", "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# ============================================================================
# Natural prompts (for position + attention extraction)
# ============================================================================

PROMPTS = [
    "The quick brown fox jumps over the lazy dog beside the river while the "
    "moon rises slowly above the hills in the quiet evening of late autumn.",
    "When the rain finally stopped falling on the small town the children "
    "ran outside to play in the puddles that had formed along the cobbled streets.",
    "Long after the sun had set behind the mountains the travellers continued "
    "walking down the narrow path that wound through the dense and silent forest.",
    "The library contained thousands of old books arranged on tall wooden "
    "shelves that reached almost to the ceiling of the high vaulted reading room.",
    "Every morning the baker would wake before dawn and walk across the cold "
    "stone floor of his shop to begin preparing the bread for the day ahead.",
    "Scientists have long wondered how birds manage to navigate across vast "
    "distances during migration without the aid of any visible landmarks or stars.",
    "In the years following the war the small village slowly began to recover "
    "as families returned and new houses were built along the main road.",
    "She picked up the heavy wooden box and carried it carefully down the "
    "stairs into the cellar where the rest of the supplies were already stored.",
    "The orchestra began playing softly at first but gradually built in volume "
    "until the music filled the entire hall and reached the highest balcony.",
    "Although the journey would be long and difficult the explorers were "
    "determined to reach the source of the river before the season ended.",
]
print(f"Using {len(PROMPTS)} natural prompts for position + attention extraction.")


# ============================================================================
# Collect single-word residuals (for Lakoff schemas + anisotropy)
# ============================================================================

all_words = set(COMMON + RARE)
for sn in SCHEMA_NAMES:
    for p, n in LAKOFF_SCHEMAS_MML[sn]:
        all_words.add(p); all_words.add(n)
all_words = sorted(all_words)
print(f"\nCollecting single-word residuals for {len(all_words)} words at all {N_LAYERS} layers...")

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


# Anisotropy from single-word residuals (consistent with prior exps)
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


def build_schema_clean(name, layer):
    pairs = LAKOFF_SCHEMAS_MML[name]
    pos = sorted(set(p[0] for p in pairs))
    neg = sorted(set(p[1] for p in pairs))
    raw = mean_word_acts(pos, layer) - mean_word_acts(neg, layer)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


# ============================================================================
# Collect natural-prompt residuals + attention patterns
# ============================================================================

print(f"\nCollecting per-position residuals + attention patterns from {len(PROMPTS)} prompts...")

# For each prompt: store residuals (L, T, D) and attention-received (L, T)
prompt_data = []
for i, prompt in enumerate(PROMPTS):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=resid_hooks + attn_hooks)
    T = toks.shape[1]
    resids = np.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0].cpu().numpy() for L in LAYERS],
        axis=0
    )  # (L, T, D)
    # Attention-received per (layer, key_pos): sum over heads and queries.
    # attn shape: (1, n_heads, query_pos, key_pos)
    attn_received = np.zeros((N_LAYERS, T))
    for L in LAYERS:
        a = cache[f"blocks.{L}.attn.hook_pattern"][0].cpu().numpy()  # (H, Q, K)
        # Sum over heads and queries → total attention this key_pos received
        attn_received[L] = a.sum(axis=(0, 1))
    prompt_data.append({"T": T, "resids": resids, "attn_received": attn_received})
    print(f"  prompt {i+1}/{len(PROMPTS)}: T={T}")


# ============================================================================
# TEST 1 — Build POSITION axis per layer
# ============================================================================

print("\n" + "=" * 78)
print("TEST 1 — POSITION axis (later-position - earlier-position residuals)")
print("=" * 78)


def build_position_axis(layer):
    """Mean of late-half residuals minus mean of early-half residuals,
    pooled across prompts. Anisotropy + freq stripped."""
    late = []
    early = []
    for pd in prompt_data:
        T = pd["T"]
        mid = T // 2
        # Use positions 1..mid as early, mid..T-1 as late (skip BOS at 0)
        early.append(pd["resids"][layer, 1:mid].mean(axis=0))
        late.append(pd["resids"][layer, mid:].mean(axis=0))
    late_mean = np.mean(late, axis=0)
    early_mean = np.mean(early, axis=0)
    raw = late_mean - early_mean
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


position_axes = [build_position_axis(L) for L in LAYERS]

# Diagnostic: cos with anisotropy BEFORE strip
print("\n1a. Diagnostic — does raw position axis correlate with anisotropy?")
print(f"\n  layer    raw cos(POS, aniso)")
for L in [0, 4, 8, 12, 16, 20, 23]:
    late = []
    early = []
    for pd in prompt_data:
        T = pd["T"]
        mid = T // 2
        early.append(pd["resids"][L, 1:mid].mean(axis=0))
        late.append(pd["resids"][L, mid:].mean(axis=0))
    raw = np.mean(late, axis=0) - np.mean(early, axis=0)
    raw = raw / np.linalg.norm(raw)
    print(f"  L{L:>3}      {float(raw @ anisotropy_dirs[L]):>+8.3f}")

# Cross-layer stability of POSITION axis
print("\n1b. Cross-layer stability of POSITION axis (L4-L22)")
WORK = list(range(4, 23))
arr = np.array([position_axes[L] for L in WORK])
cm = arr @ arr.T
pos_stab = cm[~np.eye(len(WORK), dtype=bool)].mean()
print(f"  Mean off-diag cos = {pos_stab:+.4f}")
print(f"  (exp141 baselines: MAG=+0.77, DIR=+0.74, VAL=+0.75, LAK=+0.86)")

# cos(schema, POSITION) per layer
print("\n1c. cos(Lakoff schema, POSITION) per layer — FORWARD-BACK should win")
print(f"\n  layer    " + "  ".join(f"{s[:8]:>8}" for s in SCHEMA_NAMES))
schema_pos_cos = np.zeros((N_LAYERS, len(SCHEMA_NAMES)))
for L in LAYERS:
    for j, sn in enumerate(SCHEMA_NAMES):
        s_dir = build_schema_clean(sn, L)
        schema_pos_cos[L, j] = float(s_dir @ position_axes[L])
    if L in [0, 4, 8, 12, 16, 20, 23]:
        row = f"  L{L:>3}    "
        for j in range(len(SCHEMA_NAMES)):
            row += f"{schema_pos_cos[L, j]:>+8.3f}  "
        print(row)


# ============================================================================
# TEST 2 — Build ATTENTION axis per layer
# ============================================================================

print("\n" + "=" * 78)
print("TEST 2 — ATTENTION axis (high-attention-received vs low)")
print("=" * 78)


def build_attention_axis(layer):
    """Top-quartile-attention residuals minus bottom-quartile, pooled
    across prompts. Anisotropy + freq stripped."""
    hi = []
    lo = []
    for pd in prompt_data:
        T = pd["T"]
        a = pd["attn_received"][layer]  # (T,)
        # Skip BOS (pos 0) which often dominates attention pathologically
        valid_idx = np.arange(1, T)
        a_valid = a[1:]
        # Top/bottom quartile by attention received
        n = len(valid_idx)
        sorted_idx = np.argsort(a_valid)
        bot_idx = valid_idx[sorted_idx[:max(1, n // 4)]]
        top_idx = valid_idx[sorted_idx[-max(1, n // 4):]]
        hi.append(pd["resids"][layer, top_idx].mean(axis=0))
        lo.append(pd["resids"][layer, bot_idx].mean(axis=0))
    raw = np.mean(hi, axis=0) - np.mean(lo, axis=0)
    raw = raw / np.linalg.norm(raw)
    return strip_aniso_freq(raw, layer)


attention_axes = [build_attention_axis(L) for L in LAYERS]

# Diagnostic
print("\n2a. Diagnostic — does raw attention axis correlate with anisotropy?")
print(f"\n  layer    raw cos(ATTN, aniso)")
for L in [0, 4, 8, 12, 16, 20, 23]:
    hi = []
    lo = []
    for pd in prompt_data:
        T = pd["T"]
        a = pd["attn_received"][L]
        valid_idx = np.arange(1, T)
        a_valid = a[1:]
        n = len(valid_idx)
        sorted_idx = np.argsort(a_valid)
        bot_idx = valid_idx[sorted_idx[:max(1, n // 4)]]
        top_idx = valid_idx[sorted_idx[-max(1, n // 4):]]
        hi.append(pd["resids"][L, top_idx].mean(axis=0))
        lo.append(pd["resids"][L, bot_idx].mean(axis=0))
    raw = np.mean(hi, axis=0) - np.mean(lo, axis=0)
    raw = raw / np.linalg.norm(raw)
    print(f"  L{L:>3}      {float(raw @ anisotropy_dirs[L]):>+8.3f}")

# Cross-layer stability
print("\n2b. Cross-layer stability of ATTENTION axis (L4-L22)")
arr = np.array([attention_axes[L] for L in WORK])
cm = arr @ arr.T
attn_stab = cm[~np.eye(len(WORK), dtype=bool)].mean()
print(f"  Mean off-diag cos = {attn_stab:+.4f}")

# cos(schema, ATTENTION) per layer
print("\n2c. cos(Lakoff schema, ATTENTION) per layer — LIGHT-DARK should win")
print(f"\n  layer    " + "  ".join(f"{s[:8]:>8}" for s in SCHEMA_NAMES))
schema_attn_cos = np.zeros((N_LAYERS, len(SCHEMA_NAMES)))
for L in LAYERS:
    for j, sn in enumerate(SCHEMA_NAMES):
        s_dir = build_schema_clean(sn, L)
        schema_attn_cos[L, j] = float(s_dir @ attention_axes[L])
    if L in [0, 4, 8, 12, 16, 20, 23]:
        row = f"  L{L:>3}    "
        for j in range(len(SCHEMA_NAMES)):
            row += f"{schema_attn_cos[L, j]:>+8.3f}  "
        print(row)


# ============================================================================
# TEST 3 — winner schema per axis per layer
# ============================================================================

print("\n" + "=" * 78)
print("TEST 3 — which schema has max |cos| with each substrate axis per layer")
print("=" * 78)

print("\n3a. POSITION axis — winning schema per layer (by |cos|)")
print(f"\n  layer    winner               |cos|     runner-up           |cos|")
for L in [0, 4, 8, 12, 16, 20, 23]:
    abs_cos = np.abs(schema_pos_cos[L])
    order = np.argsort(abs_cos)[::-1]
    w1 = SCHEMA_NAMES[order[0]]; c1 = abs_cos[order[0]]
    w2 = SCHEMA_NAMES[order[1]]; c2 = abs_cos[order[1]]
    print(f"  L{L:>3}    {w1:<18}  {c1:>+6.3f}    {w2:<18}  {c2:>+6.3f}")

print("\n3b. ATTENTION axis — winning schema per layer (by |cos|)")
print(f"\n  layer    winner               |cos|     runner-up           |cos|")
for L in [0, 4, 8, 12, 16, 20, 23]:
    abs_cos = np.abs(schema_attn_cos[L])
    order = np.argsort(abs_cos)[::-1]
    w1 = SCHEMA_NAMES[order[0]]; c1 = abs_cos[order[0]]
    w2 = SCHEMA_NAMES[order[1]]; c2 = abs_cos[order[1]]
    print(f"  L{L:>3}    {w1:<18}  {c1:>+6.3f}    {w2:<18}  {c2:>+6.3f}")


# ============================================================================
# Plots
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
xs = np.arange(N_LAYERS)
colors = plt.cm.tab10(np.linspace(0, 1, len(SCHEMA_NAMES)))

ax = axes[0]
for j, sn in enumerate(SCHEMA_NAMES):
    style = "-o" if sn == "FORWARD-BACK" else "-"
    lw = 2.5 if sn == "FORWARD-BACK" else 1.0
    ax.plot(xs, schema_pos_cos[:, j], style, label=sn, color=colors[j],
            linewidth=lw, markersize=4 if sn == "FORWARD-BACK" else 0)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("cos(schema, POSITION axis)")
ax.set_title("TEST 1 — Lakoff schemas vs POSITION axis\n"
             "(FORWARD-BACK should dominate if substrate-aligned)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

ax = axes[1]
for j, sn in enumerate(SCHEMA_NAMES):
    style = "-o" if sn == "LIGHT-DARK" else "-"
    lw = 2.5 if sn == "LIGHT-DARK" else 1.0
    ax.plot(xs, schema_attn_cos[:, j], style, label=sn, color=colors[j],
            linewidth=lw, markersize=4 if sn == "LIGHT-DARK" else 0)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("cos(schema, ATTENTION axis)")
ax.set_title("TEST 2 — Lakoff schemas vs ATTENTION axis\n"
             "(LIGHT-DARK should dominate if substrate-aligned)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/macn/Documents/embeddingexp/exp142_position_attention.png", dpi=120)
print("\nSaved exp142_position_attention.png")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

# Mean |cos| across L8-L20 (stable middle range) per schema per axis
MID = list(range(8, 21))
mean_pos = np.abs(schema_pos_cos[MID]).mean(axis=0)
mean_attn = np.abs(schema_attn_cos[MID]).mean(axis=0)
print("\n  Mean |cos| across L8-L20:")
print(f"\n  {'schema':<20}    POSITION     ATTENTION")
for j, sn in enumerate(SCHEMA_NAMES):
    flag = ""
    if sn == "FORWARD-BACK" and mean_pos[j] == mean_pos.max():
        flag += " ← POS winner"
    if sn == "LIGHT-DARK" and mean_attn[j] == mean_attn.max():
        flag += " ← ATTN winner"
    print(f"  {sn:<20}    {mean_pos[j]:>+8.3f}    {mean_attn[j]:>+8.3f}{flag}")

print("\n  POSITION axis cross-layer stability: ", f"{pos_stab:+.4f}")
print("  ATTENTION axis cross-layer stability:", f"{attn_stab:+.4f}")

print("\nDone.")
