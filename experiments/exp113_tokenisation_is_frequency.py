"""
exp113_tokenisation_is_frequency.py — Niamh's catch: is the tokenisation
residue we worried about actually a FREQUENCY residue?

BPE tokenisation is frequency-determined: common words get their own token,
rare ones get split. So "multi-token" should imply "rarer." If the
exp111 vs Lakoff anchor lists have different frequency profiles (because
they have different multi-token rates), then the "UP direction" each
produces inherits a different frequency component — exactly the W2V
exp99-103 confound pattern, replayed in activation space.

Tests:
  1. Token-count audit per anchor list.
  2. Word-frequency lookup from Brysbaert (we have it on disk; concreteness
     norms include SUBTLEX frequency).
  3. Build a FREQUENCY axis at L12 (common-word residuals - rare-word residuals)
     and project both UP directions onto it.
  4. Check whether removing the frequency component flips the cosine from
     -0.99 toward positive.
"""

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"


def n_tokens(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


def res(w):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    return cache[HOOK][0, -1, :].clone()


# -------------------------------------------------------------------------
# (1) Token-count audit
# -------------------------------------------------------------------------

EXP111_UP = ["high", "top", "rise", "ceiling", "above",
             "peak", "ascend", "climb", "upward", "overhead"]
EXP111_DOWN = ["low", "bottom", "fall", "floor", "below",
               "valley", "descend", "drop", "downward", "underneath"]
LAKOFF_UP = ["above", "ascend", "climb", "high", "higher", "lift", "over",
             "raise", "rise", "rising", "rose", "top", "up", "upward"]
LAKOFF_DOWN = ["below", "bottom", "descend", "down", "downward", "drop",
               "fall", "falling", "fell", "low", "lower", "sink", "under"]


def audit(label, words):
    counts = [(w, n_tokens(w)) for w in words]
    multi = [w for w, c in counts if c > 1]
    rate = len(multi) / len(words)
    print(f"\n  {label}:")
    print(f"    multi-token: {multi}  ({len(multi)}/{len(words)} = {rate:.0%})")
    return rate, multi


print("=" * 72)
print("(1) Token-count audit per anchor list")
print("=" * 72)
r_111u, m_111u = audit("exp111 UP", EXP111_UP)
r_111d, m_111d = audit("exp111 DOWN", EXP111_DOWN)
r_lku, m_lku = audit("Lakoff UP", LAKOFF_UP)
r_lkd, m_lkd = audit("Lakoff DOWN", LAKOFF_DOWN)
print(f"\n  Multi-token asymmetry (DOWN - UP):")
print(f"    exp111: {r_111d:.0%} - {r_111u:.0%} = {(r_111d - r_111u)*100:+.0f}pp")
print(f"    Lakoff: {r_lkd:.0%} - {r_lku:.0%} = {(r_lkd - r_lku)*100:+.0f}pp")


# -------------------------------------------------------------------------
# (2) Word frequency — use Brysbaert concreteness file (SUBTLEX freq attached)
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("(2) Word-frequency lookup (Brysbaert / SUBTLEX log-frequency)")
print("=" * 72)

# Brysbaert concreteness file format: tab-separated, has Word, then various
# columns; freq column varies. Read it and find.
brys_path = "/Users/macn/Documents/embeddingexp/norms/Brysbaert_concreteness.txt"
brys = pd.read_csv(brys_path, sep="\t")
print(f"\n  Brysbaert columns: {list(brys.columns)}")

# Find the freq column
freq_col = None
for c in brys.columns:
    if "freq" in c.lower() or "subtl" in c.lower():
        freq_col = c
        break
print(f"  freq column: {freq_col}")

# Lower-case lookup
brys["word_lc"] = brys["Word"].str.lower()
lookup = dict(zip(brys["word_lc"], brys[freq_col])) if freq_col else {}


def freq_summary(label, words):
    freqs = [(w, lookup.get(w.lower(), None)) for w in words]
    found = [(w, f) for w, f in freqs if f is not None]
    missing = [w for w, f in freqs if f is None]
    if found:
        mean_f = np.mean([f for _, f in found])
        print(f"  {label}: mean log-freq = {mean_f:.3f}  (n={len(found)}/{len(words)})  "
              f"missing={missing}")
        return mean_f
    return None


mf_111u = freq_summary("exp111 UP    ", EXP111_UP)
mf_111d = freq_summary("exp111 DOWN  ", EXP111_DOWN)
mf_lku = freq_summary("Lakoff UP    ", LAKOFF_UP)
mf_lkd = freq_summary("Lakoff DOWN  ", LAKOFF_DOWN)
print(f"\n  Freq asymmetry (UP - DOWN, higher = UP-words more common):")
if mf_111u and mf_111d:
    print(f"    exp111: {mf_111u - mf_111d:+.3f}")
if mf_lku and mf_lkd:
    print(f"    Lakoff: {mf_lku - mf_lkd:+.3f}")


# -------------------------------------------------------------------------
# (3) Build a FREQUENCY axis at L12 and project the UP directions
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("(3) Build L12 FREQUENCY axis: common - rare")
print("=" * 72)

# Common single-token words (top-frequency function words, all should be single-token)
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
# Rare-ish single-token content words (still in BPE vocab so they're not split)
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]
# Add rare-multitoken aware: use words that we KNOW are multi-token to stress test
MULTI = ["valley", "ceiling", "ascend", "descend", "underneath",
         "lethargic", "ambivalent", "rendezvous", "constellation"]

print(f"\n  COMMON ({len(COMMON)}): {COMMON[:5]}...")
print(f"  RARE single-token ({len(RARE)}): {RARE}")
print(f"  MULTI-token ({len(MULTI)}):")
for w in MULTI:
    toks = model.tokenizer.encode(w, add_special_tokens=False)
    decoded = [model.tokenizer.decode([t]) for t in toks]
    print(f"    {w:>15} -> {decoded}")


def dir_from(up_words, down_words):
    u = torch.stack([res(w) for w in up_words]).mean(0)
    d = torch.stack([res(w) for w in down_words]).mean(0)
    raw = u - d
    return raw / raw.norm(), raw.norm().item()


# Frequency axis #1: common vs rare-single-token (same-tokenization, freq differs)
freq_axis_clean, freq_norm_clean = dir_from(COMMON, RARE)
# Frequency axis #2: common vs multi-token (confounded by tokenisation)
freq_axis_multi, freq_norm_multi = dir_from(COMMON, MULTI)
# Pure tokenisation axis: rare-single-token vs multi-token (controlling for rarity)
tok_axis, tok_norm = dir_from(RARE, MULTI)

print(f"\n  ‖common - rare(single-tok)‖  = {freq_norm_clean:.2f}   (pure-ish frequency)")
print(f"  ‖common - multi-token‖       = {freq_norm_multi:.2f}   (freq + tokenisation)")
print(f"  ‖rare(single) - multi-token‖ = {tok_norm:.2f}   (pure-ish tokenisation)")
print(f"  cos(pure-freq, tokenisation) = {(freq_axis_clean @ tok_axis).item():+.3f}")
print(f"  cos(pure-freq, freq+tok)     = {(freq_axis_clean @ freq_axis_multi).item():+.3f}")

# Now build the two UP directions and project
u111, _ = dir_from(EXP111_UP, EXP111_DOWN)
ulakoff, _ = dir_from(LAKOFF_UP, LAKOFF_DOWN)

print(f"\n  cos(u_111,    pure-freq)        = {(u111 @ freq_axis_clean).item():+.3f}")
print(f"  cos(u_lakoff, pure-freq)        = {(ulakoff @ freq_axis_clean).item():+.3f}")
print(f"  cos(u_111,    tokenisation)     = {(u111 @ tok_axis).item():+.3f}")
print(f"  cos(u_lakoff, tokenisation)     = {(ulakoff @ tok_axis).item():+.3f}")
print(f"  cos(u_111,    u_lakoff)         = {(u111 @ ulakoff).item():+.3f}  (the original mystery)")


# -------------------------------------------------------------------------
# (4) Remove the frequency / tokenisation components and re-check cosine
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("(4) After projecting out frequency + tokenisation axes")
print("=" * 72)


def project_out(v, axes):
    """Remove projection of v onto each axis (Gram-Schmidt orthonormalise the axes first)."""
    # Orthonormalise axes
    ortho = []
    for a in axes:
        a = a.clone()
        for o in ortho:
            a = a - (a @ o) * o
        n = a.norm()
        if n > 1e-6:
            ortho.append(a / n)
    # Project out
    for o in ortho:
        v = v - (v @ o) * o
    return v / v.norm()


# Strip freq+tokenisation
axes_to_strip = [freq_axis_clean, tok_axis]
u111_clean = project_out(u111, axes_to_strip)
ulakoff_clean = project_out(ulakoff, axes_to_strip)
print(f"\n  After stripping pure-freq + pure-tokenisation:")
print(f"    cos(u_111_stripped, u_lakoff_stripped) = "
      f"{(u111_clean @ ulakoff_clean).item():+.3f}")
print(f"    (was {(u111 @ ulakoff).item():+.3f})")

# Strip the combined freq+tok axis
u111_clean2 = project_out(u111, [freq_axis_multi])
ulakoff_clean2 = project_out(ulakoff, [freq_axis_multi])
print(f"\n  After stripping only the freq+tok combined axis:")
print(f"    cos = {(u111_clean2 @ ulakoff_clean2).item():+.3f}")
