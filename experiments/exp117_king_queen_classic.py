"""
exp117_king_queen_classic.py — does the classic Mikolov king-man+woman≈queen
analogy survive in Pythia 1.4B L12 residual stream?

Two equivalent statements of the classic:
  king - man + woman ≈ queen
  → cos(king - man, queen - woman) ≈ 1     (the gender direction is shared)
  → cos(king - queen, man - woman) ≈ 1     (the royalty direction is shared)

Test in:
  (1) Raw bare-word activations at L12 (the construction the project's been
      using; we now know it's 96% frequency-axis-contaminated).
  (2) After projecting out the frequency axis.
  (3) Nearest-neighbour trick: compute king - man + woman, find the closest
      single-token word from a candidate vocabulary — does "queen" win?

Side classics for symmetry:
  paris - france + england ≈ london (geography)
  big - small + good ≈ great (degree)
  walk - walking + swimming ≈ swim (morphology)
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"


def n_toks(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


def res(w):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    return cache[HOOK][0, -1, :].clone()


def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


# ============================================================================
# Build frequency axis for stripping
# ============================================================================
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]

f_up = torch.stack([res(w) for w in COMMON]).mean(0)
f_dn = torch.stack([res(w) for w in RARE]).mean(0)
freq_axis = (f_up - f_dn) / (f_up - f_dn).norm()


def strip_freq_vec(v):
    """Strip the frequency component from a vector (not normalised)."""
    return v - (v @ freq_axis) * freq_axis


# ============================================================================
# (1) Classic analogy cosines, raw and freq-stripped
# ============================================================================

ANALOGIES = [
    # gender × royalty
    ("king", "man", "woman", "queen", "GENDER/ROYALTY"),
    # gender × parenthood
    ("father", "man", "woman", "mother", "GENDER/PARENT"),
    # gender × sibling
    ("brother", "man", "woman", "sister", "GENDER/SIBLING"),
    # gender × actor
    ("actor", "man", "woman", "actress", "GENDER/ACTOR"),
    # capital × country
    ("paris", "france", "england", "london", "CAPITAL/COUNTRY"),
    # degree
    ("big", "small", "good", "great", "DEGREE"),
    # morphology: verb → ing
    ("walk", "walking", "swimming", "swim", "MORPHOLOGY"),
]

print("\n" + "=" * 78)
print("Classic analogies — cos(a-b, d-c)  (Mikolov: should be near +1)")
print("=" * 78)
print(f"\n  Tokenisation check (single-token preferred):")
for a, b, c, d, lab in ANALOGIES:
    sizes = [n_toks(w) for w in (a, b, c, d)]
    print(f"    {lab:>20}: {a}({sizes[0]}) {b}({sizes[1]}) {c}({sizes[2]}) {d}({sizes[3]})")

print(f"\n  Format: cos(a-b, d-c) in RAW / FREQ-STRIPPED activations\n")
for a, b, c, d, lab in ANALOGIES:
    va, vb, vc, vd = res(a), res(b), res(c), res(d)
    # Raw analogy cosine
    raw = (va - vb) - (vd - vc)  # should be ~0 → cos(va-vb, vd-vc) ~1
    raw_cos = cos(va - vb, vd - vc)
    # Freq-stripped versions
    sa, sb, sc, sd = (strip_freq_vec(v) for v in (va, vb, vc, vd))
    strip_cos = cos(sa - sb, sd - sc)
    # Magnitude info
    norm_diff = (va - vb).norm().item()
    norm_diff_strip = (sa - sb).norm().item()
    print(f"  {lab:>20}: cos(raw) = {raw_cos:+.3f}   cos(freq-strip) = {strip_cos:+.3f}"
          f"    ‖a-b‖ raw={norm_diff:.1f} strip={norm_diff_strip:.1f}")


# ============================================================================
# (2) Nearest-neighbour: compute king-man+woman, who's closest?
# ============================================================================

print("\n" + "=" * 78)
print("Nearest-neighbour search: king - man + woman, who's closest?")
print("=" * 78)

CANDIDATES = [
    # Royal/gender
    "queen", "princess", "king", "prince", "duchess", "duke", "lady", "lord",
    # Family
    "mother", "father", "sister", "brother", "wife", "husband", "daughter", "son",
    # Generic
    "woman", "man", "girl", "boy", "person", "child", "human",
    # Authority/leadership
    "ruler", "monarch", "leader", "president", "queen", "empress", "emperor",
    # Distractors
    "the", "and", "of", "house", "throne", "crown",
]
CANDIDATES = sorted(set(CANDIDATES))

vk, vm, vw = res("king"), res("man"), res("woman")
target_raw = vk - vm + vw
target_strip = strip_freq_vec(vk) - strip_freq_vec(vm) + strip_freq_vec(vw)

# Compute residuals for all candidates
print("\nCandidate-residual cosines with (king - man + woman):")
print(f"  {'word':>15}  {'tokens':>6}  {'cos RAW':>10}  {'cos STRIP':>10}")
scores_raw, scores_strip = [], []
for w in CANDIDATES:
    n = n_toks(w)
    rw = res(w)
    rs = strip_freq_vec(rw)
    c_raw = cos(rw, target_raw)
    c_strip = cos(rs, target_strip)
    scores_raw.append((c_raw, w, n))
    scores_strip.append((c_strip, w, n))
    print(f"  {w:>15}  {n:>6}  {c_raw:>+10.3f}  {c_strip:>+10.3f}")

scores_raw.sort(reverse=True)
scores_strip.sort(reverse=True)

print(f"\n  Top 5 RAW    : {[(w, f'{c:+.3f}') for c, w, _ in scores_raw[:5]]}")
print(f"  Top 5 STRIP  : {[(w, f'{c:+.3f}') for c, w, _ in scores_strip[:5]]}")

# Also: exclude the input words (Linzen 2016 critique)
input_words = {"king", "man", "woman"}
raw_excl = [(c, w) for c, w, _ in scores_raw if w not in input_words][:5]
strip_excl = [(c, w) for c, w, _ in scores_strip if w not in input_words][:5]
print(f"\n  EXCLUDING input words (Linzen-honest):")
print(f"  Top 5 RAW    : {[(w, f'{c:+.3f}') for c, w in raw_excl]}")
print(f"  Top 5 STRIP  : {[(w, f'{c:+.3f}') for c, w in strip_excl]}")


# ============================================================================
# (3) Also: cos of "gender direction" using different anchor pairs
# ============================================================================

print("\n" + "=" * 78)
print("Gender direction stability across anchor pairs (cos with man-woman, RAW & STRIP)")
print("=" * 78)

GENDER_PAIRS = [
    ("man", "woman"),
    ("boy", "girl"),
    ("king", "queen"),
    ("father", "mother"),
    ("brother", "sister"),
    ("actor", "actress"),
    ("husband", "wife"),
]
base_raw = res("man") - res("woman")
base_strip = strip_freq_vec(res("man")) - strip_freq_vec(res("woman"))
print(f"\n  Anchor pair        cos(pair_raw, man-woman_raw)   cos(pair_strip, man-woman_strip)")
for m, f in GENDER_PAIRS:
    d_raw = res(m) - res(f)
    d_strip = strip_freq_vec(res(m)) - strip_freq_vec(res(f))
    print(f"  ({m:>8}, {f:>8})        {cos(d_raw, base_raw):>+.3f}                          "
          f"{cos(d_strip, base_strip):>+.3f}")

print("\n" + "=" * 78)
print("DONE.")
