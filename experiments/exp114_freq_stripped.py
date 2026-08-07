"""
exp114_freq_stripped.py — strip L12 frequency axis from steering vectors,
re-run the exp111/exp112 sweep, see what survives.

exp113 found:
  cos(u_111,    freq_axis) = -0.995
  cos(u_lakoff, freq_axis) = +0.998
  cos(u_111, u_lakoff)     = -0.990
  cos(u_111_freq_stripped, u_lakoff_freq_stripped) = +0.613

So before stripping, both "UP directions" are essentially frequency axes
(opposite signs because the anchor sets have opposite freq asymmetries).
After stripping, they agree (+0.61) — the residual semantic component.

This script tests whether that residual semantic component, on its own,
still drives monotone height / affect / quantity shifts.

Three conditions per direction (exp111 anchors, Lakoff anchors):
  raw   — original vector (frequency-dominated)
  clean — projection of frequency axis removed
  rand  — random direction matched-norm (baseline)

Same DV harness as exp111/exp112.
"""

import json
from contextlib import nullcontext

import numpy as np
import torch
from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("\nLoading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"


def res(w):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    return cache[HOOK][0, -1, :].clone()


def dir_from(up, down):
    u = torch.stack([res(w) for w in up]).mean(0)
    d = torch.stack([res(w) for w in down]).mean(0)
    raw = u - d
    return raw / raw.norm(), raw.norm().item()


# ----- Anchors -----
EXP111_UP = ["high", "top", "rise", "ceiling", "above",
             "peak", "ascend", "climb", "upward", "overhead"]
EXP111_DOWN = ["low", "bottom", "fall", "floor", "below",
               "valley", "descend", "drop", "downward", "underneath"]
LAKOFF_UP = ["above", "ascend", "climb", "high", "higher", "lift", "over",
             "raise", "rise", "rising", "rose", "top", "up", "upward"]
LAKOFF_DOWN = ["below", "bottom", "descend", "down", "downward", "drop",
               "fall", "falling", "fell", "low", "lower", "sink", "under"]

# ----- Frequency axis (same as exp113) -----
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]

print("\nBuilding directions at L12...")
freq_axis, freq_norm = dir_from(COMMON, RARE)
u111_raw, n111 = dir_from(EXP111_UP, EXP111_DOWN)
ulak_raw, nlak = dir_from(LAKOFF_UP, LAKOFF_DOWN)

print(f"  ‖freq axis (raw)‖     = {freq_norm:.2f}")
print(f"  ‖u_111 (raw)‖         = {n111:.2f}")
print(f"  ‖u_lakoff (raw)‖      = {nlak:.2f}")
print(f"  cos(u_111, freq)      = {(u111_raw @ freq_axis).item():+.3f}")
print(f"  cos(u_lakoff, freq)   = {(ulak_raw @ freq_axis).item():+.3f}")
print(f"  cos(u_111, u_lakoff)  = {(u111_raw @ ulak_raw).item():+.3f}")


def strip_freq(v):
    """Project out the frequency axis, return unit vector."""
    v = v - (v @ freq_axis) * freq_axis
    return v / v.norm()


u111_clean = strip_freq(u111_raw)
ulak_clean = strip_freq(ulak_raw)
print(f"\n  cos(u_111_clean, freq)        = {(u111_clean @ freq_axis).item():+.4f}  (~0)")
print(f"  cos(u_lakoff_clean, freq)     = {(ulak_clean @ freq_axis).item():+.4f}  (~0)")
print(f"  cos(u_111_clean, u_lakoff_cl) = {(u111_clean @ ulak_clean).item():+.3f}")
print(f"  cos(u_111_clean, u_111_raw)   = {(u111_clean @ u111_raw).item():+.3f}")
print(f"  cos(u_lakoff_clean, u_lakoff_raw) = {(ulak_clean @ ulak_raw).item():+.3f}")

torch.manual_seed(7)
rand_unit = torch.randn_like(u111_raw)
rand_unit = rand_unit / rand_unit.norm()


# ----- DVs (identical to exp111) -----

HEIGHT_PROMPTS = [
    "His height in centimetres is",
    "Her height in centimetres is",
    "The man's height in centimetres is",
    "The woman's height in centimetres is",
]
HEIGHT_VALUES = list(range(140, 225, 5))

AFFECT_PROMPTS = [
    "When I think about the future, I feel",
    "My current mood is",
    "Today I am feeling",
    "The state of my mind right now is",
]
POS_WORDS = [" hopeful", " optimistic", " excited", " happy", " elated", " uplifted"]
NEG_WORDS = [" anxious", " sad", " worried", " depressed", " hopeless", " low"]

QUANTITY_PROMPTS = [
    "The number of items on the table is",
    "The number of people in the room is",
    "The number of books on the shelf is",
]
QUANTITY_VALUES = list(range(2, 51, 2))


def make_hook(d, s):
    def hook_fn(resid, hook):
        return resid + s * d
    return hook_fn


def steered_context(d, s):
    if s == 0.0:
        return nullcontext()
    return model.hooks(fwd_hooks=[(HOOK, make_hook(d, s))])


def next_token_logprobs(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return torch.log_softmax(logits[0, -1, :], dim=-1)


def expected_value_battery(direction, strength, prompts, values):
    suf = {v: model.tokenizer.encode(f" {v}", add_special_tokens=False) for v in values}
    means = []
    with steered_context(direction, strength):
        for prompt in prompts:
            pids = model.tokenizer.encode(prompt, add_special_tokens=False)
            lp = {}
            for v in values:
                s = suf[v]
                full = pids + s
                tokens = torch.tensor([full], device=device)
                with torch.no_grad():
                    logits = model(tokens)
                logp = torch.log_softmax(logits[0], dim=-1)
                start = len(pids) - 1
                lp[v] = sum(logp[start + i, tid].item() for i, tid in enumerate(s))
            lps = np.array([lp[v] for v in values])
            lps -= lps.max()
            p = np.exp(lps); p /= p.sum()
            means.append(float(np.sum(np.array(values) * p)))
    return float(np.mean(means))


def affect_dv(direction, strength):
    pos_ids = [model.tokenizer.encode(w, add_special_tokens=False)[0]
               for w in POS_WORDS
               if len(model.tokenizer.encode(w, add_special_tokens=False)) == 1]
    neg_ids = [model.tokenizer.encode(w, add_special_tokens=False)[0]
               for w in NEG_WORDS
               if len(model.tokenizer.encode(w, add_special_tokens=False)) == 1]
    diffs = []
    with steered_context(direction, strength):
        for p in AFFECT_PROMPTS:
            lp = next_token_logprobs(p)
            diffs.append((torch.logsumexp(lp[pos_ids], 0) - torch.logsumexp(lp[neg_ids], 0)).item())
    return float(np.mean(diffs))


# ----- Run -----

STRENGTHS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]

CONDITIONS = [
    ("u111_raw",   u111_raw),
    ("u111_clean", u111_clean),
    ("ulak_raw",   ulak_raw),
    ("ulak_clean", ulak_clean),
    ("rand",       rand_unit),
    ("freq_only",  freq_axis),     # NEW — steer along PURE frequency
]

results = {name: {"eh": [], "af": [], "eq": []} for name, _ in CONDITIONS}

print("\n" + "=" * 76)
print("PART A — sweep on all 6 conditions")
print("=" * 76)

for name, d in CONDITIONS:
    print(f"\n--- {name} ---")
    for s in STRENGTHS:
        eh = expected_value_battery(d, s, HEIGHT_PROMPTS, HEIGHT_VALUES)
        af = affect_dv(d, s)
        eq = expected_value_battery(d, s, QUANTITY_PROMPTS, QUANTITY_VALUES)
        results[name]["eh"].append(eh)
        results[name]["af"].append(af)
        results[name]["eq"].append(eq)
        print(f"  s={s:>+4}  E[h]={eh:6.2f}  af={af:+.3f}  E[q]={eq:5.2f}")


# ----- Compact comparison -----

print("\n" + "=" * 76)
print("Δ from baseline (s=0) for each condition — same DVs, 4 directions")
print("=" * 76)

i0 = STRENGTHS.index(0)

for dv, lab in [("eh", "E[height] cm"), ("af", "affect"), ("eq", "E[quantity]")]:
    print(f"\n{lab} — Δ from baseline")
    print(f"{'s':>4}  {'u111_raw':>10}  {'u111_clean':>11}  {'ulak_raw':>10}  "
          f"{'ulak_clean':>11}  {'rand':>8}  {'freq_only':>10}")
    for i, s in enumerate(STRENGTHS):
        row = [s]
        for name, _ in CONDITIONS:
            row.append(results[name][dv][i] - results[name][dv][i0])
        print(f"{row[0]:>+4}  {row[1]:>+10.3f}  {row[2]:>+11.3f}  "
              f"{row[3]:>+10.3f}  {row[4]:>+11.3f}  {row[5]:>+8.3f}  {row[6]:>+10.3f}")


# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp114_results.npz",
    strengths=np.array(STRENGTHS),
    **{f"{name}_{dv}": np.array(results[name][dv])
       for name, _ in CONDITIONS for dv in ["eh", "af", "eq"]},
    cos_111_lakoff_raw=(u111_raw @ ulak_raw).item(),
    cos_111_lakoff_clean=(u111_clean @ ulak_clean).item(),
    cos_111_freq=(u111_raw @ freq_axis).item(),
    cos_lakoff_freq=(ulak_raw @ freq_axis).item(),
    layer=LAYER,
)

print("\n" + "=" * 76)
print("Saved exp114_results.npz")
print("=" * 76)
