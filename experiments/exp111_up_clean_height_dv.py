"""
exp111_up_clean_height_dv.py — clean UP rebuild + height-completion DV
========================================================================

Out of the word2vec rabbit hole; back to causal steering on Pythia 1.4B.
Implements CAUSAL_VALIDATION_PLAN.md §3.1:

  1. Build UP from SPATIAL-ONLY contrasts (no affect leak).
  2. LITERAL DOMAIN control: "His height in centimetres is ___" → expected
     height, varied with steering coefficient. The linchpin.
  3. ABSTRACT TRANSFER: same vector → more positive affect on neutral prompts.
  4. DOSE-RESPONSE: graded numeric DV vs coefficient, with a random-direction
     control at matched norm.
  5. MANIA KICKER: at high amplitude does the SAME vector simultaneously shift
     verticality + valence + quantity? (= vertical-elevation superordinate
     cluster, not sentiment scalar.)

Anchors are strictly spatial — `uplifting/soaring/elevating` dropped per the
plan; `higher/lower` also dropped (evaluative). Single-layer steering at L12
(exp3b "clean plateau"); per-layer-strength sweep covers normal + mania regimes.
"""

import json
import re
from contextlib import nullcontext

import numpy as np
import torch
from transformer_lens import HookedTransformer

# ============================================================================
# Setup
# ============================================================================

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

print("\nLoading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"  {n_layers} layers, d_model={d_model}")

LAYER = 12  # exp3b's clean plateau for 1.4B (24 layers; L12 = mid)
HOOK = f"blocks.{LAYER}.hook_resid_post"


# ============================================================================
# Anchors — strictly spatial. NO affect leak.
# ============================================================================

UP_WORDS = ["high", "top", "rise", "ceiling", "above",
            "peak", "ascend", "climb", "upward", "overhead"]
DOWN_WORDS = ["low", "bottom", "fall", "floor", "below",
              "valley", "descend", "drop", "downward", "underneath"]

# Dropped from exp3/exp17:
#   uplifting, soaring, elevating  — affect leak (target-domain content)
#   higher, lower                  — evaluative connotation
#   rising, falling, lifting       — kept the base form (rise, fall) only
#   sinking, plummeting, dropping, lowering, collapsing  — drop side noise


def get_residual_at_last_token(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


print(f"\nBuilding UP direction at L{LAYER}...")
up_acts = torch.stack([get_residual_at_last_token(w, HOOK) for w in UP_WORDS])
down_acts = torch.stack([get_residual_at_last_token(w, HOOK) for w in DOWN_WORDS])
up_raw = up_acts.mean(0) - down_acts.mean(0)
up_unit = up_raw / up_raw.norm()
up_norm = up_raw.norm().item()
print(f"  ‖up_raw‖ = {up_norm:.2f}  (raw diff-of-means)")
print(f"  using unit vector for steering; coefficient = steering magnitude")

# Random control at matched norm (matched to the unit vector — same as up_unit).
torch.manual_seed(7)
rand_unit = torch.randn_like(up_unit)
rand_unit = rand_unit / rand_unit.norm()


# ============================================================================
# Steering plumbing
# ============================================================================

def make_hook(direction, strength):
    def hook_fn(resid, hook):
        return resid + strength * direction
    return hook_fn


def steered_context(direction, strength):
    """nullcontext if strength==0, else model.hooks context with the steering hook."""
    if strength == 0.0:
        return nullcontext()
    return model.hooks(fwd_hooks=[(HOOK, make_hook(direction, strength))])


# ============================================================================
# DV 1: HEIGHT — primary literal-domain control
#
# Two reads:
#   (a) First-digit logit shift:  P("2"|prompt) - P("1"|prompt)
#       at the next-token distribution.  Cheap, dimensionless, monotone in
#       "expected height moved from 1xx → 2xx".
#   (b) Expected-height battery:  for each h in [140, 145, ..., 220],
#       score logP(" {h}" | prompt) teacher-forced, softmax over h,
#       compute E[h].  Slower but reads the model's height belief in cm.
# ============================================================================

HEIGHT_PROMPTS = [
    "His height in centimetres is",
    "Her height in centimetres is",
    "The man's height in centimetres is",
    "The woman's height in centimetres is",
]

HEIGHT_VALUES = list(range(140, 225, 5))  # 140, 145, ..., 220


def next_token_logprobs(prompt):
    """logP over vocab at the next-token position after prompt."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return torch.log_softmax(logits[0, -1, :], dim=-1)  # [vocab]


def first_digit_dv(direction, strength, prompts=HEIGHT_PROMPTS):
    """Mean over prompts of  logP(" 2") - logP(" 1")  at next-token position.
    > 0 = model expects a height starting with 2 (i.e. 200+ cm, "tall").
    < 0 = expects 1xx.
    """
    # Token ids for " 1" and " 2" (leading space — the natural number continuation)
    tok_1 = model.tokenizer.encode(" 1", add_special_tokens=False)
    tok_2 = model.tokenizer.encode(" 2", add_special_tokens=False)
    assert len(tok_1) == 1 and len(tok_2) == 1, f"unexpected tokenization: {tok_1} {tok_2}"
    id_1, id_2 = tok_1[0], tok_2[0]

    diffs = []
    with steered_context(direction, strength):
        for p in prompts:
            lp = next_token_logprobs(p)
            diffs.append((lp[id_2] - lp[id_1]).item())
    return float(np.mean(diffs)), diffs


def expected_height(direction, strength, prompts=HEIGHT_PROMPTS, heights=HEIGHT_VALUES):
    """Teacher-forced score of each ' {h}' suffix; softmax over h, return E[h].
    """
    # Pre-tokenize suffixes once (per height — independent of prompt).
    suffix_tokens = {}
    for h in heights:
        suffix = f" {h}"
        ids = model.tokenizer.encode(suffix, add_special_tokens=False)
        suffix_tokens[h] = ids

    all_prompt_means = []
    all_logprobs = {h: [] for h in heights}
    with steered_context(direction, strength):
        for prompt in prompts:
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            # Score each suffix: logP(suffix | prompt) = sum logP(tok_i | prompt + tok_<i)
            logp_per_h = {}
            for h in heights:
                suf = suffix_tokens[h]
                full_ids = prompt_ids + suf
                tokens = torch.tensor([full_ids], device=device)
                with torch.no_grad():
                    logits = model(tokens)  # [1, T, V]
                log_probs = torch.log_softmax(logits[0], dim=-1)  # [T, V]
                # Suffix tokens occupy positions len(prompt_ids) .. len(full_ids)-1
                # Their prob is read from log_probs at positions
                # len(prompt_ids)-1 .. len(full_ids)-2.
                start = len(prompt_ids) - 1
                total = 0.0
                for i, tid in enumerate(suf):
                    total += log_probs[start + i, tid].item()
                logp_per_h[h] = total
                all_logprobs[h].append(total)

            # Softmax over h, compute E[h]
            logps = np.array([logp_per_h[h] for h in heights])
            logps -= logps.max()
            probs = np.exp(logps)
            probs /= probs.sum()
            all_prompt_means.append(float(np.sum(np.array(heights) * probs)))

    return float(np.mean(all_prompt_means)), all_prompt_means, all_logprobs


# ============================================================================
# DV 2: AFFECT — abstract-domain transfer
#
# Mass shift on positive vs negative affect completions, mean across prompts.
# ============================================================================

AFFECT_PROMPTS = [
    "When I think about the future, I feel",
    "My current mood is",
    "Today I am feeling",
    "The state of my mind right now is",
]

POS_WORDS = [" hopeful", " optimistic", " excited", " happy", " elated", " uplifted"]
NEG_WORDS = [" anxious", " sad", " worried", " depressed", " hopeless", " low"]


def affect_dv(direction, strength, prompts=AFFECT_PROMPTS,
              pos=POS_WORDS, neg=NEG_WORDS):
    """Mean over prompts of  logsumexp(P_pos) - logsumexp(P_neg).
    > 0 = more positive affect mass.
    """
    pos_ids = []
    neg_ids = []
    for w in pos:
        ids = model.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            pos_ids.append(ids[0])
        else:
            # multi-token: skip, we'll only use single-token affect words for the cheap DV
            pass
    for w in neg:
        ids = model.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            neg_ids.append(ids[0])
    assert pos_ids and neg_ids, "no single-token affect words survived tokenization"

    diffs = []
    pos_masses = []
    neg_masses = []
    with steered_context(direction, strength):
        for p in prompts:
            lp = next_token_logprobs(p)
            pos_mass = torch.logsumexp(lp[pos_ids], dim=0).item()
            neg_mass = torch.logsumexp(lp[neg_ids], dim=0).item()
            diffs.append(pos_mass - neg_mass)
            pos_masses.append(pos_mass)
            neg_masses.append(neg_mass)
    return float(np.mean(diffs)), diffs, pos_masses, neg_masses, pos_ids, neg_ids


# ============================================================================
# DV 3: QUANTITY — for mania kicker
# ============================================================================

QUANTITY_PROMPTS = [
    "The number of items on the table is",
    "The number of people in the room is",
    "The number of books on the shelf is",
]

QUANTITY_VALUES = list(range(2, 51, 2))  # 2, 4, ..., 50


def expected_quantity(direction, strength,
                      prompts=QUANTITY_PROMPTS, quants=QUANTITY_VALUES):
    """Same machinery as expected_height — for the mania-kicker check."""
    suffix_tokens = {q: model.tokenizer.encode(f" {q}", add_special_tokens=False)
                     for q in quants}
    all_means = []
    with steered_context(direction, strength):
        for prompt in prompts:
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            logp_per_q = {}
            for q in quants:
                suf = suffix_tokens[q]
                full_ids = prompt_ids + suf
                tokens = torch.tensor([full_ids], device=device)
                with torch.no_grad():
                    logits = model(tokens)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                start = len(prompt_ids) - 1
                total = sum(log_probs[start + i, tid].item()
                            for i, tid in enumerate(suf))
                logp_per_q[q] = total
            logps = np.array([logp_per_q[q] for q in quants])
            logps -= logps.max()
            probs = np.exp(logps)
            probs /= probs.sum()
            all_means.append(float(np.sum(np.array(quants) * probs)))
    return float(np.mean(all_means)), all_means


# ============================================================================
# Free-gen samples (qualitative sanity)
# ============================================================================

def free_gen(direction, strength, prompt, max_new_tokens=40,
             temperature=0.7, seed=42):
    torch.manual_seed(seed)
    with steered_context(direction, strength):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new_tokens,
                                    temperature=temperature, do_sample=True,
                                    verbose=False)
    return model.to_string(output[0])


# ============================================================================
# Run
# ============================================================================

STRENGTHS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]

results = {
    "config": {
        "model": "pythia-1.4b",
        "layer": LAYER,
        "up_words": UP_WORDS,
        "down_words": DOWN_WORDS,
        "up_raw_norm": up_norm,
        "height_prompts": HEIGHT_PROMPTS,
        "height_values": HEIGHT_VALUES,
        "affect_prompts": AFFECT_PROMPTS,
        "pos_words": POS_WORDS,
        "neg_words": NEG_WORDS,
        "quantity_prompts": QUANTITY_PROMPTS,
        "quantity_values": QUANTITY_VALUES,
        "strengths": STRENGTHS,
    },
    "up": {"first_digit": [], "expected_height": [], "affect": [], "expected_quantity": []},
    "rand": {"first_digit": [], "expected_height": [], "affect": [], "expected_quantity": []},
}

print("\n" + "=" * 72)
print("PART A — DOSE-RESPONSE on HEIGHT, AFFECT, QUANTITY")
print(f"  layer={LAYER}, single-layer steering, strengths={STRENGTHS}")
print("=" * 72)

for label, direction in [("up", up_unit), ("rand", rand_unit)]:
    print(f"\n--- direction = {label} ---")
    for s in STRENGTHS:
        fd_mean, fd_diffs = first_digit_dv(direction, s)
        eh_mean, eh_prompts, _ = expected_height(direction, s)
        af_mean, af_diffs, _, _, _, _ = affect_dv(direction, s)
        eq_mean, eq_prompts = expected_quantity(direction, s)
        results[label]["first_digit"].append(fd_mean)
        results[label]["expected_height"].append(eh_mean)
        results[label]["affect"].append(af_mean)
        results[label]["expected_quantity"].append(eq_mean)
        print(f"  s={s:>+4}  first-digit(P2-P1)={fd_mean:+.3f}  "
              f"E[h]cm={eh_mean:6.2f}  affect(pos-neg)={af_mean:+.3f}  "
              f"E[qty]={eq_mean:5.2f}")


# ============================================================================
# PART B — Free-gen samples for qualitative read
# ============================================================================

print("\n" + "=" * 72)
print("PART B — FREE-GEN samples (T=0.7) at key strengths")
print("=" * 72)

freegen_samples = {}
for prompt in AFFECT_PROMPTS[:2] + HEIGHT_PROMPTS[:1]:
    freegen_samples[prompt] = {}
    print(f"\nPROMPT: '{prompt}'")
    for s in [-8, -4, 0, 4, 8, 12]:
        # Average over 3 seeds for a slightly less seed-dependent read
        gens = []
        for seed in [42, 7, 99]:
            text = free_gen(up_unit, s, prompt, seed=seed)
            comp = text[len(prompt):] if text.startswith(prompt) else text
            gens.append(comp.strip())
        freegen_samples[prompt][s] = gens
        print(f"  [s={s:+}]")
        for g in gens:
            print(f"     · {g}")


# ============================================================================
# PART C — Mania kicker:  joint shift at high amplitude
# ============================================================================

print("\n" + "=" * 72)
print("PART C — MANIA KICKER")
print("Does ONE intervention simultaneously shift verticality + valence + quantity?")
print("=" * 72)

# Already computed above — print joint table
print(f"\n{'strength':>8}  {'E[h]cm':>8}  {'ΔE[h]':>8}  "
      f"{'affect':>8}  {'Δaffect':>8}  {'E[qty]':>8}  {'ΔE[qty]':>8}")
base_h = results["up"]["expected_height"][STRENGTHS.index(0)]
base_a = results["up"]["affect"][STRENGTHS.index(0)]
base_q = results["up"]["expected_quantity"][STRENGTHS.index(0)]
for i, s in enumerate(STRENGTHS):
    h = results["up"]["expected_height"][i]
    a = results["up"]["affect"][i]
    q = results["up"]["expected_quantity"][i]
    print(f"{s:>+8}  {h:>8.2f}  {h - base_h:>+8.2f}  "
          f"{a:>+8.3f}  {a - base_a:>+8.3f}  {q:>8.2f}  {q - base_q:>+8.2f}")


# ============================================================================
# Save
# ============================================================================

# Convert to npz-friendly
np.savez(
    "/Users/macn/Documents/embeddingexp/exp111_results.npz",
    strengths=np.array(STRENGTHS),
    up_first_digit=np.array(results["up"]["first_digit"]),
    up_expected_height=np.array(results["up"]["expected_height"]),
    up_affect=np.array(results["up"]["affect"]),
    up_expected_quantity=np.array(results["up"]["expected_quantity"]),
    rand_first_digit=np.array(results["rand"]["first_digit"]),
    rand_expected_height=np.array(results["rand"]["expected_height"]),
    rand_affect=np.array(results["rand"]["affect"]),
    rand_expected_quantity=np.array(results["rand"]["expected_quantity"]),
    up_raw_norm=up_norm,
    layer=LAYER,
)

# Also dump a JSON for free-gen samples + config
with open("/Users/macn/Documents/embeddingexp/exp111_freegen.json", "w") as f:
    json.dump({"config": results["config"],
               "freegen": freegen_samples}, f, indent=2)

print("\n" + "=" * 72)
print("Saved exp111_results.npz + exp111_freegen.json")
print("=" * 72)
