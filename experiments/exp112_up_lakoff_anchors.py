"""
exp112_up_lakoff_anchors.py — UP direction from Lakoff's literal-motion sublist
================================================================================

Anchor-set ablation of exp111. Niamh's catch: anchor selection is fragile, and
`peak` in particular is suspect (peak performance / peak experience are already
metaphorical). External validation move: use Lakoff's *Master Metaphor List*
"Literal vertical motion" block (curated by Lakoff's research team, in
`lakoff_canonical_vocabulary.py:UP_DOWN_MML` lines 51-65) instead of my
hand-picked set.

What's different vs exp111:
  exp111 (hand-picked, 10 each): high top rise ceiling above peak ascend climb
                                 upward overhead / low bottom fall floor below
                                 valley descend drop downward underneath
  exp112 (Lakoff MML literal):   up rise rose rising ascend raise climb lift
                                 above over top high higher upward / down fall
                                 fell falling descend lower drop sink below
                                 under bottom low downward

  Changes:
   + 'up'/'down' added (somehow omitted from exp111)
   + past + -ing inflections (rose/fell, rising/falling)
   + transitive forms (raise/lower, lift/sink)
   + 'over'/'under', 'higher'/'lower' added back
   - 'peak'/'valley' dropped (not in Lakoff literal-motion)
   - 'ceiling'/'floor', 'overhead'/'underneath' dropped (spatial nouns, not motion)

Same DVs, same layer, same strengths as exp111 → direct comparison of effect
sizes. Tells us whether anchor-set discipline (Lakoff-curated vs hand-picked)
changes the magnitude or shape of the dose-response.
"""

import json
from contextlib import nullcontext

import numpy as np
import torch
from transformer_lens import HookedTransformer

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

LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"


# ============================================================================
# Anchors — Lakoff MML "Literal vertical motion" sublist
# ============================================================================
# From lakoff_canonical_vocabulary.py:UP_DOWN_MML lines 51-65
# The pair structure preserved for reference; we use mean-of-means for the
# direction (same as exp111) so it's a fair anchor-set comparison.

LITERAL_PAIRS = [
    ("up", "down"),
    ("rise", "fall"),
    ("rose", "fell"),
    ("rising", "falling"),
    ("ascend", "descend"),
    ("raise", "lower"),
    ("climb", "drop"),
    ("lift", "sink"),
    ("above", "below"),
    ("over", "under"),
    ("top", "bottom"),
    ("high", "low"),
    ("higher", "lower"),     # 'lower' appears twice as DOWN partner
    ("upward", "downward"),
]

UP_WORDS = sorted({p[0] for p in LITERAL_PAIRS})
DOWN_WORDS = sorted({p[1] for p in LITERAL_PAIRS})

print(f"\nUP anchors ({len(UP_WORDS)}): {UP_WORDS}")
print(f"DOWN anchors ({len(DOWN_WORDS)}): {DOWN_WORDS}")


def get_residual_at_last_token(text, hook_name):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name][0, -1, :].clone()


print(f"\nBuilding UP direction at L{LAYER} from Lakoff anchors...")
up_acts = torch.stack([get_residual_at_last_token(w, HOOK) for w in UP_WORDS])
down_acts = torch.stack([get_residual_at_last_token(w, HOOK) for w in DOWN_WORDS])
up_raw = up_acts.mean(0) - down_acts.mean(0)
up_unit = up_raw / up_raw.norm()
up_norm = up_raw.norm().item()
print(f"  ‖up_raw_lakoff‖ = {up_norm:.2f}")
print(f"  (exp111 was 130.28)")

# Compare to exp111's direction — cosine similarity tells us how different the
# two anchor sets' directions actually are.
EXP111_UP_WORDS = ["high", "top", "rise", "ceiling", "above",
                   "peak", "ascend", "climb", "upward", "overhead"]
EXP111_DOWN_WORDS = ["low", "bottom", "fall", "floor", "below",
                     "valley", "descend", "drop", "downward", "underneath"]
up_acts_111 = torch.stack([get_residual_at_last_token(w, HOOK)
                            for w in EXP111_UP_WORDS])
down_acts_111 = torch.stack([get_residual_at_last_token(w, HOOK)
                              for w in EXP111_DOWN_WORDS])
up_raw_111 = up_acts_111.mean(0) - down_acts_111.mean(0)
up_unit_111 = up_raw_111 / up_raw_111.norm()
cos_lakoff_handpicked = (up_unit @ up_unit_111).item()
print(f"  cos(Lakoff direction, exp111 hand-picked) = {cos_lakoff_handpicked:+.3f}")

torch.manual_seed(7)
rand_unit = torch.randn_like(up_unit)
rand_unit = rand_unit / rand_unit.norm()


# ============================================================================
# DVs — identical to exp111
# ============================================================================

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


def make_hook(direction, strength):
    def hook_fn(resid, hook):
        return resid + strength * direction
    return hook_fn


def steered_context(direction, strength):
    if strength == 0.0:
        return nullcontext()
    return model.hooks(fwd_hooks=[(HOOK, make_hook(direction, strength))])


def next_token_logprobs(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    return torch.log_softmax(logits[0, -1, :], dim=-1)


def first_digit_dv(direction, strength, prompts=HEIGHT_PROMPTS):
    tok_1 = model.tokenizer.encode(" 1", add_special_tokens=False)
    tok_2 = model.tokenizer.encode(" 2", add_special_tokens=False)
    id_1, id_2 = tok_1[0], tok_2[0]
    diffs = []
    with steered_context(direction, strength):
        for p in prompts:
            lp = next_token_logprobs(p)
            diffs.append((lp[id_2] - lp[id_1]).item())
    return float(np.mean(diffs))


def expected_value_battery(direction, strength, prompts, values):
    """Generic expected-value DV: score ' {v}' suffix, softmax over v, return E[v]."""
    suffix_tokens = {v: model.tokenizer.encode(f" {v}", add_special_tokens=False)
                     for v in values}
    all_means = []
    with steered_context(direction, strength):
        for prompt in prompts:
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            logp_per_v = {}
            for v in values:
                suf = suffix_tokens[v]
                full_ids = prompt_ids + suf
                tokens = torch.tensor([full_ids], device=device)
                with torch.no_grad():
                    logits = model(tokens)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                start = len(prompt_ids) - 1
                total = sum(log_probs[start + i, tid].item()
                            for i, tid in enumerate(suf))
                logp_per_v[v] = total
            logps = np.array([logp_per_v[v] for v in values])
            logps -= logps.max()
            probs = np.exp(logps)
            probs /= probs.sum()
            all_means.append(float(np.sum(np.array(values) * probs)))
    return float(np.mean(all_means))


def affect_dv(direction, strength, prompts=AFFECT_PROMPTS,
              pos=POS_WORDS, neg=NEG_WORDS):
    pos_ids = []
    neg_ids = []
    for w in pos:
        ids = model.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            pos_ids.append(ids[0])
    for w in neg:
        ids = model.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            neg_ids.append(ids[0])
    diffs = []
    with steered_context(direction, strength):
        for p in prompts:
            lp = next_token_logprobs(p)
            pos_mass = torch.logsumexp(lp[pos_ids], dim=0).item()
            neg_mass = torch.logsumexp(lp[neg_ids], dim=0).item()
            diffs.append(pos_mass - neg_mass)
    return float(np.mean(diffs))


# ============================================================================
# Run identical sweep to exp111
# ============================================================================

STRENGTHS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]

results = {"up": {"fd": [], "eh": [], "af": [], "eq": []},
           "rand": {"fd": [], "eh": [], "af": [], "eq": []}}

print("\n" + "=" * 72)
print("PART A — DOSE-RESPONSE (Lakoff anchors), L=12")
print("=" * 72)

for label, direction in [("up", up_unit), ("rand", rand_unit)]:
    print(f"\n--- direction = {label} ---")
    for s in STRENGTHS:
        fd = first_digit_dv(direction, s)
        eh = expected_value_battery(direction, s, HEIGHT_PROMPTS, HEIGHT_VALUES)
        af = affect_dv(direction, s)
        eq = expected_value_battery(direction, s, QUANTITY_PROMPTS, QUANTITY_VALUES)
        results[label]["fd"].append(fd)
        results[label]["eh"].append(eh)
        results[label]["af"].append(af)
        results[label]["eq"].append(eq)
        print(f"  s={s:>+4}  fd={fd:+.3f}  E[h]={eh:6.2f}  af={af:+.3f}  E[q]={eq:5.2f}")


# ============================================================================
# Direct exp111 vs exp112 comparison
# ============================================================================

# Reload exp111's numbers from disk
exp111 = np.load("/Users/macn/Documents/embeddingexp/exp111_results.npz")

print("\n" + "=" * 72)
print("EXP111 vs EXP112 — same DVs, anchor-set ablation")
print("=" * 72)

i_base = STRENGTHS.index(0)

def delta(arr):
    return [v - arr[i_base] for v in arr]

print(f"\nE[height] cm — Δ from baseline at each strength")
print(f"  {'s':>4}  {'exp111 (hand)':>14}  {'exp112 (Lakoff)':>16}")
for i, s in enumerate(STRENGTHS):
    d111 = exp111["up_expected_height"][i] - exp111["up_expected_height"][i_base]
    d112 = results["up"]["eh"][i] - results["up"]["eh"][i_base]
    print(f"  {s:>+4}  {d111:>+14.3f}  {d112:>+16.3f}")

print(f"\naffect (pos-neg) — Δ from baseline")
print(f"  {'s':>4}  {'exp111 (hand)':>14}  {'exp112 (Lakoff)':>16}")
for i, s in enumerate(STRENGTHS):
    d111 = exp111["up_affect"][i] - exp111["up_affect"][i_base]
    d112 = results["up"]["af"][i] - results["up"]["af"][i_base]
    print(f"  {s:>+4}  {d111:>+14.3f}  {d112:>+16.3f}")

print(f"\nE[quantity] — Δ from baseline")
print(f"  {'s':>4}  {'exp111 (hand)':>14}  {'exp112 (Lakoff)':>16}")
for i, s in enumerate(STRENGTHS):
    d111 = exp111["up_expected_quantity"][i] - exp111["up_expected_quantity"][i_base]
    d112 = results["up"]["eq"][i] - results["up"]["eq"][i_base]
    print(f"  {s:>+4}  {d111:>+14.3f}  {d112:>+16.3f}")


# ============================================================================
# Save
# ============================================================================

np.savez(
    "/Users/macn/Documents/embeddingexp/exp112_results.npz",
    strengths=np.array(STRENGTHS),
    up_first_digit=np.array(results["up"]["fd"]),
    up_expected_height=np.array(results["up"]["eh"]),
    up_affect=np.array(results["up"]["af"]),
    up_expected_quantity=np.array(results["up"]["eq"]),
    rand_first_digit=np.array(results["rand"]["fd"]),
    rand_expected_height=np.array(results["rand"]["eh"]),
    rand_affect=np.array(results["rand"]["af"]),
    rand_expected_quantity=np.array(results["rand"]["eq"]),
    up_raw_norm=up_norm,
    cos_lakoff_vs_handpicked=cos_lakoff_handpicked,
    layer=LAYER,
)

print("\n" + "=" * 72)
print(f"Direction cosine vs exp111: {cos_lakoff_handpicked:+.3f}")
print("Saved exp112_results.npz")
print("=" * 72)
