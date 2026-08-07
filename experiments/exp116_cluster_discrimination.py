"""
exp116_cluster_discrimination.py — discriminate the cluster reading.

After exp114 (clean affect shift) + exp115 (qualitative completions showing
height + weight + status all co-shifting under cleaned UP steering),
we hypothesise that the freq-stripped UP direction is Lakoff's
vertical-elevation superordinate metaphor cluster, not just verticality.

This experiment tests that hypothesis with parallel-structured DVs:

  VERTICAL UP        Patient height (cm)         hypothesis: ↑ with +s
  VERTICAL DOWN      Pool depth (cm)              ↑ if "magnitude"; ↓ if "verticality-with-sign"
  HORIZONTAL         Door width (cm)              ↓ flat if vertical-specific; ↑ if "any number"
  PURE QUANTITY      Number of widgets in box     ↑ if MORE-IS-UP cluster member
  STATUS             Occupation mass contrast     ↑ if HIGH-STATUS-IS-UP cluster member
  AFFECT (re-run)    Mood mass contrast           ↑ (already shown clean in exp114)

Discrimination logic:
  - ALL go up at +s → "make the model say bigger thing" (number inflation
    or general magnitude inflation), not specifically verticality cluster.
  - patient_height ↑ + pool_depth ↓ + door_width ~flat + status ↑ + affect ↑
    → verticality WITH sign + status cluster member (the dream outcome).
  - patient_height ↑ + pool_depth ↑ + door_width flat + status ↑ + affect ↑
    → MORE-IS-UP cluster (Lakoff metaphor cluster without bipolar sign).
  - patient_height ↑ + status flat → I was confabulating status from
    completions; cluster reading dies.

Same conditions as exp114: u111_clean, ulak_clean, freq_only, random.
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
    return raw / raw.norm()


# Anchors
EXP111_UP = ["high", "top", "rise", "ceiling", "above",
             "peak", "ascend", "climb", "upward", "overhead"]
EXP111_DOWN = ["low", "bottom", "fall", "floor", "below",
               "valley", "descend", "drop", "downward", "underneath"]
LAKOFF_UP = ["above", "ascend", "climb", "high", "higher", "lift", "over",
             "raise", "rise", "rising", "rose", "top", "up", "upward"]
LAKOFF_DOWN = ["below", "bottom", "descend", "down", "downward", "drop",
               "fall", "falling", "fell", "low", "lower", "sink", "under"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]

print("\nBuilding directions...")
freq_axis = dir_from(COMMON, RARE)
u111_raw = dir_from(EXP111_UP, EXP111_DOWN)
ulak_raw = dir_from(LAKOFF_UP, LAKOFF_DOWN)


def strip_freq(v):
    v = v - (v @ freq_axis) * freq_axis
    return v / v.norm()


u111_clean = strip_freq(u111_raw)
ulak_clean = strip_freq(ulak_raw)
torch.manual_seed(7)
rand_unit = torch.randn_like(u111_raw)
rand_unit = rand_unit / rand_unit.norm()


# ============================================================================
# Spatial / numeric DVs (parallel structure)
# ============================================================================

# Try medical-chart-format prompts — less Wikipedia-bio-attractor than exp111's
PATIENT_HEIGHT_PROMPTS = [
    "Patient height (cm):",
    "Patient height in cm:",
    "Height (cm):",
]
POOL_DEPTH_PROMPTS = [
    "Pool depth (cm):",
    "Swimming pool depth in cm:",
    "Depth of the pool (cm):",
]
DOOR_WIDTH_PROMPTS = [
    "Door width (cm):",
    "Width of the door (cm):",
    "Doorway width in cm:",
]
WIDGET_QUANTITY_PROMPTS = [
    "Number of widgets in the box:",
    "Number of items on the table:",
    "Count of objects in the room:",
]

# Value ranges chosen to be sensible for each domain
HEIGHT_VALUES = list(range(140, 225, 5))   # cm, adult human range
DEPTH_VALUES = list(range(50, 405, 25))    # cm, shallow paddling to deep diving
WIDTH_VALUES = list(range(60, 205, 10))    # cm, narrow doorway to wide French doors
WIDGET_VALUES = list(range(2, 101, 4))     # 2 to 100 widgets


# ============================================================================
# Status DV — single-token mass contrast
# ============================================================================

STATUS_PROMPTS = [
    "He works as a",
    "She works as a",
    "The man was a",
    "The woman was a",
    "His occupation is",
    "Her job title is",
]
# Niamh: include the canonical high-status archetypes (king/queen) alongside
# professional-class status (CEO/doctor) — much more saliently "high status"
# in language than CEO is.
HIGH_STATUS_CANDIDATES = [
    # Royalty / nobility — the canonical archetypes
    " king", " queen", " prince", " princess", " duke", " duchess",
    " lord", " lady", " emperor", " empress", " monarch", " noble",
    # Professional class
    " CEO", " executive", " director", " doctor", " surgeon", " professor",
    " lawyer", " judge", " general", " president", " manager", " banker",
    " consultant", " scientist", " engineer", " architect",
]
LOW_STATUS_CANDIDATES = [
    # Archetypal low-status (paired with royalty side)
    " peasant", " serf", " servant", " slave", " beggar", " commoner",
    # Service / manual labour
    " janitor", " cleaner", " labourer", " dishwasher", " porter",
    " gardener", " maid", " waiter", " cashier", " bartender", " cook",
]


# ============================================================================
# Affect DV — re-use exp114
# ============================================================================

AFFECT_PROMPTS = [
    "When I think about the future, I feel",
    "My current mood is",
    "Today I am feeling",
    "The state of my mind right now is",
]
POS_CANDIDATES = [" hopeful", " optimistic", " excited", " happy", " elated", " uplifted"]
NEG_CANDIDATES = [" anxious", " sad", " worried", " depressed", " hopeless", " low"]


def single_token_ids(candidates):
    ids = []
    for w in candidates:
        toks = model.tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return ids


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


def expected_value(direction, strength, prompts, values):
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


def mass_contrast(direction, strength, prompts, pos_ids, neg_ids):
    diffs = []
    with steered_context(direction, strength):
        for p in prompts:
            lp = next_token_logprobs(p)
            pos = torch.logsumexp(lp[pos_ids], 0).item()
            neg = torch.logsumexp(lp[neg_ids], 0).item()
            diffs.append(pos - neg)
    return float(np.mean(diffs))


# Resolve single-token ID lists
HIGH_STATUS_IDS = single_token_ids(HIGH_STATUS_CANDIDATES)
LOW_STATUS_IDS = single_token_ids(LOW_STATUS_CANDIDATES)
POS_IDS = single_token_ids(POS_CANDIDATES)
NEG_IDS = single_token_ids(NEG_CANDIDATES)

print(f"\nSingle-token survival:")
print(f"  HIGH STATUS: {len(HIGH_STATUS_IDS)}/{len(HIGH_STATUS_CANDIDATES)}")
print(f"  LOW STATUS:  {len(LOW_STATUS_IDS)}/{len(LOW_STATUS_CANDIDATES)}")
print(f"  POS AFFECT:  {len(POS_IDS)}/{len(POS_CANDIDATES)}")
print(f"  NEG AFFECT:  {len(NEG_IDS)}/{len(NEG_CANDIDATES)}")


# ============================================================================
# Sweep
# ============================================================================

STRENGTHS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]

CONDITIONS = [
    ("u111_clean", u111_clean),
    ("ulak_clean", ulak_clean),
    ("freq_only",  freq_axis),
    ("rand",       rand_unit),
]

DVS = [
    ("patient_height_cm",  "ev", PATIENT_HEIGHT_PROMPTS,  HEIGHT_VALUES),
    ("pool_depth_cm",      "ev", POOL_DEPTH_PROMPTS,      DEPTH_VALUES),
    ("door_width_cm",      "ev", DOOR_WIDTH_PROMPTS,      WIDTH_VALUES),
    ("widget_quantity",    "ev", WIDGET_QUANTITY_PROMPTS, WIDGET_VALUES),
    ("status",             "mc", STATUS_PROMPTS,          (HIGH_STATUS_IDS, LOW_STATUS_IDS)),
    ("affect",             "mc", AFFECT_PROMPTS,          (POS_IDS, NEG_IDS)),
]

results = {cname: {dname: [] for dname, *_ in DVS} for cname, _ in CONDITIONS}

print("\n" + "=" * 76)
print("PART A — sweep on all DVs × conditions")
print("=" * 76)

for cname, d in CONDITIONS:
    print(f"\n--- {cname} ---")
    print(f"  {'s':>4}  " + "  ".join(f"{dn[:12]:>12}" for dn, *_ in DVS))
    for s in STRENGTHS:
        row = {}
        for dname, kind, prompts, payload in DVS:
            if kind == "ev":
                val = expected_value(d, s, prompts, payload)
            else:  # mc
                pos_ids, neg_ids = payload
                val = mass_contrast(d, s, prompts, pos_ids, neg_ids)
            results[cname][dname].append(val)
            row[dname] = val
        print(f"  {s:>+4}  " + "  ".join(f"{row[dn]:>12.3f}" for dn, *_ in DVS))


# ============================================================================
# Δ-from-baseline comparison tables
# ============================================================================

i0 = STRENGTHS.index(0)

print("\n" + "=" * 76)
print("Δ FROM BASELINE — per DV, all conditions side by side")
print("=" * 76)

for dname, *_ in DVS:
    print(f"\n{dname} — Δ from baseline")
    print(f"  {'s':>4}  " + "  ".join(f"{cn:>12}" for cn, _ in CONDITIONS))
    for i, s in enumerate(STRENGTHS):
        deltas = [results[cn][dname][i] - results[cn][dname][i0] for cn, _ in CONDITIONS]
        print(f"  {s:>+4}  " + "  ".join(f"{d:>+12.3f}" for d in deltas))


# ============================================================================
# Cluster discrimination summary at s=+12
# ============================================================================

i_test = STRENGTHS.index(12)
print("\n" + "=" * 76)
print(f"CLUSTER DISCRIMINATION at s=+12 (Δ from baseline)")
print("=" * 76)
print(f"\n  {'DV':>20}  " + "  ".join(f"{cn:>12}" for cn, _ in CONDITIONS))
for dname, *_ in DVS:
    deltas = [results[cn][dname][i_test] - results[cn][dname][i0] for cn, _ in CONDITIONS]
    print(f"  {dname:>20}  " + "  ".join(f"{d:>+12.3f}" for d in deltas))


# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp116_results.npz",
    strengths=np.array(STRENGTHS),
    **{f"{cn}__{dn}": np.array(results[cn][dn])
       for cn, _ in CONDITIONS for dn, *_ in DVS}
)

with open("/Users/macn/Documents/embeddingexp/exp116_config.json", "w") as f:
    json.dump({
        "model": "pythia-1.4b",
        "layer": LAYER,
        "strengths": STRENGTHS,
        "conditions": [c[0] for c in CONDITIONS],
        "dvs": {dn: {"kind": kind, "prompts": prompts} for dn, kind, prompts, _ in DVS},
        "high_status_words": HIGH_STATUS_CANDIDATES,
        "low_status_words": LOW_STATUS_CANDIDATES,
        "pos_affect_words": POS_CANDIDATES,
        "neg_affect_words": NEG_CANDIDATES,
        "n_high_status_singletok": len(HIGH_STATUS_IDS),
        "n_low_status_singletok": len(LOW_STATUS_IDS),
        "n_pos_singletok": len(POS_IDS),
        "n_neg_singletok": len(NEG_IDS),
    }, f, indent=2)

print("\nSaved exp116_results.npz + exp116_config.json")
