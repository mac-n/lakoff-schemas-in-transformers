"""
exp169_affect_freegen_cleaned.py — free generation on AFFECT prompts under
the frequency-stripped UP steering vectors.

Motivation (blog revision, 2026-07-02): exp111's free-gen samples were
produced with the RAW UP vector (~96% frequency axis) and are noisy
seed-to-seed; the blog now declines to quote them. exp115 ran cleaned-vector
free-gen but only on HEIGHT prompts. This experiment fills the gap: cleaned
vectors, affect prompts, every sample published unselected.

PRE-COMMITTED PREDICTION (written before running):
  Based on exp114's quantitative result (ulak_clean shifts affect by
  +0.677 nats at s=+12; freq_only +0.011; rand +0.005), we predict:
  - u111_clean / ulak_clean at s=+12 produce completions that skew
    positive-valence relative to the same seeds at s=-12.
  - freq_only and rand show no consistent valence ordering.
  - Signal is expected to be visible as a TENDENCY across the full sample
    set, not in every individual completion (temperature 0.7 sampling is
    noisy; exp111 showed single samples can contradict the mean).
  Reporting rule: ALL completions are saved to JSON and none are excluded.
  Any blog quotation must show all seeds for a given (prompt, strength).

Protocol identical to exp115 except for the prompt battery.
"""

import json
from contextlib import nullcontext
import torch
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 1.4B...")
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

freq_axis = dir_from(COMMON, RARE)
u111 = dir_from(EXP111_UP, EXP111_DOWN)
ulak = dir_from(LAKOFF_UP, LAKOFF_DOWN)

def strip_freq(v):
    v = v - (v @ freq_axis) * freq_axis
    return v / v.norm()

u111_clean = strip_freq(u111)
ulak_clean = strip_freq(ulak)
torch.manual_seed(7)
rand_unit = torch.randn_like(u111)
rand_unit = rand_unit / rand_unit.norm()

print(f"cos(u111_clean, ulak_clean) = {(u111_clean @ ulak_clean).item():+.3f}"
      "  (exp114 reference: +0.571)")


def make_hook(d, s):
    def hook_fn(resid, hook):
        return resid + s * d
    return hook_fn

def steered_context(d, s):
    if s == 0.0:
        return nullcontext()
    return model.hooks(fwd_hooks=[(HOOK, make_hook(d, s))])


def gen(direction, strength, prompt, seed, n=40):
    torch.manual_seed(seed)
    with steered_context(direction, strength):
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            out = model.generate(toks, max_new_tokens=n, temperature=0.7,
                                 do_sample=True, verbose=False)
    return model.to_string(out[0])


# Affect prompts — same battery as exp111/exp114's affect DV.
PROMPTS = [
    "When I think about the future, I feel",
    "My current mood is",
    "Today I am feeling",
    "The state of my mind right now is",
]
STRENGTHS = [-12, -8, 0, 8, 12]
SEEDS = [42, 7, 99]

CONDITIONS = [
    ("u111_clean (freq-stripped exp111)", u111_clean),
    ("ulak_clean (freq-stripped Lakoff)", ulak_clean),
    ("freq_only (pure frequency)",        freq_axis),
    ("rand",                              rand_unit),
]

all_records = []
for prompt in PROMPTS:
    print("\n" + "#" * 78)
    print(f"PROMPT: '{prompt}'")
    print("#" * 78)
    for cname, d in CONDITIONS:
        print(f"\n--- direction: {cname} ---")
        for s in STRENGTHS:
            print(f"  [s={s:+}]")
            for seed in SEEDS:
                text = gen(d, s, prompt, seed)
                comp = text[len(prompt):] if text.startswith(prompt) else text
                all_records.append({
                    "prompt": prompt,
                    "condition": cname,
                    "strength": s,
                    "seed": seed,
                    "completion_raw": comp,
                    "completion_stripped": comp.strip(),
                })
                comp_disp = comp.strip().replace("\n", " ⏎ ")
                if len(comp_disp) > 200:
                    comp_disp = comp_disp[:200] + "..."
                print(f"     · seed={seed}: {comp_disp}")

out_path = "/Users/macn/Documents/embeddingexp/exp169_freegen.json"
with open(out_path, "w") as f:
    json.dump({
        "config": {
            "model": "pythia-1.4b",
            "layer": LAYER,
            "prompts": PROMPTS,
            "strengths": STRENGTHS,
            "seeds": SEEDS,
            "conditions": [c[0] for c in CONDITIONS],
            "temperature": 0.7,
            "max_new_tokens": 40,
            "exp111_up": EXP111_UP, "exp111_down": EXP111_DOWN,
            "lakoff_up": LAKOFF_UP, "lakoff_down": LAKOFF_DOWN,
            "common": COMMON, "rare": RARE,
            "reporting_rule": "all completions saved, none excluded; "
                              "blog quotes must show all seeds for a given "
                              "(prompt, strength)",
        },
        "records": all_records,
    }, f, indent=2)
print(f"\nSaved {len(all_records)} records to {out_path}")
