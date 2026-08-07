"""
exp112_diag.py — verify the cos=-0.99 between exp111 and exp112 UP directions.

Multiple independent rebuilds, plus check shared words.
"""
import torch
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 1.4B...")
model = HookedTransformer.from_pretrained("pythia-1.4b", device=device)
model.eval()
LAYER = 12
HOOK = f"blocks.{LAYER}.hook_resid_post"


def res(word):
    toks = model.to_tokens(word)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    return cache[HOOK][0, -1, :].clone()


EXP111_UP = ["high", "top", "rise", "ceiling", "above",
             "peak", "ascend", "climb", "upward", "overhead"]
EXP111_DOWN = ["low", "bottom", "fall", "floor", "below",
               "valley", "descend", "drop", "downward", "underneath"]

LAKOFF_UP = ["above", "ascend", "climb", "high", "higher", "lift", "over",
             "raise", "rise", "rising", "rose", "top", "up", "upward"]
LAKOFF_DOWN = ["below", "bottom", "descend", "down", "downward", "drop",
               "fall", "falling", "fell", "low", "lower", "sink", "under"]

# Shared anchors (intersect)
shared_up = set(EXP111_UP) & set(LAKOFF_UP)
shared_down = set(EXP111_DOWN) & set(LAKOFF_DOWN)
print(f"\nShared UP: {sorted(shared_up)}")    # high, top, rise, above, ascend, climb, upward
print(f"Shared DOWN: {sorted(shared_down)}")  # low, bottom, fall, below, descend, drop, downward

# Build the 4 directions
def dir_from(up_words, down_words):
    u = torch.stack([res(w) for w in up_words]).mean(0)
    d = torch.stack([res(w) for w in down_words]).mean(0)
    raw = u - d
    return raw / raw.norm(), raw.norm().item()

d_111, n_111 = dir_from(EXP111_UP, EXP111_DOWN)
d_lakoff, n_lakoff = dir_from(LAKOFF_UP, LAKOFF_DOWN)
d_shared, n_shared = dir_from(sorted(shared_up), sorted(shared_down))

# Only-in-exp111 (the unique anchors)
only_111_up = set(EXP111_UP) - set(LAKOFF_UP)
only_111_dn = set(EXP111_DOWN) - set(LAKOFF_DOWN)
only_lk_up = set(LAKOFF_UP) - set(EXP111_UP)
only_lk_dn = set(LAKOFF_DOWN) - set(EXP111_DOWN)
print(f"\nUnique to exp111 UP: {sorted(only_111_up)}")
print(f"Unique to exp111 DOWN: {sorted(only_111_dn)}")
print(f"Unique to Lakoff UP: {sorted(only_lk_up)}")
print(f"Unique to Lakoff DOWN: {sorted(only_lk_dn)}")

d_only111, _ = dir_from(sorted(only_111_up), sorted(only_111_dn))
d_onlylk, _ = dir_from(sorted(only_lk_up), sorted(only_lk_dn))

print(f"\nDirection norms:")
print(f"  exp111: {n_111:.2f}")
print(f"  Lakoff: {n_lakoff:.2f}")
print(f"  shared-only: {n_shared:.2f}")

print(f"\nCosines:")
print(f"  cos(exp111, Lakoff)             = {(d_111 @ d_lakoff).item():+.4f}")
print(f"  cos(exp111, shared-only)        = {(d_111 @ d_shared).item():+.4f}")
print(f"  cos(Lakoff, shared-only)        = {(d_lakoff @ d_shared).item():+.4f}")
print(f"  cos(exp111, exp111-unique)      = {(d_111 @ d_only111).item():+.4f}")
print(f"  cos(Lakoff, Lakoff-unique)      = {(d_lakoff @ d_onlylk).item():+.4f}")
print(f"  cos(exp111-unique, Lakoff-uniq) = {(d_only111 @ d_onlylk).item():+.4f}")
print(f"  cos(shared, exp111-unique)      = {(d_shared @ d_only111).item():+.4f}")
print(f"  cos(shared, Lakoff-unique)      = {(d_shared @ d_onlylk).item():+.4f}")

# Pairwise activation similarities between the 'opposite' words to sanity check
print(f"\nPair cosines (single-word activations at L12):")
for a, b in [("up", "down"), ("rise", "fall"), ("high", "low"),
             ("peak", "valley"), ("ceiling", "floor"),
             ("up", "high"), ("down", "low"),
             ("up", "rise"), ("down", "fall"),
             ("up", "ceiling"), ("up", "peak"),
             ("down", "floor"), ("down", "valley")]:
    va, vb = res(a), res(b)
    cos = (va @ vb / (va.norm() * vb.norm())).item()
    print(f"  cos({a!r:>10}, {b!r:>10}) = {cos:+.3f}")
