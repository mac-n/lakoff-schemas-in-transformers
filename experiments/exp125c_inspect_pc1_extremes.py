"""
exp125c_inspect_pc1_extremes.py — what does PC1 at L12 actually encode?

Recompute the L12 residuals for the same 530 balanced-band words from
exp125b, compute PC1, sort words by PC1 projection, show top/bottom 30.
"""

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

device = "mps"
print("Loading Pythia 410M...")
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
model.eval()

L = 12
HOOK = f"blocks.{L}.hook_resid_post"

# Same word list as exp125b
brys = pd.read_csv("/Users/macn/Documents/embeddingexp/norms/Brysbaert_concreteness.txt",
                   sep="\t")
brys = brys[brys["SUBTLEX"] > 0].copy()
brys["word"] = brys["Word"].astype(str).str.lower()
brys = brys[brys["word"].apply(lambda w: isinstance(w, str) and w.isalpha())]
brys = brys.sort_values("SUBTLEX", ascending=False).reset_index(drop=True)
brys["rank"] = np.arange(1, len(brys) + 1)


def n_toks(w):
    return len(model.tokenizer.encode(w, add_special_tokens=False))


BANDS = [
    ("B1_spike",         1, 10),
    ("B2_top_function", 11, 50),
    ("B3_top_content", 51, 300),
    ("B4_high",       301, 1000),
    ("B5_mid",       1001, 3000),
    ("B6_mid_low",   3001, 7000),
    ("B7_low",       7001, 15000),
    ("B8_rare",     15001, 30000),
]

TARGET = 80
print("\nRebuilding 530-word balanced sample...")
band_of = {}
all_words = []
for name, lo, hi in BANDS:
    candidates = brys[(brys["rank"] >= lo) & (brys["rank"] <= hi)]["word"].tolist()
    np.random.seed(hash(name) % 2**32)
    np.random.shuffle(candidates)
    kept = []
    for w in candidates:
        if n_toks(w) != 1:
            continue
        if w in band_of:
            continue
        kept.append(w)
        band_of[w] = name
        if len(kept) >= TARGET:
            break
    all_words.extend(kept)
    print(f"  {name}: {len(kept)} words")

print(f"  Total: {len(all_words)}")


print(f"\nExtracting residuals at L{L}...")
residuals = []
for i, w in enumerate(all_words):
    toks = model.to_tokens(w)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    r = cache[HOOK][0, -1, :].cpu().numpy()
    residuals.append(r)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(all_words)}")

R = np.array(residuals)


# PCA
mean = R.mean(axis=0)
Rc = R - mean
U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
pc1 = Vt[0]
proj = Rc @ pc1
var_explained = (S**2) / (S**2).sum()
print(f"\nPC1 explains {var_explained[0]:.1%} of variance at L{L}")
print(f"PC2 explains {var_explained[1]:.1%}")
print(f"PC3 explains {var_explained[2]:.1%}")
print(f"PC4 explains {var_explained[3]:.1%}")
print(f"PC5 explains {var_explained[4]:.1%}")


# Sort by PC1 projection
order = np.argsort(proj)
words_arr = np.array(all_words)

print("\n" + "=" * 70)
print(f"TOP 30 highest PC1 projection at L{L}:")
print("=" * 70)
for i in order[::-1][:30]:
    print(f"  {proj[i]:>+8.3f}  {words_arr[i]:<25}  ({band_of[words_arr[i]]})")

print("\n" + "=" * 70)
print(f"BOTTOM 30 lowest PC1 projection at L{L}:")
print("=" * 70)
for i in order[:30]:
    print(f"  {proj[i]:>+8.3f}  {words_arr[i]:<25}  ({band_of[words_arr[i]]})")

# Also: middle 30 (around median PC1)
mid = len(order) // 2
print("\n" + "=" * 70)
print(f"MIDDLE 30 (around median PC1):")
print("=" * 70)
for i in order[mid-15:mid+15]:
    print(f"  {proj[i]:>+8.3f}  {words_arr[i]:<25}  ({band_of[words_arr[i]]})")

# Also: let's see PC2 quickly since PC1 might miss the frequency signal that
# lives in another component
pc2 = Vt[1]
proj2 = Rc @ pc2
order2 = np.argsort(proj2)

print("\n" + "=" * 70)
print(f"TOP 30 PC2 projection at L{L}:")
print("=" * 70)
for i in order2[::-1][:30]:
    print(f"  {proj2[i]:>+8.3f}  {words_arr[i]:<25}  ({band_of[words_arr[i]]})")

print("\n" + "=" * 70)
print(f"BOTTOM 30 PC2 projection at L{L}:")
print("=" * 70)
for i in order2[:30]:
    print(f"  {proj2[i]:>+8.3f}  {words_arr[i]:<25}  ({band_of[words_arr[i]]})")
