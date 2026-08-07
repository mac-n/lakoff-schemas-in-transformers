"""
exp104_operations_plural.py

The 'primitives as OPERATIONS' program, v1 (Niamh).
Reframe: a primitive is not a direction (PCA/contrast view, which just deflated)
but an OPERATION T: embedding -> embedding. Test the cleanest invertible operation
-- singular -> plural -- as a first-order (linear) map, with today's discipline:

  1. Fit in BOTH raw (mean-centered) and FREQUENCY-STRIPPED (PC1 removed) space.
     Plural forms are systematically rarer than singulars, so a 'plural operation'
     could be faked by a frequency shift. If it dies when PC1 is stripped, it was
     frequency. (Same control that killed HARDNESS.)
  2. The fit is free -- only generalization + beating nulls count. Predictors:
        identity      (do nothing; STRONG baseline: cat ~ cats already)
        translation   (a + t, TransE; t = mean(b-a) over train)
        linear map    (b ~ a W, least squares; n_train >> d so well-posed)
        procrustes    (b ~ a R, R orthogonal -- the rotation/SO(n) version)
  3. Demonstrate the Linzen (2016) 'exclude the inputs' artifact directly:
     report retrieval accuracy INCLUDING vs EXCLUDING the source word, and how
     often the nearest neighbour to the prediction is just the source itself.

Train: large (w, w+'s') pairs harvested from vocab (so the linear map is
well-posed). Test: the clean Google-analogy gram8-plural noun pairs (held out).
"""
import numpy as np
import gensim.downloader as api

rng = np.random.default_rng(0)

def unit_rows(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300")
vocab = set(kv.index_to_key)
mu = kv.vectors.mean(axis=0)

# global PC1 (= frequency in GloVe) for the stripping control
S = kv.vectors[:100000] - mu
evals, evecs = np.linalg.eigh((S.T @ S) / S.shape[0])
pc1 = evecs[:, -1]; pc1 = pc1 / np.linalg.norm(pc1)

# ---- clean TEST pairs: Google analogy gram8-plural (true noun plurals) ----
test_pairs, sect = [], None
for line in open("norms/questions-words.txt"):
    if line.startswith(":"):
        sect = line[1:].strip(); continue
    if sect == "gram8-plural":
        a, b, c, d = line.lower().split()
        for s, p in [(a, b), (c, d)]:
            if s in vocab and p in vocab:
                test_pairs.append((s, p))
test_pairs = sorted(set(test_pairs))
test_words = set(w for pr in test_pairs for w in pr)
print(f"  clean test pairs (gram8-plural): {len(test_pairs)}")

# ---- large TRAIN pairs: (w, w+'s') harvested from vocab, test words excluded ----
train_pairs = []
for w in kv.index_to_key[:80000]:
    if w.isalpha() and len(w) >= 3 and not w.endswith("s"):
        p = w + "s"
        if p in vocab and w not in test_words and p not in test_words:
            train_pairs.append((w, p))
print(f"  train pairs (w -> w+'s'): {len(train_pairs)}   (d=300, so n_train >> d: {len(train_pairs) > 300})")

def vec(space, w):
    v = kv[w] - mu
    if space == "pc1_stripped":
        v = v - np.dot(v, pc1) * pc1
    return v

def matrix(space, pairs):
    A = np.stack([vec(space, a) for a, b in pairs])
    B = np.stack([vec(space, b) for a, b in pairs])
    return A, B

for space in ["raw (mean-centered)", "pc1_stripped (frequency removed)"]:
    sp = "pc1_stripped" if "stripped" in space else "raw"
    print("\n" + "=" * 76)
    print(f"SPACE: {space}")
    print("=" * 76)
    Atr, Btr = matrix(sp, train_pairs)
    Ate, Bte = matrix(sp, test_pairs)

    t = (Btr - Atr).mean(axis=0)                      # translation (TransE)
    W, *_ = np.linalg.lstsq(Atr, Btr, rcond=None)     # linear map  b ~ a W
    U, _, Vt = np.linalg.svd(Atr.T @ Btr)             # orthogonal Procrustes
    R = U @ Vt

    preds = {
        "identity":    Ate,
        "translation": Ate + t,
        "linear map":  Ate @ W,
        "procrustes":  Ate @ R,
    }

    # retrieval pool: test singulars + plurals + random distractors
    pool_words = list(test_words) + [w for w in rng.choice(kv.index_to_key[:50000], 4000, replace=False)
                                     if w not in test_words]
    pool_words = list(dict.fromkeys(pool_words))
    Pool = unit_rows(np.stack([vec(sp, w) for w in pool_words]))
    idx = {w: i for i, w in enumerate(pool_words)}

    print(f"\n  {'predictor':<13}{'mean cos':>10}{'top1 incl':>11}{'top1 excl':>11}{'NN=source':>11}")
    print("  " + "-" * 55)
    Bte_n = unit_rows(Bte)
    for name, P in preds.items():
        Pn = unit_rows(P)
        mean_cos = float(np.mean(np.sum(Pn * Bte_n, axis=1)))
        sims = Pn @ Pool.T                            # (n_test, n_pool)
        incl = excl = src = 0
        for k, (a, b) in enumerate(test_pairs):
            row = sims[k].copy()
            nn_incl = pool_words[int(np.argmax(row))]
            if nn_incl == b: incl += 1
            if nn_incl == a: src += 1
            row[idx[a]] = -2.0                        # exclude the source word
            nn_excl = pool_words[int(np.argmax(row))]
            if nn_excl == b: excl += 1
        n = len(test_pairs)
        print(f"  {name:<13}{mean_cos:>10.3f}{incl/n:>10.0%}{excl/n:>11.0%}{src/n:>11.0%}")

print("""
READING
  - identity is a strong baseline (singular ~ plural already). The operation is
    'real and linear' only if translation / linear map beat identity on mean-cos
    AND on EXCLUSIVE retrieval (the honest metric).
  - 'NN=source' shows the Linzen artifact: when it's high, inclusive top1 is
    flattered by the prediction just sitting on the input word.
  - If the numbers collapse in pc1_stripped space, 'plural' was a frequency shift.
  - If linear map ~ translation, plural is just an offset (TransE suffices);
    if linear map > translation, the operation needs matrix structure.
""")
