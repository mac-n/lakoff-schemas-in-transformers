"""
exp105_composition_commutativity.py  (v2 — Part 1 refit in a well-conditioned subspace)

The group-structure payoff of the operations program (Niamh). Two parts:

PART 1 — operation algebra (large suffix-harvested data).
  v1 fit a full 300x300 map and it was wildly ill-conditioned (inverse error ~1e4,
  commutator a meaningless 0.000). Embedding dims have hugely unequal variance, so
  the unregularized map blows up in low-variance directions. FIX: fit the operation
  in the top-K semantic subspace (frequency PC removed, top-100 PCs kept), where the
  map is well-conditioned. Then the algebra means something:
    - INVERSE (group elements have inverses): W_fwd @ W_bwd ~ I_K ?
    - COMMUTATIVITY: ||EiEj-EjEi|| / (||Ei||*||Ej||) on deviations E=W-I, vs a
      random-matrix null. Commute => shared eigenbasis => disentanglement for free.

PART 2 — composition vs ground truth (gender x number grid, leave-one-out).
  Predict the doubly-inflected word (king -> queens) via gender+number translations
  and check it RETRIEVES the real composed word. Pure-translation orders are
  identical (commutativity automatic); the non-trivial commutativity is PART 1.
"""
import numpy as np
import gensim.downloader as api

rng = np.random.default_rng(0)
def unit_rows(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300")
vocab = set(kv.index_to_key)
mu = kv.vectors.mean(axis=0)
S = kv.vectors[:100000] - mu
ev, evec = np.linalg.eigh((S.T @ S) / S.shape[0])
pc1 = evec[:, -1]; pc1 = pc1 / np.linalg.norm(pc1)
d = 300
K = 100
P = evec[:, -(K + 1):-1]          # PCs 2..K+1 : frequency PC1 dropped, top-K semantic kept

def vec(w):                        # full-space, mean-centered + PC1-stripped (Part 2)
    v = kv[w] - mu
    return v - np.dot(v, pc1) * pc1

def proj(w):                       # K-dim semantic-subspace coords (Part 1)
    return P.T @ (kv[w] - mu)

def harvest(suffix):
    return [(w, w + suffix) for w in kv.index_to_key[:80000]
            if w.isalpha() and len(w) >= 3 and not w.endswith("s") and (w + suffix) in vocab]

OPS = {"+s": "s", "+ed": "ed", "+ing": "ing"}
print("\n" + "=" * 72)
print(f"PART 1 — operation algebra in top-{K} semantic subspace (frequency removed)")
print("=" * 72)
devs = {}
for name, suf in OPS.items():
    pairs = harvest(suf); rng.shuffle(pairs)
    cut = int(0.85 * len(pairs)); tr, te = pairs[:cut], pairs[cut:]
    A = np.stack([proj(a) for a, b in tr]); B = np.stack([proj(b) for a, b in tr])
    W, *_ = np.linalg.lstsq(A, B, rcond=None)            # K x K, well-posed (n >> K)
    Wb, *_ = np.linalg.lstsq(B, A, rcond=None)           # backward map
    Ate = np.stack([proj(a) for a, b in te]); Bte = unit_rows(np.stack([proj(b) for a, b in te]))
    id_cos = float(np.mean(np.sum(unit_rows(Ate) * Bte, axis=1)))
    map_cos = float(np.mean(np.sum(unit_rows(Ate @ W) * Bte, axis=1)))
    inv_err = np.linalg.norm(W @ Wb - np.eye(K)) / np.sqrt(K)
    devs[name] = W - np.eye(K)
    print(f"  {name:<5} n={len(pairs):>5}  held-out cos: identity {id_cos:.3f} -> map {map_cos:.3f}"
          f"   |  inverse err ||W_f@W_b - I||: {inv_err:.3f}")

print("\n  COMMUTATIVITY of deviations  ||EiEj-EjEi|| / (||Ei||*||Ej||)   (0=commute, ~1=not):")
names = list(OPS)
def relcomm(E1, E2):
    return np.linalg.norm(E1 @ E2 - E2 @ E1) / (np.linalg.norm(E1) * np.linalg.norm(E2) + 1e-12)
print("        " + "".join(f"{n:>8}" for n in names))
for n1 in names:
    print(f"  {n1:<6}" + "".join(f"{relcomm(devs[n1], devs[n2]):>8.3f}" for n2 in names))
scale = np.mean([np.linalg.norm(devs[n]) for n in names]) / K
R1 = rng.standard_normal((K, K)) * scale; R2 = rng.standard_normal((K, K)) * scale
print(f"  null (two random matrices, matched norm): {relcomm(R1, R2):.3f}")

# ---- PART 2 — composition vs ground truth (gender x number) ----
print("\n" + "=" * 72)
print("PART 2 — composition vs ground truth (gender x number, leave-one-out)")
print("=" * 72)
QUADS = [
    ("king","queen","kings","queens"),("prince","princess","princes","princesses"),
    ("actor","actress","actors","actresses"),("waiter","waitress","waiters","waitresses"),
    ("host","hostess","hosts","hostesses"),("god","goddess","gods","goddesses"),
    ("lion","lioness","lions","lionesses"),("heir","heiress","heirs","heiresses"),
    ("emperor","empress","emperors","empresses"),("duke","duchess","dukes","duchesses"),
    ("tiger","tigress","tigers","tigresses"),("master","mistress","masters","mistresses"),
    ("priest","priestess","priests","priestesses"),("man","woman","men","women"),
    ("boy","girl","boys","girls"),("father","mother","fathers","mothers"),
    ("brother","sister","brothers","sisters"),("son","daughter","sons","daughters"),
    ("uncle","aunt","uncles","aunts"),("nephew","niece","nephews","nieces"),
    ("husband","wife","husbands","wives"),("gentleman","lady","gentlemen","ladies"),
    ("monk","nun","monks","nuns"),("wizard","witch","wizards","witches"),
    ("bull","cow","bulls","cows"),("widower","widow","widowers","widows"),
]
QUADS = [q for q in QUADS if all(w in vocab for w in q)]
print(f"  usable quadruples (all 4 in vocab): {len(QUADS)}")
pool_w = sorted({w for q in QUADS for w in q} |
                set(rng.choice(kv.index_to_key[:50000], 3000, replace=False)))
Pool = unit_rows(np.stack([vec(w) for w in pool_w])); pidx = {w: i for i, w in enumerate(pool_w)}

hits_comp = hits_num = hits_gen = 0; cos_comp = []
for i, (ms, fs, mp, fp) in enumerate(QUADS):
    others = [q for j, q in enumerate(QUADS) if j != i]
    t_gender = np.mean([vec(o[1]) - vec(o[0]) for o in others], axis=0)
    t_number = np.mean([vec(o[2]) - vec(o[0]) for o in others], axis=0)
    base = vec(ms)
    pred = base + t_gender + t_number
    cos_comp.append(float(unit_rows(pred[None])[0] @ unit_rows(vec(fp)[None])[0]))
    sims = Pool @ unit_rows(pred[None])[0]
    for w in (ms, fs, mp): sims[pidx[w]] = -2
    if pool_w[int(np.argmax(sims))] == fp: hits_comp += 1
    s_n = Pool @ unit_rows((base + t_number)[None])[0]; s_n[pidx[ms]] = -2
    if pool_w[int(np.argmax(s_n))] == mp: hits_num += 1
    s_g = Pool @ unit_rows((base + t_gender)[None])[0]; s_g[pidx[ms]] = -2
    if pool_w[int(np.argmax(s_g))] == fs: hits_gen += 1

n = len(QUADS)
print(f"\n  single op  number (masc_sg -> masc_pl)  top1: {hits_num/n:.0%}")
print(f"  single op  gender (masc_sg -> fem_sg)   top1: {hits_gen/n:.0%}")
print(f"  COMPOSITION (masc_sg + gender + number -> fem_pl):")
print(f"    mean cosine to true composed word: {np.mean(cos_comp):.3f}")
print(f"    top1 retrieval of the real composed word: {hits_comp/n:.0%}")
print("""
READING
  PART 1: inverse err near 0 => operation is genuinely invertible (a group element).
          commutator near the random null => operations do NOT commute (non-abelian);
          commutator << null => they share structure (toward free disentanglement).
  PART 2: composition retrieval near the single-op rates => the operation algebra
          CORRESPONDS to semantics (A∘B lands where the composed concept is) — the
          thing PCA cannot give. Degradation => errors compound under stacking.
""")
