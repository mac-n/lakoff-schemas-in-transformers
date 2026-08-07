"""
exp102_clean_contrasts_freq_stripped.py

Niamh's clean redesign: drop the multi-anchor constructions (which introduced an
anchor-frequency imbalance, exp101) and use simple TWO-WORD contrast vectors.
Run on GloVe and fastText, with frequency stripped (All-But-The-Top, k=3), and
compare across substrates.

Seven contrasts (direction = the 'to' word is +):
    soft->hard, warm->cold, up->down, forward->back, past->future,
    chaos->order, darkness->light

Per substrate:
  - identify frequency-laden top PCs (report spearman(PC, freq-rank)), strip top 3
  - each axis's frequency loading BEFORE stripping (was it confounded?)
  - 7x7 pairwise cosines AFTER stripping (are these independent axes?)
  - pole words AFTER stripping (semantic sanity)
  - coverage of the exp90 cognitive sample AFTER stripping (honest, freq-free)
Cross-substrate:
  - for each axis, spearman of word-projections GloVe-vs-fastText over shared vocab
    (high => the axis is substrate-general, not a quirk of one space)
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr
import gc

CONTRASTS = [  # (name, source_word, target_word) ; axis = target - source
    ("soft->hard",       "soft",     "hard"),
    ("warm->cold",       "warm",     "cold"),
    ("up->down",         "up",       "down"),
    ("forward->back",    "forward",  "back"),
    ("past->future",     "past",     "future"),
    ("chaos->order",     "chaos",    "order"),
    ("darkness->light",  "darkness", "light"),
]
NAMES = [c[0] for c in CONTRASTS]

COG = ["happiness","sadness","anger","envy","jealousy","pride","humility","contentment","longing",
    "delight","melancholy","rage","elation","despair","serenity","anguish","disgust","ambition",
    "determination","resignation","willpower","discipline","procrastination","perseverance","complacency",
    "vigilance","diligence","negligence","carelessness","trust","betrayal","friendship","enmity","loyalty",
    "rivalry","respect","contempt","admiration","scorn","gratitude","resentment","knowledge","ignorance",
    "belief","doubt","uncertainty","conviction","skepticism","confidence","hesitation","speculation",
    "intuition","memory","forgetting","chair","table","dog","stone","tree","river","mountain","hammer",
    "rope","lamp","cup","window","theorem","philosophy","ontology","epistemology","axiom","principle",
    "framework","paradigm","schema","abstraction","hypothetical","imaginary","fictional","speculative",
    "conjectural","perhaps","supposedly","allegedly","putative"]

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def process(kv, label, common_words):
    print("\n" + "#"*78)
    print(f"# {label}   (vocab {len(kv.index_to_key):,})")
    print("#"*78)
    mu = kv.vectors.mean(axis=0)

    # top PCs from a 100k centered slice
    S = kv.vectors[:100000] - mu
    cov = (S.T @ S) / S.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    PCs = evecs[:, ::-1]                      # most-significant first
    var = evals[::-1] / evals.sum()

    # frequency-rank correlation of the top 6 PCs (this substrate's own order)
    own = kv.index_to_key[:50000]
    Vown = np.stack([kv[w] for w in own]) - mu
    ranks = np.arange(len(own))
    print("\n  top-PC frequency-rank correlation (|rho| high => that PC ~ frequency):")
    for i in range(6):
        rho, _ = spearmanr(Vown @ PCs[:, i], ranks)
        print(f"    PC{i+1}: var={var[i]*100:4.1f}%   spearman(PC,freq-rank)={rho:+.3f}")
    E = PCs[:, :3]                            # ABTT: strip top 3
    print("  -> stripping top 3 PCs (All-But-The-Top)")

    def clean(v):
        c = v - mu
        return c - E @ (E.T @ c)

    # axes (clean) + frequency loading of the RAW contrast (before stripping)
    axes, fl = {}, {}
    pc1 = unit(PCs[:, 0])
    for name, a, b in CONTRASTS:
        raw = kv[b] - kv[a]
        fl[name] = float(unit(raw) @ pc1)
        axes[name] = unit(clean(raw))

    print("\n  each axis's frequency loading BEFORE stripping  cos(raw, PC1):")
    for name in NAMES:
        print(f"    {name:<16} {fl[name]:+.3f}")

    # 7x7 pairwise cosines (clean)
    print("\n  pairwise cosines AFTER stripping (independence of the 7 axes):")
    print("       " + "".join(f"{n.split('->')[1][:6]:>8}" for n in NAMES))
    for n1 in NAMES:
        print(f"  {n1.split('->')[1][:5]:<5}" + "".join(f"{axes[n1]@axes[n2]:>8.2f}" for n2 in NAMES))

    # pole words (clean, normalized, over own top-50k)
    C = np.stack([clean(kv[w]) for w in own])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    print("\n  pole words AFTER stripping (top 12 each end):")
    for name in NAMES:
        sc = C @ axes[name]
        o = np.argsort(sc)
        src, tgt = name.split("->")
        print(f"    {name}")
        print(f"      +{tgt:<9}: {[own[i] for i in o[::-1][:12]]}")
        print(f"      +{src:<9}: {[own[i] for i in o[:12]]}")

    # cognitive coverage (clean), per axis
    cogv = np.stack([unit(clean(kv[w])) for w in COG if w in kv])
    print(f"\n  cognitive-sample coverage AFTER stripping (mean|cos|, n={cogv.shape[0]}):")
    for name in NAMES:
        mc = float(np.mean(np.abs(cogv @ axes[name])))
        print(f"    {name:<16} {mc:.3f}")

    # cross-substrate projections over the shared word list
    proj = {}
    for w in common_words:
        if w in kv:
            cv = unit(clean(kv[w]))
            proj[w] = np.array([cv @ axes[n] for n in NAMES])
    return proj

print("Loading GloVe...")
glove = api.load("glove-wiki-gigaword-300")
common_words = glove.index_to_key[:50000]      # canonical shared vocab
proj_glove = process(glove, "GLOVE (glove-wiki-gigaword-300)", common_words)
del glove; gc.collect()

print("\nLoading fastText...")
ft = api.load("fasttext-wiki-news-subwords-300")
proj_ft = process(ft, "FASTTEXT (fasttext-wiki-news-subwords-300)", common_words)
del ft; gc.collect()

# ---- cross-substrate agreement ----
shared = [w for w in common_words if w in proj_glove and w in proj_ft]
G = np.stack([proj_glove[w] for w in shared])
F = np.stack([proj_ft[w] for w in shared])
print("\n" + "="*78)
print(f"CROSS-SUBSTRATE AGREEMENT  (spearman of word-projections, n={len(shared)})")
print("  high => the axis ranks words the same way in GloVe and fastText (substrate-general)")
print("="*78)
for i, name in enumerate(NAMES):
    rho, _ = spearmanr(G[:, i], F[:, i])
    print(f"    {name:<16} rho = {rho:+.3f}")
