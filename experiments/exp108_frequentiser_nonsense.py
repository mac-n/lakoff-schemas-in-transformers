"""
exp108_frequentiser_nonsense.py

Niamh: a frequentiser matrix -- but the rare end of frequency is mostly NONSENSE
(junk tokens), not rare real words. So "make rarer" naively = "make more junk".
Three things tangle at the low-frequency end: register/formality, genuine rare
real words, and pure nonsense. We want the register shift, cleanly separated.

PART A  -- DIAGNOSTIC: how much does the nonsense tail distort the frequency axis?
  f_all   = regression of log-rank on embeddings over a WIDE range (incl. junk tail)
  f_clean = same regression over MEANINGFUL words only (alphabetic, length>=4, top ranks)
  Compare directions; show what sits at each axis's rare extreme.

PART B  -- a clean FREQUENTISER from register synonyms (formal/rare -> casual/common).
  Meaning-preserving by construction (synonyms), never junk. Fit the casual-direction
  translation; check it (i) really is a frequency shift, (ii) aligns with f_clean not
  the nonsense axis, (iii) generalises leave-one-out, (iv) preserves meaning.
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr

def unit(v): return v / (np.linalg.norm(v) + 1e-12)
def unit_rows(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300"); vocab = set(kv.index_to_key)
mu = kv.vectors.mean(axis=0); d = 300
rank = {w: i for i, w in enumerate(kv.index_to_key)}

def freq_axis(words):
    X = np.stack([kv[w] - mu for w in words])
    y = np.log1p(np.array([rank[w] for w in words], float)); y = y - y.mean()
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return unit(w)

# PC1 for reference
S = kv.vectors[:100000] - mu
ev, evec = np.linalg.eigh((S.T @ S) / S.shape[0]); pc1 = unit(evec[:, -1])

# wide set (includes junk tail) vs meaningful-only set
wide = kv.index_to_key[:200000]
meaningful = [w for w in kv.index_to_key[:50000] if w.isalpha() and w.islower() and 4 <= len(w) <= 15]
f_all = freq_axis(wide)
f_clean = freq_axis(meaningful)

print("\n" + "=" * 70); print("PART A — does the nonsense tail distort the frequency axis?"); print("=" * 70)
print(f"  meaningful words: {len(meaningful)} of top-50k")
print(f"  cos(f_all, f_clean) = {f_all @ f_clean:+.3f}")
print(f"  cos(f_all,  PC1)    = {f_all @ pc1:+.3f}")
print(f"  cos(f_clean, PC1)   = {f_clean @ pc1:+.3f}")

# what sits at the rare extreme of each axis (over top-200k)?
W = kv.index_to_key[:200000]; Xw = np.stack([kv[w] - mu for w in W])
for name, ax in [("f_all", f_all), ("f_clean", f_clean)]:
    proj = Xw @ ax
    rare = [W[i] for i in np.argsort(proj)[:15]]; comm = [W[i] for i in np.argsort(proj)[::-1][:15]]
    # which end is "rare"? check sign via correlation with rank
    rho, _ = spearmanr(proj, np.arange(len(W)))
    rare_end, comm_end = (rare, comm) if rho > 0 else (comm, rare)
    print(f"\n  {name}: rare extreme -> {rare_end[:12]}")

print("\n" + "=" * 70); print("PART B — a clean FREQUENTISER from register synonyms"); print("=" * 70)
PAIRS = [  # (formal/rarer, casual/commoner)
    ("purchase","buy"),("utilize","use"),("commence","begin"),("assist","help"),
    ("obtain","get"),("require","need"),("sufficient","enough"),("demonstrate","show"),
    ("approximately","about"),("numerous","many"),("additional","more"),("residence","home"),
    ("vehicle","car"),("beverage","drink"),("inquire","ask"),("depart","leave"),
    ("reside","live"),("construct","build"),("terminate","end"),("comprehend","understand"),
    ("attempt","try"),("permit","let"),("acquire","get"),("consume","eat"),
    ("sufficient","enough"),("initial","first"),("final","last"),("rapid","fast"),
    ("difficult","hard"),("assistance","help"),("commence","start"),("conceal","hide"),
    ("inquire","ask"),("observe","watch"),("respond","answer"),("anticipate","expect"),
]
PAIRS = [(a, b) for a, b in PAIRS if a in vocab and b in vocab]
df = np.mean([rank[b] - rank[a] for a, b in PAIRS])   # casual rank - formal rank (want <0)
print(f"  usable pairs: {len(PAIRS)}")
print(f"  mean(rank_casual - rank_formal) = {df:,.0f}   (negative => casual really is more frequent)")

t_freq = unit(np.mean([(kv[b] - mu) - (kv[a] - mu) for a, b in PAIRS], axis=0))  # formal->casual
print(f"\n  frequentiser direction t_freq (formal->casual):")
print(f"    cos(t_freq, f_clean) = {t_freq @ f_clean:+.3f}   (aligns with clean frequency axis?)")
print(f"    cos(t_freq, f_all)   = {t_freq @ f_all:+.3f}")
print(f"    cos(t_freq, PC1)     = {t_freq @ pc1:+.3f}")

# leave-one-out generalisation + meaning preservation
pool_w = sorted({w for p in PAIRS for w in p} |
                set(np.random.default_rng(0).choice(meaningful, 3000, replace=False)))
Pool = unit_rows(np.stack([kv[w] - mu for w in pool_w])); pidx = {w: i for i, w in enumerate(pool_w)}
hit = 0; cos_t = []; cos_pres = []; rank_gain = 0
for i, (a, b) in enumerate(PAIRS):
    t = unit(np.mean([(kv[bb]-mu)-(kv[aa]-mu) for j,(aa,bb) in enumerate(PAIRS) if j!=i], axis=0))
    base = kv[a] - mu
    pred = base + np.linalg.norm(kv[b]-kv[a]) * t      # step by a typical pair magnitude
    cos_t.append(float(unit(pred) @ unit(kv[b]-mu)))
    cos_pres.append(float(unit(pred) @ unit(base)))
    sims = Pool @ unit(pred); sims[pidx[a]] = -2
    nn = pool_w[int(np.argmax(sims))]
    if nn == b: hit += 1
    if rank.get(nn, 1e9) < rank[a]: rank_gain += 1
n = len(PAIRS)
print(f"\n  leave-one-out:")
print(f"    retrieves the casual synonym (top1): {hit/n:.0%}")
print(f"    prediction lands on a MORE-frequent word: {rank_gain/n:.0%}")
print(f"    mean cosine to target casual word: {np.mean(cos_t):.3f}")
print(f"    mean cosine to original (meaning preserved): {np.mean(cos_pres):.3f}")
print("""
READING
  PART A: cos(f_all,f_clean) << 1 => the nonsense tail was steering the frequency
          axis; f_clean (meaningful-only) is the register/frequency axis we want.
  PART B: t_freq aligning with f_clean (not PC1/nonsense) + meaning preserved +
          lands more-frequent => a clean frequentiser that stays in real-word space.
          This is the direction; the curved/matrix version follows from fitting the
          action on many meaningful pairs (next), now safe from junk.
""")
