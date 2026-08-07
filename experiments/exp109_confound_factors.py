"""
exp109_confound_factors.py

Niamh: build a nonsensifier AND a frequentiser as factors of the "frequency"
confound -- and should there also be a FORMALISER? Don't assume the number of
factors: fit candidate directions and measure whether they're distinct or collapse.

Candidate factors (all unit directions):
  nonsensifier  n  = centroid(junk tokens) - centroid(real words)      (meaning -> junk)
  formaliser    fo = mean over WordNet synsets of (rare_lemma - common_lemma)
                     (meaning held CONSTANT within a synset -> pure register)
  frequentiser  fr = f_clean = regression of log-rank on MEANINGFUL words
                     (overall common <-> rare among real words)
Reference: PC1, and the hand-curated formal->casual direction from exp108 (t_hand).

Tests: pairwise cosines (distinct or collapsing?), and how much of the HARDNESS
contrast (the result that started this) lies along each / in their span.
"""
import numpy as np
import gensim.downloader as api
from nltk.corpus import wordnet as wn

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300"); vocab = set(kv.index_to_key)
mu = kv.vectors.mean(axis=0); d = 300
rank = {w: i for i, w in enumerate(kv.index_to_key)}
def cvec(w): return kv[w] - mu

S = kv.vectors[:100000] - mu
ev, evec = np.linalg.eigh((S.T @ S) / S.shape[0]); pc1 = unit(evec[:, -1])

top200 = kv.index_to_key[:200000]
meaningful = [w for w in kv.index_to_key[:50000] if w.isalpha() and w.islower() and 4 <= len(w) <= 15]

# --- nonsensifier: junk centroid - real centroid ---
def is_junk(w):
    return (not w.isalpha()) or len(w) <= 2 or any(ch.isdigit() for ch in w)
junk = [w for w in top200 if is_junk(w)]
real = meaningful
n_axis = unit(np.mean([cvec(w) for w in junk], axis=0) - np.mean([cvec(w) for w in real], axis=0))
print(f"  junk tokens: {len(junk):,}   real words: {len(real):,}")

# --- frequentiser (rariser): regression of log-rank on meaningful words ---
Xm = np.stack([cvec(w) for w in meaningful])
ym = np.log1p(np.array([rank[w] for w in meaningful], float)); ym -= ym.mean()
wfr, *_ = np.linalg.lstsq(Xm, ym, rcond=None); fr_axis = unit(wfr)

# --- formaliser: within-synset (rare - common), meaning held constant ---
pairs = set()
for w in meaningful[:20000]:
    for syn in wn.synsets(w):
        lems = [l for l in syn.lemma_names() if l.isalpha() and l.lower() in vocab]
        lems = sorted(set(l.lower() for l in lems), key=lambda x: rank[x])
        if len(lems) >= 2:
            common = lems[0]
            for rare in lems[1:]:
                if rank[rare] - rank[common] > 500:      # require a real frequency gap
                    pairs.add((common, rare))
pairs = list(pairs)
fo_axis = unit(np.mean([cvec(rare) - cvec(common) for common, rare in pairs], axis=0))
print(f"  WordNet register pairs (common->rare, gap>500): {len(pairs):,}")

# --- hand-curated formal->casual from exp108 (sanity) ---
HAND = [("purchase","buy"),("utilize","use"),("commence","begin"),("assist","help"),
        ("obtain","get"),("require","need"),("demonstrate","show"),("numerous","many"),
        ("residence","home"),("vehicle","car"),("inquire","ask"),("depart","leave"),
        ("construct","build"),("terminate","end"),("comprehend","understand"),("attempt","try")]
HAND = [(a,b) for a,b in HAND if a in vocab and b in vocab]
t_hand = unit(np.mean([cvec(a) - cvec(b) for a,b in HAND], axis=0))  # casual->formal (rare dir)

axes = {"nonsensifier": n_axis, "formaliser": fo_axis, "frequentiser": fr_axis,
        "PC1": pc1, "hand-formal": t_hand}
print("\n" + "="*70); print("PAIRWISE COSINES of candidate factors"); print("="*70)
names = list(axes)
print("            " + "".join(f"{n[:9]:>11}" for n in names))
for n1 in names:
    print(f"  {n1:<11}" + "".join(f"{axes[n1]@axes[n2]:>11.2f}" for n2 in names))

# --- hardness contrast: how much lies along each factor / in their span? ---
HARD = [("hard","soft"),("firm","mushy"),("rigid","pliable"),("solid","flimsy")]
hardness = unit(np.mean([cvec(a) - cvec(b) for a,b in HARD], axis=0))
print("\n" + "="*70); print("HARDNESS contrast vs the confound factors"); print("="*70)
for nm in ["nonsensifier","formaliser","frequentiser","PC1"]:
    print(f"  cos(hardness, {nm:<13}) = {hardness @ axes[nm]:+.3f}")
# fraction of hardness in span{n, fo, fr}
B = np.stack([n_axis, fo_axis, fr_axis]).T
Q, _ = np.linalg.qr(B)
proj = Q @ (Q.T @ hardness)
print(f"\n  fraction of ||hardness|| in span(nonsens, formal, frequent): {np.linalg.norm(proj):.3f}")
print(f"  (so {np.linalg.norm(proj)**2*100:.0f}% of hardness's variance is confound-space)")

print("""
READING
  - cos(formaliser, frequentiser) high (>~0.7) => they collapse into ONE factor;
    low (<~0.5) => distinct, keep both (Niamh's formaliser is warranted).
  - cos(nonsensifier, PC1) high => confirms PC1 is the nonsense axis.
  - hardness's loadings say which confound it was riding (nonsense? register?
    rarity?) and how much of it is confound vs genuine semantics.
""")
