"""
exp100_pc1_vs_hardness.py

Niamh's question: exp90 mean-centered but did NOT remove PC1, and PC1 ~ the
COHERENCE/SENSE direction (exp97). So how much of HARDNESS's cognitive-content
coverage is PC1 leakage? How much variance does hardness explain with PC1 present
vs after removing it (ABTT k=1)?

Replicates exp90's coverage metric exactly:
  coverage = mean over cognitive test words of |cos(word, axis)|
  (vectors = mean-centered then unit-normalized -- 'get_deanisotropized')
Adds: variance fraction = mean(cos^2); cos(hardness, PC1); and the same numbers
after deflating PC1 from both the sample and the axis.
"""
import numpy as np
import gensim.downloader as api

def unit(v):
    return v / np.linalg.norm(v)

def build_axis(wv, pairs):
    # pairs are (positive_pole, negative_pole); direction = pos - neg
    offs = [wv[a] - wv[b] for a, b in pairs if a in wv and b in wv]
    return unit(np.stack(offs).mean(axis=0))

# exp90's canonical axis and the firm/solid-free version (a,b)=(hard-pole, soft-pole)
HARD_EXP90 = [("hard", "soft"), ("firm", "mushy"), ("rigid", "pliable"), ("solid", "flimsy")]
HARD_CLEAN = [("hard", "soft"), ("stiff", "mushy"), ("rigid", "pliable"), ("rigid", "squishy")]

# exp90 cognitive test categories (verbatim)
test_categories = {
    "EMOTIONS_AFFECTIVE": ["happiness","sadness","anger","envy","jealousy","pride","humility",
        "contentment","longing","delight","melancholy","rage","elation","despair","serenity","anguish","disgust"],
    "AGENTIVE_STATES": ["ambition","determination","resignation","willpower","discipline","procrastination",
        "perseverance","complacency","vigilance","diligence","industriousness","negligence","carelessness"],
    "SOCIAL_RELATIONAL": ["trust","betrayal","friendship","enmity","loyalty","rivalry","respect","contempt",
        "admiration","scorn","gratitude","resentment"],
    "EPISTEMIC_STATES": ["knowledge","ignorance","belief","doubt","uncertainty","conviction","skepticism",
        "confidence","hesitation","speculation","intuition","memory","forgetting"],
    "CONCRETE_NOUNS": ["chair","table","dog","stone","tree","river","mountain","hammer","rope","lamp","cup","window"],
    "ABSTRACT_FORMAL": ["theorem","philosophy","ontology","epistemology","axiom","principle","framework",
        "paradigm","schema","abstraction"],
    "MODAL_HYPOTHETICAL": ["hypothetical","imaginary","fictional","speculative","conjectural","perhaps",
        "supposedly","allegedly","putative"],
}

print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)

def deaniso(word):
    if word not in wv: return None
    return unit(wv[word] - mu)

# ---- cognitive test sample (exactly as exp90) ----
sample = np.stack([v for w in sum(test_categories.values(), []) if (v := deaniso(w)) is not None])
print(f"  cognitive test sample: {sample.shape[0]} words, mean-centered + unit-normalized\n")

# ---- axes ----
h90 = build_axis(wv, HARD_EXP90)
hcl = build_axis(wv, HARD_CLEAN)

# ---- PC1: global (all GloVe, mean-centered) and local (cognitive sample) ----
# global PC1 via covariance eigendecomp on a large mean-centered slice
X = wv.vectors[:100000] - mu          # 100k most-frequent, mean-centered (unnormalized -> standard PCA)
cov = (X.T @ X) / X.shape[0]
evals, evecs = np.linalg.eigh(cov)
pc1_global = unit(evecs[:, -1])
gvar = evals[-1] / evals.sum()

# local PC1 of the cognitive sample (the 'coherence/sense' PC1 from exp97 lives here)
Xs = sample - sample.mean(axis=0)
covs = (Xs.T @ Xs) / Xs.shape[0]
es, ev = np.linalg.eigh(covs)
pc1_local = unit(ev[:, -1])
lvar = es[-1] / es.sum()

def coverage(axis, samp):
    cos = samp @ axis
    return float(np.mean(np.abs(cos))), float(np.mean(cos**2))

def deflate(vecs, u):
    # remove component along unit vector u, then renormalize rows (if 2D)
    if vecs.ndim == 1:
        return unit(vecs - (vecs @ u) * u)
    out = vecs - np.outer(vecs @ u, u)
    return out / np.linalg.norm(out, axis=1, keepdims=True)

print("=" * 74)
print("PC1 STRUCTURE")
print("=" * 74)
print(f"  global PC1 explains {gvar*100:5.1f}% of variance (top-100k GloVe)")
print(f"  local  PC1 explains {lvar*100:5.1f}% of variance (cognitive sample)")
cg, cg2 = coverage(pc1_global, sample)
cl, cl2 = coverage(pc1_local, sample)
print(f"  coverage of PC1_global on cognitive sample: mean|cos|={cg:.3f}  mean(cos^2)={cg2:.3f}")
print(f"  coverage of PC1_local  on cognitive sample: mean|cos|={cl:.3f}  mean(cos^2)={cl2:.3f}")

print("\n" + "=" * 74)
print("HARDNESS vs PC1  (alignment)")
print("=" * 74)
for name, h in [("HARDNESS_exp90", h90), ("HARDNESS_clean", hcl)]:
    print(f"  {name:<16}  cos(.,PC1_global)={h@pc1_global:+.3f}   cos(.,PC1_local)={h@pc1_local:+.3f}")

print("\n" + "=" * 74)
print("VARIANCE / COVERAGE EXPLAINED BY HARDNESS  (PC1 present vs removed)")
print("  metric: mean|cos| (exp90's coverage)  |  mean(cos^2) (variance fraction)")
print("=" * 74)
for name, h in [("HARDNESS_exp90", h90), ("HARDNESS_clean", hcl)]:
    mc, mc2 = coverage(h, sample)
    # remove global PC1 from both sample and axis (ABTT k=1)
    s_g = deflate(sample, pc1_global); h_g = deflate(h, pc1_global)
    mcg, mcg2 = coverage(h_g, s_g)
    # remove local PC1 (the coherence/sense direction) from both
    s_l = deflate(sample, pc1_local); h_l = deflate(h, pc1_local)
    mcl, mcl2 = coverage(h_l, s_l)
    print(f"\n  {name}")
    print(f"    PC1 PRESENT (= exp90 setup):       mean|cos|={mc:.3f}   var={mc2:.3f}")
    print(f"    minus global PC1 (ABTT k=1):       mean|cos|={mcg:.3f}   var={mcg2:.3f}   "
          f"(Δ|cos|={mcg-mc:+.3f})")
    print(f"    minus local PC1 (coherence/sense): mean|cos|={mcl:.3f}   var={mcl2:.3f}   "
          f"(Δ|cos|={mcl-mc:+.3f})")

print("\n" + "=" * 74)
print("READING")
print("=" * 74)
print("""  - 'PC1 PRESENT mean|cos|' should reproduce exp90's ~0.21 for HARDNESS_exp90.
  - If cos(hardness, PC1) is small AND coverage barely changes when PC1 is
    removed, the hardness finding is NOT PC1 leakage -- it survives.
  - If coverage collapses when PC1 is removed, the '21%' was largely hardness
    riding on the coherence/sense gradient, and the headline needs softening.""")
