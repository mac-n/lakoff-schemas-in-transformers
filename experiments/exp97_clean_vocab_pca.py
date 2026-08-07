"""
exp97 — Clean vocab PCA: principled non-circular cognitive subspace.

Niamh's framing: "most of the dataset is entropy / attention sinks."
The noise tokens (email addresses, decimal numbers, weird strings) absorb
embedding-space variance without carrying semantic content. They're the
embedding analog of attention sinks.

Principled method to remove them: filter GloVe vocab by simple external
rules (no reference to our cognitive axes), then PCA on the cleaned vocab.

Filter rules:
  - exclude tokens with digits
  - exclude tokens with punctuation (except hyphens in compound words)
  - exclude very short tokens (< 3 chars)
  - exclude tokens that start with capital letter (rough proper-noun filter)
  - exclude tokens that contain non-ASCII characters

Then take top-N by frequency. PCA. Compare PC1 to:
  (a) PC1 of unfiltered GloVe vocab (the tokenization-residue direction)
  (b) PC1 of cognitive test vocab (the coherence direction)
  (c) Our cognitive axes

Predictions:
  - clean-vocab PC1 should be MORE semantic than unfiltered PC1
  - clean-vocab PC2/3 should approximate cognitive-vocab PC2/3
  - if PC1 of clean vocab is still sense/coherence, that's substantial confirmation
"""
import numpy as np
import gensim.downloader as api
import re


def unit(v):
    return v / np.linalg.norm(v)


def is_clean_english_content_word(token):
    """Filter to clean English content words via external rules."""
    if len(token) < 3:
        return False
    if any(c.isdigit() for c in token):
        return False
    # Allow only lowercase letters and possibly internal hyphen
    if not re.match(r'^[a-z]+(?:-[a-z]+)?$', token):
        return False
    return True


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)
print(f"  Full vocab: {len(wv.key_to_index)}")

# GloVe vocab is sorted by frequency
# Filter to clean content words
print("\nFiltering vocab to clean English content words...")
clean_words = []
for i, w in enumerate(wv.index_to_key):
    if is_clean_english_content_word(w):
        clean_words.append(w)
    if len(clean_words) >= 30000:
        break  # top 30K clean words

print(f"  First 30K clean words found from top {i+1} of vocab")
print(f"  Filtering ratio: {30000/(i+1)*100:.1f}% of GloVe tokens are 'clean'")
print(f"\n  First 30 clean words: {clean_words[:30]}")
print(f"  Last 10 (least common in 30K): {clean_words[-10:]}")


# Get vectors and deanisotropize
clean_indices = [wv.key_to_index[w] for w in clean_words]
clean_vecs = wv.vectors[clean_indices] - mu
norms = np.linalg.norm(clean_vecs, axis=1, keepdims=True)
norms[norms < 1e-10] = 1
clean_vecs = clean_vecs / norms

print(f"\n  Final clean vocab sample: {len(clean_vecs)} unit-normalized vectors")


# PCA
print("\nComputing PCA on clean content-word vocab...")
_, S, Vt = np.linalg.svd(clean_vecs, full_matrices=False)
var = (S ** 2) / (S ** 2).sum()

print(f"\nVariance distribution:")
for i in range(15):
    cum = sum(var[:i+1])
    print(f"  PC{i+1}: {var[i]*100:>5.2f}%  (cum: {cum*100:.2f}%)")

# Effective dimensionality
for thresh in [0.5, 0.8, 0.9, 0.95]:
    n = int(np.searchsorted(np.cumsum(var), thresh) + 1)
    print(f"  To capture {thresh*100:.0f}%: {n} PCs")

# Pole vocab of top 5 PCs
print("\nPC1-PC5 of CLEAN content-word vocab:")
for k in range(5):
    pc = unit(Vt[k])
    print(f"\n=== CLEAN-PC{k+1} (var: {var[k]*100:.2f}%) ===")
    print("  Positive pole (top 15):")
    for w, s in wv.similar_by_vector(pc.astype(np.float32), topn=15):
        print(f"    {w:25s}  {s:+.4f}")
    print("  Negative pole (top 15):")
    for w, s in wv.similar_by_vector((-pc).astype(np.float32), topn=15):
        print(f"    {w:25s}  {s:+.4f}")


# Compare to PCs of unfiltered (random) vocab
print("\n\n" + "=" * 78)
print("Comparison: PC1 of clean vs PC1 of unfiltered GloVe")
print("=" * 78)

# Sample equal size from unfiltered vocab
np.random.seed(42)
unfilt_idx = np.random.choice(len(wv.vectors), 30000, replace=False)
unfilt = wv.vectors[unfilt_idx] - mu
unfilt_norms = np.linalg.norm(unfilt, axis=1, keepdims=True)
unfilt_norms[unfilt_norms < 1e-10] = 1
unfilt = unfilt / unfilt_norms

_, _, Vt_unfilt = np.linalg.svd(unfilt, full_matrices=False)
pc1_unfilt = unit(Vt_unfilt[0])
pc1_clean = unit(Vt[0])

print(f"\ncos(PC1_clean, PC1_unfiltered) = {float(pc1_clean @ pc1_unfilt):+.4f}")
print("  (low if filtering changed PC1 direction substantively)")

print("\nPC1_unfiltered positive pole (top 10):")
for w, s in wv.similar_by_vector(pc1_unfilt.astype(np.float32), topn=10):
    print(f"    {w:25s}  {s:+.4f}")
print("PC1_unfiltered negative pole (top 10):")
for w, s in wv.similar_by_vector((-pc1_unfilt).astype(np.float32), topn=10):
    print(f"    {w:25s}  {s:+.4f}")


# Compare to cognitive PCs
print("\n\n" + "=" * 78)
print("Comparison: clean PCs vs cognitive vocab PCs")
print("=" * 78)

# Build cognitive sample
def get(w):
    if w not in wv.key_to_index: return None
    v = wv[w] - mu
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else None

cog_words = ['happiness','sadness','anger','fear','joy','grief','envy','jealousy','pride','humility','shame','guilt','contentment','longing','yearning','delight','melancholy','rage','elation','despair','serenity','anguish','love','hate','compassion','cruelty','tenderness','harshness','disgust','awe','wonder','boredom','ecstasy','agony','bliss','loneliness','togetherness','intimacy','alienation','isolation','ambition','determination','resignation','willpower','discipline','procrastination','perseverance','complacency','vigilance','diligence','industriousness','negligence','carelessness','resolution','indecision','commitment','hesitation','courage','cowardice','boldness','timidity','trust','betrayal','friendship','enmity','loyalty','rivalry','respect','contempt','admiration','scorn','gratitude','resentment','cooperation','competition','kindness','generosity','greed','empathy','indifference','honesty','deception','knowledge','ignorance','belief','doubt','uncertainty','conviction','skepticism','confidence','speculation','intuition','memory','forgetting','understanding','confusion','insight','delusion','wisdom','foolishness','discernment','naivete','attention','distraction','focus','concentration','curiosity','interest','fascination','disinterest','creativity','imagination','logic','reasoning','judgment','theory','principle','concept','framework','paradigm','schema','abstraction','generalization','specification','category','definition','axiom','premise','conclusion','inference','deduction','induction','possibility','actuality','necessity','contingency','probability','certainty','hypothesis','fact','truth','falsehood','myth','reality','freedom','constraint','choice','obligation','permission','prohibition','intention','accident','purpose','happenstance','experience','perception','sensation','thought','feeling','awareness','consciousness','unconsciousness','presence','absence','self','selfhood','identity','personhood','individuality','autonomy','ego','soul','spirit','mind','body','subjectivity','objectivity','virtue','vice','morality','ethics','goodness','evil','righteousness','depravity','honor','dishonor','dignity','indignity','past','future','anticipation','regret','hope','nostalgia','expectation','remembrance','foresight']
cog_vecs = np.stack([v for w in cog_words for v in [get(w)] if v is not None])
_, _, Vt_cog = np.linalg.svd(cog_vecs, full_matrices=False)
pc1_cog = unit(Vt_cog[0])
pc2_cog = unit(Vt_cog[1])
pc3_cog = unit(Vt_cog[2])

print(f"\ncos(PC1_clean, PC1_cog) = {float(pc1_clean @ pc1_cog):+.4f}")
print(f"cos(PC1_clean, PC2_cog) = {float(pc1_clean @ pc2_cog):+.4f}")
print(f"cos(PC1_clean, PC3_cog) = {float(pc1_clean @ pc3_cog):+.4f}")
print(f"cos(PC2_clean, PC1_cog) = {float(unit(Vt[1]) @ pc1_cog):+.4f}")
print(f"cos(PC2_clean, PC2_cog) = {float(unit(Vt[1]) @ pc2_cog):+.4f}")
print(f"cos(PC3_clean, PC1_cog) = {float(unit(Vt[2]) @ pc1_cog):+.4f}")
print(f"cos(PC3_clean, PC2_cog) = {float(unit(Vt[2]) @ pc2_cog):+.4f}")
print(f"cos(PC3_clean, PC3_cog) = {float(unit(Vt[2]) @ pc3_cog):+.4f}")

np.savez("/Users/macn/Documents/embeddingexp/exp97_results.npz",
         clean_var=var,
         pc1_clean=pc1_clean)
print("\nSaved exp97_results.npz")
