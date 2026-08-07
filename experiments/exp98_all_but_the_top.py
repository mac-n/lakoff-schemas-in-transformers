"""
exp98 — All-But-The-Top (Mu et al 2017) on cognitive vocab in both substrates.

Niamh's framing: PC1 is the coherence/sense direction (structural, ~31% var
in both GloVe and fastText). The principled cleanup is to subtract not just
the anisotropy mean but also the top K PCs.

Procedure:
  1. Compute PCA on cognitive vocab
  2. Subtract top K PCs (K=1, 2, 3) from each vector
  3. Renormalize, compute new PCA
  4. Look at what the "new PC1" is at each cleanup level
  5. Compare GloVe and fastText to see if substrate-invariance improves

The substantive question: at what level of cleanup do the dominant directions
become substrate-invariant SEMANTIC content (not tokenization residue)?
"""
import numpy as np
import gensim.downloader as api


def unit(v):
    return v / np.linalg.norm(v)


cognitive_words = [
    "happiness", "sadness", "anger", "fear", "joy", "grief", "envy", "jealousy",
    "pride", "humility", "shame", "guilt", "contentment", "longing", "yearning",
    "delight", "melancholy", "rage", "elation", "despair", "serenity", "anguish",
    "love", "hate", "compassion", "cruelty", "tenderness", "harshness",
    "disgust", "awe", "wonder", "boredom", "ecstasy", "agony", "bliss",
    "loneliness", "togetherness", "intimacy", "alienation", "isolation",
    "ambition", "determination", "resignation", "willpower", "discipline",
    "procrastination", "perseverance", "complacency", "vigilance", "diligence",
    "industriousness", "negligence", "carelessness", "resolution", "indecision",
    "commitment", "hesitation", "courage", "cowardice", "boldness", "timidity",
    "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
    "respect", "contempt", "admiration", "scorn", "gratitude", "resentment",
    "cooperation", "competition", "kindness", "generosity", "greed",
    "empathy", "indifference", "honesty", "deception",
    "knowledge", "ignorance", "belief", "doubt", "uncertainty", "conviction",
    "skepticism", "confidence", "hesitation", "speculation", "intuition",
    "memory", "forgetting", "understanding", "confusion", "insight", "delusion",
    "wisdom", "foolishness", "discernment", "naivete",
    "attention", "distraction", "focus", "concentration",
    "curiosity", "interest", "fascination", "disinterest",
    "creativity", "imagination", "logic", "reasoning", "judgment",
    "theory", "principle", "concept", "framework", "paradigm", "schema",
    "abstraction", "generalization", "specification", "category", "definition",
    "axiom", "premise", "conclusion", "inference", "deduction", "induction",
    "possibility", "actuality", "necessity", "contingency", "probability",
    "certainty", "hypothesis", "fact", "truth", "falsehood", "myth", "reality",
    "freedom", "constraint", "choice", "obligation", "permission", "prohibition",
    "intention", "accident", "purpose", "happenstance",
    "experience", "perception", "sensation", "thought", "feeling", "awareness",
    "consciousness", "unconsciousness", "presence", "absence",
    "self", "selfhood", "identity", "personhood", "individuality", "autonomy",
    "ego", "soul", "spirit", "mind", "body", "subjectivity", "objectivity",
    "virtue", "vice", "morality", "ethics", "goodness", "evil", "righteousness",
    "depravity", "honor", "dishonor", "dignity", "indignity",
    "past", "future", "anticipation", "regret", "hope", "nostalgia",
    "expectation", "remembrance", "foresight",
]


def run_substrate(substrate_name, model_name):
    print(f"\n{'=' * 78}")
    print(f"SUBSTRATE: {substrate_name}")
    print(f"{'=' * 78}")
    wv = api.load(model_name)
    mu = wv.vectors.mean(axis=0)

    def get(w):
        if w not in wv.key_to_index:
            return None
        v = wv[w] - mu
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else None

    vecs = np.stack([v for w in cognitive_words for v in [get(w)] if v is not None])
    print(f"  {len(vecs)} cognitive vectors")

    # Iteratively subtract top K PCs and look at what's left as the new PC1
    for K in [0, 1, 2, 3]:
        if K == 0:
            current_vecs = vecs.copy()
        else:
            # Compute PCA on current, subtract top K
            _, _, Vt_init = np.linalg.svd(current_vecs, full_matrices=False) \
                if False else (None, None, None)
            # Recompute from scratch each level
            current_vecs = vecs.copy()
            _, _, Vt_remove = np.linalg.svd(current_vecs, full_matrices=False)
            for k in range(K):
                pc = Vt_remove[k]
                # Subtract this PC from every vector
                projections = current_vecs @ pc
                current_vecs = current_vecs - projections[:, None] * pc[None, :]
            # Renormalize
            norms = np.linalg.norm(current_vecs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            current_vecs = current_vecs / norms

        _, S, Vt = np.linalg.svd(current_vecs, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()

        print(f"\n--- K = {K} (subtracted mean{' + top ' + str(K) + ' PCs' if K else ''}) ---")
        print(f"  Variance distribution: PC1={var[0]*100:.2f}%, PC2={var[1]*100:.2f}%, "
              f"PC3={var[2]*100:.2f}%, PC4={var[3]*100:.2f}%, PC5={var[4]*100:.2f}%")

        # Pole vocab of new PC1
        new_pc1 = unit(Vt[0])
        print(f"  NEW PC1 positive pole (top 10):")
        for w, s in wv.similar_by_vector(new_pc1.astype(np.float32), topn=10):
            print(f"    {w:25s}  {s:+.4f}")
        print(f"  NEW PC1 negative pole (top 10):")
        for w, s in wv.similar_by_vector((-new_pc1).astype(np.float32), topn=10):
            print(f"    {w:25s}  {s:+.4f}")

    # Return final cleaned vectors for cross-substrate comparison
    return current_vecs


# Run on both substrates
print("\n##############################################################################")
print("# All-But-The-Top experiment: cognitive vocab in GloVe and fastText")
print("##############################################################################")

print("\nLoading GloVe...")
wv_glove = api.load("glove-wiki-gigaword-300")
mu_glove = wv_glove.vectors.mean(axis=0)


def get_glove(w):
    if w not in wv_glove.key_to_index:
        return None
    v = wv_glove[w] - mu_glove
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else None


vecs_glove = np.stack([v for w in cognitive_words for v in [get_glove(w)] if v is not None])
print(f"GloVe: {len(vecs_glove)} cognitive vectors")

print("\nLoading fastText...")
wv_ft = api.load("fasttext-wiki-news-subwords-300")
mu_ft = wv_ft.vectors.mean(axis=0)


def get_ft(w):
    if w not in wv_ft.key_to_index:
        return None
    v = wv_ft[w] - mu_ft
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else None


vecs_ft = np.stack([v for w in cognitive_words for v in [get_ft(w)] if v is not None])
print(f"fastText: {len(vecs_ft)} cognitive vectors")


def all_but_the_top(vecs, K):
    """Subtract top K PCs and renormalize. K=0 returns original."""
    if K == 0:
        return vecs.copy()
    current = vecs.copy()
    _, _, Vt = np.linalg.svd(current, full_matrices=False)
    for k in range(K):
        pc = Vt[k]
        projections = current @ pc
        current = current - projections[:, None] * pc[None, :]
    norms = np.linalg.norm(current, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    return current / norms


def get_top_pcs(vecs, n=5):
    _, S, Vt = np.linalg.svd(vecs, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    pcs = [unit(Vt[i]) for i in range(n)]
    return pcs, var[:n]


print("\n" + "=" * 78)
print("RESULTS — variance of PC1 after K-level cleanup, and cross-substrate cos(PC1, PC1)")
print("=" * 78)

for K in [0, 1, 2, 3]:
    glove_cleaned = all_but_the_top(vecs_glove, K)
    ft_cleaned = all_but_the_top(vecs_ft, K)
    glove_pcs, glove_var = get_top_pcs(glove_cleaned, 5)
    ft_pcs, ft_var = get_top_pcs(ft_cleaned, 5)

    print(f"\n=== After K = {K} cleanup ===")
    print(f"  GloVe new PC1 variance: {glove_var[0]*100:.2f}%")
    print(f"  fastText new PC1 variance: {ft_var[0]*100:.2f}%")

    # Cross-substrate cosines between top PCs
    print(f"  Cross-substrate alignment of top PCs:")
    for gi in range(3):
        for fi in range(3):
            c = abs(float(glove_pcs[gi] @ ft_pcs[fi]))
            if c > 0.3:
                print(f"    |cos(GloVe-PC{gi+1}, fastText-PC{fi+1})| = {c:.4f}"
                      f"{' ← substantial alignment' if c > 0.5 else ''}")

    # Pole vocab of GloVe-PC1 at this cleanup level
    print(f"\n  GloVe PC1 after K={K} cleanup:")
    print(f"    + pole: ", end="")
    pole_words = [w for w, _ in wv_glove.similar_by_vector(glove_pcs[0].astype(np.float32), topn=6)]
    print(", ".join(pole_words))
    print(f"    − pole: ", end="")
    neg_words = [w for w, _ in wv_glove.similar_by_vector((-glove_pcs[0]).astype(np.float32), topn=6)]
    print(", ".join(neg_words))

    print(f"  fastText PC1 after K={K} cleanup:")
    print(f"    + pole: ", end="")
    pole_words = [w for w, _ in wv_ft.similar_by_vector(ft_pcs[0].astype(np.float32), topn=6)]
    print(", ".join(pole_words))
    print(f"    − pole: ", end="")
    neg_words = [w for w, _ in wv_ft.similar_by_vector((-ft_pcs[0]).astype(np.float32), topn=6)]
    print(", ".join(neg_words))


np.savez("/Users/macn/Documents/embeddingexp/exp98_results.npz",
         glove_vecs=vecs_glove,
         ft_vecs=vecs_ft)
print("\nSaved exp98_results.npz")
