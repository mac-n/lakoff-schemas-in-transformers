"""
Experiment 1: Does adding an 'up' direction to 'I feel sad' move the embedding
toward improvement/lifting language?

This tests whether Lakoffian somatic primitives (UP, DOWN, IN, OUT...) compose
with affect states via simple vector arithmetic in sentence embedding space.

Setup:
    pip install sentence-transformers numpy

Run:
    python exp1_embedding_arithmetic.py

Expected runtime: ~30 seconds (first run downloads model, ~80MB)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Small, fast, runs fine on CPU. ~80MB.
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    """Get unit-normalized embeddings."""
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs


def direction_from_pairs(positive_words, negative_words):
    """
    Build a directional vector by taking mean(positive) - mean(negative).
    This is the standard 'concept axis' approach.
    """
    pos_vecs = embed(positive_words)
    neg_vecs = embed(negative_words)
    direction = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    # Normalize so we can scale it cleanly
    direction = direction / np.linalg.norm(direction)
    return direction


def nearest_neighbors(query_vec, candidates, candidate_texts, k=5):
    """Find top-k candidates by cosine similarity to query_vec."""
    # Normalize query
    q = query_vec / np.linalg.norm(query_vec)
    sims = candidates @ q  # cosine sim since both are unit-normalized
    top_idx = np.argsort(-sims)[:k]
    return [(candidate_texts[i], float(sims[i])) for i in top_idx]


# ---------- Build the UP direction ----------
print("\nBuilding UP direction from word pairs...")
up_words = ["rising", "lifting", "ascending", "climbing", "soaring", "elevating", "uplifting"]
down_words = ["falling", "sinking", "descending", "dropping", "plummeting", "lowering", "collapsing"]
up_direction = direction_from_pairs(up_words, down_words)

# ---------- Candidate pool of sentences to compare against ----------
# Mix of: improvement, decline, neutral, off-topic.
# We want to see if 'I feel sad' + UP lands closer to the improvement cluster.
candidates_text = [
    # Improvement / mood-lifting
    "I feel better now",
    "My mood is lifting",
    "I'm starting to feel happy again",
    "Things are looking up",
    "I feel uplifted",
    "I am cheering up",
    "My spirits are rising",
    # Decline / mood-falling
    "I feel worse now",
    "My mood is sinking",
    "I'm getting more depressed",
    "Things are falling apart",
    "I feel crushed",
    "My spirits are dropping",
    # Sustained sad (control: shouldn't be the top match if UP works)
    "I feel sad",
    "I am unhappy",
    "I feel down",
    "I am miserable",
    # Off-topic (control: shouldn't show up at all if direction is meaningful)
    "The cat is on the mat",
    "I bought some apples at the market",
    "The weather is cloudy today",
    "Python is a programming language",
]
candidate_vecs = embed(candidates_text)

# ---------- The actual test ----------
print("\n" + "=" * 60)
print("TEST: 'I feel sad' + UP direction")
print("=" * 60)

sad_vec = embed(["I feel sad"])[0]

# Try different scales of the UP direction
for alpha in [0.0, 0.3, 0.5, 1.0, 1.5]:
    shifted = sad_vec + alpha * up_direction
    print(f"\nalpha = {alpha} (UP strength):")
    neighbors = nearest_neighbors(shifted, candidate_vecs, candidates_text, k=5)
    for text, sim in neighbors:
        print(f"   {sim:.3f}  {text}")

# ---------- Bonus: try a few other compositions ----------
print("\n" + "=" * 60)
print("BONUS: 'I am anxious' + DOWN-as-calming")
print("(testing whether DOWN -> calming/grounding rather than depressing)")
print("=" * 60)

# DOWN can mean 'depressed' OR 'grounded/settled' - which does the model encode?
down_direction = -up_direction
anxious_vec = embed(["I am anxious"])[0]

calming_candidates = [
    "I am calm",
    "I feel grounded",
    "I am at peace",
    "I feel settled",
    "I am relaxed",
    "I am more anxious",
    "I am panicking",
    "I am depressed",
    "I feel low",
    "I feel heavy",
]
calming_vecs = embed(calming_candidates)

for alpha in [0.0, 0.5, 1.0]:
    shifted = anxious_vec + alpha * down_direction
    print(f"\nalpha = {alpha} (DOWN strength):")
    neighbors = nearest_neighbors(shifted, calming_vecs, calming_candidates, k=4)
    for text, sim in neighbors:
        print(f"   {sim:.3f}  {text}")

# ---------- Sanity check: orthogonal direction should NOT move things meaningfully ----------
print("\n" + "=" * 60)
print("SANITY CHECK: 'I feel sad' + random direction")
print("(should produce noise, not a clean shift)")
print("=" * 60)

np.random.seed(42)
random_dir = np.random.randn(*up_direction.shape)
random_dir = random_dir / np.linalg.norm(random_dir)

for alpha in [0.0, 1.0]:
    shifted = sad_vec + alpha * random_dir
    print(f"\nalpha = {alpha} (random direction):")
    neighbors = nearest_neighbors(shifted, candidate_vecs, candidates_text, k=5)
    for text, sim in neighbors:
        print(f"   {sim:.3f}  {text}")

print("\n" + "=" * 60)
print("Done. If UP works as a primitive, alpha > 0 should push the")
print("nearest neighbors toward the improvement cluster and away from")
print("the sad/decline cluster. The random direction shouldn't do this.")
print("=" * 60)
