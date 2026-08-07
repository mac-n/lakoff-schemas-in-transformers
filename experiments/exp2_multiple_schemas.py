"""
Experiment 2: Multiple Lakoffian schemas as compositional directions.

Tests:
  1. CONTAINER (OUT) on "I am stuck"        -> should land near 'free/released'
  2. PATH (FORWARD) on "I am stuck"         -> should land near 'progressing/moving'
     (Same stem, different schema, different destination - schema-disambiguation test)
  3. CONTAINER (IN) on "I am lonely"        -> should land near 'held/surrounded/included'
  4. Cross-schema controls (lonely+FORWARD, stuck+IN) - shouldn't help much
  5. Random direction control - shouldn't help at all

Run:
    python exp2_multiple_schemas.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


def direction(positive_words, negative_words):
    pos = embed(positive_words).mean(axis=0)
    neg = embed(negative_words).mean(axis=0)
    d = pos - neg
    return d / np.linalg.norm(d)


def nearest(query_vec, candidates_vecs, candidates_text, k=5):
    q = query_vec / np.linalg.norm(query_vec)
    sims = candidates_vecs @ q
    top_idx = np.argsort(-sims)[:k]
    return [(candidates_text[i], float(sims[i])) for i in top_idx]


def run_test(label, base_sentence, direction_vec, candidates_vecs, candidates_text, alphas=(0.0, 0.5, 1.0, 1.5)):
    print(f"\n{'=' * 64}")
    print(f"{label}")
    print(f"  base: '{base_sentence}'")
    print('=' * 64)
    base_vec = embed([base_sentence])[0]
    for alpha in alphas:
        shifted = base_vec + alpha * direction_vec
        print(f"\nalpha = {alpha}:")
        for text, sim in nearest(shifted, candidates_vecs, candidates_text, k=5):
            print(f"   {sim:.3f}  {text}")


# ---------- Build directions ----------
print("\nBuilding schema directions...")

# CONTAINER: IN <-> OUT
out_direction = direction(
    positive_words=["free", "released", "liberated", "escaped", "unbound", "outside"],
    negative_words=["trapped", "confined", "imprisoned", "enclosed", "bound", "inside"],
)
in_direction = -out_direction  # symmetric

# PATH: FORWARD <-> BACKWARD
forward_direction = direction(
    positive_words=["advancing", "progressing", "moving forward", "proceeding", "continuing", "developing"],
    negative_words=["stagnating", "halting", "stopping", "stalled", "regressing", "retreating"],
)

# CONTAINER (INCLUSION): being-with vs being-apart
# Slightly different framing of CONTAINER - for the loneliness test, we want
# "inside the circle of others" rather than "inside a trap"
inclusion_direction = direction(
    positive_words=["surrounded", "embraced", "included", "held", "connected", "accompanied"],
    negative_words=["isolated", "abandoned", "excluded", "alone", "separated", "apart"],
)

# Sanity-check random direction
np.random.seed(7)
random_direction = np.random.randn(*out_direction.shape)
random_direction = random_direction / np.linalg.norm(random_direction)


# ---------- Candidate pools ----------

stuck_candidates = [
    # Free / released cluster (CONTAINER-OUT escape)
    "I am free",
    "I am released",
    "I have escaped",
    "I am liberated",
    "I have broken free",
    # Progressing cluster (PATH-FORWARD escape)
    "I am moving forward",
    "I am making progress",
    "I am advancing",
    "I am proceeding",
    "Things are starting to move",
    # Sustained stuckness (the original state)
    "I am stuck",
    "I am trapped",
    "I am frozen",
    "I am paralyzed",
    "I cannot move",
    # Sad/down (semantically nearby but wrong direction for our tests)
    "I am unhappy",
    "I feel terrible",
    # Off-topic controls
    "The cat is on the mat",
    "I bought groceries",
    "Python is a language",
]
stuck_vecs = embed(stuck_candidates)

lonely_candidates = [
    # Held / surrounded / included cluster (CONTAINER-IN cure for loneliness)
    "I am surrounded by friends",
    "I am held",
    "I am loved",
    "I am embraced",
    "I am included",
    "I am connected to others",
    "I am with people who care",
    # Progressing (cross-schema control - shouldn't be the top match)
    "I am moving forward",
    "I am making progress",
    "I am developing",
    # Sustained loneliness (original state)
    "I am lonely",
    "I am alone",
    "I am isolated",
    "I feel abandoned",
    "I have no one",
    # Off-topic
    "The cat is on the mat",
    "Python is a language",
]
lonely_vecs = embed(lonely_candidates)


# ==================== TESTS ====================

# Test 1: CONTAINER-OUT on "I am stuck"
# Prediction: should pull toward free/released cluster
run_test(
    "TEST 1: 'I am stuck' + OUT (CONTAINER escape)",
    "I am stuck",
    out_direction,
    stuck_vecs,
    stuck_candidates,
)

# Test 2: PATH-FORWARD on "I am stuck"
# Prediction: should pull toward progressing/moving cluster
# (Different schema, different escape from same starting state)
run_test(
    "TEST 2: 'I am stuck' + FORWARD (PATH escape)",
    "I am stuck",
    forward_direction,
    stuck_vecs,
    stuck_candidates,
)

# Test 3: CONTAINER-IN (inclusion) on "I am lonely"
# Prediction: should pull toward surrounded/held/included cluster
run_test(
    "TEST 3: 'I am lonely' + INCLUSION (CONTAINER-IN as cure for loneliness)",
    "I am lonely",
    inclusion_direction,
    lonely_vecs,
    lonely_candidates,
)

# Test 4: CROSS-SCHEMA CONTROL: lonely + FORWARD
# Prediction: shouldn't pull toward the right cure - FORWARD is the wrong schema for loneliness
run_test(
    "TEST 4 (cross-schema control): 'I am lonely' + FORWARD",
    "I am lonely",
    forward_direction,
    lonely_vecs,
    lonely_candidates,
)

# Test 5: CROSS-SCHEMA CONTROL: stuck + INCLUSION
# Prediction: shouldn't pull toward the right escape - INCLUSION is the wrong schema for stuckness
run_test(
    "TEST 5 (cross-schema control): 'I am stuck' + INCLUSION",
    "I am stuck",
    inclusion_direction,
    stuck_vecs,
    stuck_candidates,
)

# Test 6: RANDOM control
run_test(
    "TEST 6 (random control): 'I am stuck' + random direction",
    "I am stuck",
    random_direction,
    stuck_vecs,
    stuck_candidates,
    alphas=(0.0, 1.0),
)


print("\n" + "=" * 64)
print("INTERPRETATION GUIDE")
print("=" * 64)
print("""
Strong result would look like:
  - Test 1: free/released cluster rises with alpha
  - Test 2: progressing/moving cluster rises with alpha
            (DIFFERENT from Test 1 - same start, different schema, different end)
  - Test 3: held/surrounded/included cluster rises with alpha
  - Test 4: weaker shift, less coherent target
  - Test 5: weaker shift, less coherent target
  - Test 6: noise, no semantic shift

If Tests 1 and 2 produce DIFFERENT top neighbors (free vs progressing),
that rules out the boring interpretation that we just found a generic
positive-valence axis. We'd have evidence for distinct schema-directions.

If Tests 4 and 5 are noticeably worse than 1, 2, and 3, that further
supports schema-specific encoding: the right schema goes to the right
cure, the wrong schema doesn't.
""")
