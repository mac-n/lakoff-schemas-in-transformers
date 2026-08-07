"""
exp27: Phase B — word2vec substrate comparison for Lakoff schema axes.

The point: in word2vec / GloVe, what falls out is pure linguistic-distributional
structure. No transformer machinery, no next-token-prediction compute. If Lakoff
schemas exist as primitives in HUMAN language (encoded through which words
appear near which others), they should show up here as coherent geometric axes.

If word2vec gives clean Lakoff structure (UD/LD/FB align positively as polar
contrast axes, IO orthogonal, LR weak/orthogonal) and Pythia-SAE gives the
salience-dominated structure from exp22+exp26 (only UD is truly polar, others
load asymmetrically onto a salience direction), that's the cooking case:
linguistic primitives are Lakoff-shaped, computation reorganizes them.

For each schema, axis is constructed as: A = mean(w[pole_a] - w[pole_b]) across
matched word pairs spanning multiple Lakoff children (literal + metaphorical
extensions). Then compute pairwise cosines between axes, compare to exp22's
SAE-substrate matrix.
"""
import numpy as np
import gensim.downloader as api

MODEL_NAME = "glove-wiki-gigaword-300"
print(f"Loading {MODEL_NAME} (~400MB, downloads on first run)...")
wv = api.load(MODEL_NAME)
print(f"Loaded: {len(wv.key_to_index)} words, {wv.vector_size}-dim")


# ============================== SCHEMA WORD PAIRS ==============================
# Matched (pole_a, pole_b) pairs spanning multiple Lakoff children per schema.
# Single words only (word2vec is word-level).

UD_PAIRS = [
    # Literal vertical motion (pole_a = UP, pole_b = DOWN)
    ("rose", "fell"), ("rising", "falling"), ("climbing", "descending"),
    ("ascended", "descended"), ("soaring", "plunging"), ("lifted", "sunk"),
    ("higher", "lower"), ("upward", "downward"), ("up", "down"),
    # HAPPY IS UP
    ("happy", "sad"), ("cheerful", "gloomy"), ("elated", "dejected"),
    ("joyful", "sorrowful"), ("uplifted", "depressed"),
    # MORE IS UP
    ("more", "less"), ("increase", "decrease"), ("grew", "shrank"),
    ("expanded", "contracted"), ("multiplied", "dwindled"),
    # HIGH STATUS IS UP
    ("promoted", "demoted"), ("prestigious", "disgraced"),
    ("esteemed", "despised"), ("eminent", "lowly"),
    # HEALTHY IS UP
    ("healthy", "sick"), ("thriving", "ailing"), ("strong", "weak"),
    ("vigorous", "feeble"), ("robust", "frail"),
]

IO_PAIRS = [
    # Literal containment (pole_a = IN, pole_b = OUT)
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    # Mind containment
    ("remembered", "forgotten"), ("retained", "dismissed"),
    # Relationship
    ("married", "divorced"), ("engaged", "estranged"), ("together", "apart"),
    # Difficulty
    ("trapped", "escaped"), ("stranded", "rescued"), ("captured", "freed"),
    # Group membership
    ("included", "excluded"), ("admitted", "expelled"),
]

FB_PAIRS = [
    # Literal motion (pole_a = FORWARD, pole_b = BACK)
    ("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"),
    ("forwards", "backwards"),
    # Progress
    ("progress", "regress"), ("improved", "declined"), ("gained", "lost"),
    ("advanced", "retreated"),
    # Time
    ("future", "past"), ("next", "previous"), ("later", "earlier"),
    ("upcoming", "former"),
    # Development
    ("evolved", "devolved"), ("developed", "regressed"),
    # Journey
    ("onward", "return"),
]

LD_PAIRS = [
    # Literal illumination (pole_a = LIGHT, pole_b = DARK)
    ("bright", "dark"), ("light", "dark"), ("illuminated", "shadowed"),
    ("radiant", "dim"), ("sunny", "gloomy"), ("shining", "dim"),
    ("luminous", "shadowy"), ("glowing", "darkened"),
    # Clarity
    ("clear", "obscure"), ("lucid", "murky"), ("transparent", "opaque"),
    ("obvious", "hidden"),
    # Hope
    ("hopeful", "hopeless"), ("optimistic", "pessimistic"),
    # Goodness
    ("good", "evil"), ("pure", "tainted"), ("noble", "wicked"),
    ("virtuous", "sinful"), ("righteous", "corrupt"),
    # Knowledge
    ("enlightened", "ignorant"), ("informed", "uninformed"),
    ("known", "unknown"), ("revealed", "concealed"),
]

LR_PAIRS = [
    # Spatial
    ("left", "right"), ("leftward", "rightward"),
    # Nautical
    ("port", "starboard"),
    # Political
    ("liberal", "conservative"),
]


SCHEMAS = {
    "UP-DOWN":      UD_PAIRS,
    "IN-OUT":       IO_PAIRS,
    "FORWARD-BACK": FB_PAIRS,
    "LIGHT-DARK":   LD_PAIRS,
    "LEFT-RIGHT":   LR_PAIRS,
}


# ============================== BUILD AXES ==============================
def build_axis(pairs, schema_name):
    offsets = []
    missing = []
    for a, c in pairs:
        if a not in wv.key_to_index:
            missing.append(a)
            continue
        if c not in wv.key_to_index:
            missing.append(c)
            continue
        v = wv[a] - wv[c]
        offsets.append(v)
    if missing:
        print(f"  [{schema_name}] missing from vocab: {missing}")
    if not offsets:
        return None, 0, []
    offsets = np.stack(offsets)
    axis_raw = offsets.mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    axis = axis_raw / nrm if nrm > 1e-12 else axis_raw
    return axis, len(offsets), offsets


print(f"\n=== Building schema axes from word pairs ===")
axes = {}
all_offsets = {}
for name, pairs in SCHEMAS.items():
    axis, n_used, offsets = build_axis(pairs, name)
    axes[name] = axis
    all_offsets[name] = offsets
    print(f"  {name}: built from {n_used} pairs")


# ============================== WITHIN-PAIR COHERENCE ==============================
print(f"\n=== Within-schema cross-pair coherence (do pairs within a schema agree?) ===")
print(f"  (mean pairwise cosine of normalized offsets within each schema; high = coherent)")
for name, offsets in all_offsets.items():
    if len(offsets) < 2:
        continue
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    unit = offsets / norms
    sims = unit @ unit.T
    iu = np.triu_indices_from(sims, k=1)
    mean_within = float(sims[iu].mean())
    print(f"  {name}: mean within-schema cos = {mean_within:+.4f}  ({len(offsets)} pairs)")


# ============================== PAIRWISE COSINES ==============================
print(f"\n=== Pairwise cosine matrix between schema axes (word2vec) ===")
names = list(SCHEMAS.keys())
M = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M[i, j] = float(axes[ni] @ axes[nj])

# Header
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names))
for i, ni in enumerate(names):
    print(f"  {ni:>14}" + "".join(f"  {M[i, j]:>+13.4f}" for j in range(len(names))))


# ============================== COMPARISON TO EXP22 (Pythia 70m res-sm L3) ==============================
# exp22 layer 3 matrix (from results_exp22_multi_schema_orthogonality.md or our running summary):
# Used the same names order, but only had UD, IO, FB, LD, TUE-WED (sham was tue-wed, not LR).
# Reconstructing from exp22 numbers at layer 3:
exp22_L3 = {
    ("UP-DOWN", "UP-DOWN"):       1.00,
    ("UP-DOWN", "IN-OUT"):       -0.14,
    ("UP-DOWN", "FORWARD-BACK"): +0.40,
    ("UP-DOWN", "LIGHT-DARK"):   +0.59,
    ("IN-OUT", "IN-OUT"):         1.00,
    ("IN-OUT", "FORWARD-BACK"): -0.07,
    ("IN-OUT", "LIGHT-DARK"):   -0.22,
    ("FORWARD-BACK", "FORWARD-BACK"): 1.00,
    ("FORWARD-BACK", "LIGHT-DARK"):  +0.44,
    ("LIGHT-DARK", "LIGHT-DARK"):    1.00,
}

# LR-vs-others at L3 from exp25 results:
# cos(LR, UD) = +0.24 at L3-L4, cos(LR, IO) ≈ 0, etc. — exp25 file would have the rest.

print(f"\n=== SAE-substrate comparison (Pythia 70m res-sm L3, from exp22 + exp25) ===")
print(f"  (negative or near-zero = orthogonal between schemas; positive = aligned)")
sae_pairs = [
    ("UP-DOWN", "IN-OUT", -0.14),
    ("UP-DOWN", "FORWARD-BACK", +0.40),
    ("UP-DOWN", "LIGHT-DARK", +0.59),
    ("UP-DOWN", "LEFT-RIGHT", +0.24),  # from exp25
    ("IN-OUT", "FORWARD-BACK", -0.07),
    ("IN-OUT", "LIGHT-DARK", -0.22),
    ("FORWARD-BACK", "LIGHT-DARK", +0.44),
]

print(f"\n  {'pair':>30}  {'word2vec':>10}  {'SAE L3':>10}  {'difference':>11}  {'verdict':>10}")
for a, b, sae_val in sae_pairs:
    i, j = names.index(a), names.index(b)
    w2v_val = M[i, j]
    diff = w2v_val - sae_val
    if abs(w2v_val - sae_val) < 0.1:
        verdict = "same"
    elif abs(w2v_val) < 0.15 and abs(sae_val) > 0.3:
        verdict = "SAE-only"
    elif abs(sae_val) < 0.15 and abs(w2v_val) > 0.3:
        verdict = "w2v-only"
    else:
        verdict = "different"
    print(f"  {a + ' vs ' + b:>30}  {w2v_val:>+10.4f}  {sae_val:>+10.4f}  {diff:>+11.4f}  {verdict:>10}")


# ============================== INTERPRETATION ==============================
print(f"\n=== Quick reading guide ===")
print(f"  'Cooking case' (w2v Lakoff-shaped, SAE different):")
print(f"    expect w2v UD-LD to be MORE polar / less collapsed onto common direction")
print(f"    expect w2v IO to remain orthogonal in both substrates")
print(f"    expect w2v LR to be orthogonal to UD (sym pole, no shared direction)")
print(f"")
print(f"  'Primitives are linguistic' (same shape both substrates):")
print(f"    cos pairs largely agree between word2vec and SAE")
print(f"    same alignment patterns (UD-LD-FB cluster, IO orthogonal)")

# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp27_results.npz",
    cosine_matrix=M,
    names=names,
)
print(f"\nSaved: exp27_results.npz")
