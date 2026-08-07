"""
exp49: Proper substrate comparison with MATCHED inputs.

exp48 compared w2v PCA (from word-pair offsets) to SAE PCA (from sentence-
triple offsets) — methodologically asymmetric. This experiment uses the SAME
sentence triples as input to both substrates, building axes from sentence-
offset differences in each substrate's space, then PCA-comparing.

For w2v sentence encoding: mean-pool the word vectors in each sentence
(standard kludge for sentence-level w2v). For SAE: encode through Pythia 70m
SAE at L3 (matches exp45's setup).

Both substrates now see the same input data. Any remaining PC differences are
substrate-specific representational reorganization.
"""
import numpy as np
import gensim.downloader as api
import torch
from sae_lens import SAE
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp26_phase_a_measurement_gaps import model, device, collect, SCHEMAS


# ========== TRIPLE DEFINITIONS (same as exp45) ==========
VALENCE_TRIPLES = [
    ("The soup tasted ordinary.", "The soup tasted delicious.", "The soup tasted awful."),
    ("The day was overcast.", "The day was beautiful.", "The day was miserable."),
    ("The song was forgettable.", "The song was wonderful.", "The song was terrible."),
    ("The fabric felt average.", "The fabric felt luxurious.", "The fabric felt scratchy."),
    ("The air smelled neutral.", "The air smelled fragrant.", "The air smelled foul."),
    ("The party was unremarkable.", "The party was delightful.", "The party was dreadful."),
    ("The sunset was muted.", "The sunset was gorgeous.", "The sunset was dull."),
    ("The music played.", "The music was lovely.", "The music was grating."),
    ("The pillow felt ordinary.", "The pillow felt soft.", "The pillow felt rough."),
    ("She felt fine.", "She felt joyful.", "She felt miserable."),
    ("The wine was passable.", "The wine was exquisite.", "The wine was vile."),
    ("The garden was usual.", "The garden was charming.", "The garden was dismal."),
]
AROUSAL_TRIPLES = [
    ("Her heart beat normally.", "Her heart raced.", "Her heart slowed."),
    ("The river flowed.", "The river raged.", "The river trickled."),
    ("The fire burned.", "The fire roared.", "The fire smoldered."),
    ("She breathed evenly.", "She breathed sharply.", "She breathed softly."),
    ("The crowd murmured.", "The crowd erupted.", "The crowd hushed."),
    ("The wind blew.", "The wind howled.", "The wind whispered."),
    ("He walked.", "He bolted.", "He ambled."),
    ("The dog watched.", "The dog charged.", "The dog dozed."),
    ("She drank coffee.", "She gulped coffee.", "She sipped coffee."),
    ("Music played.", "Music blasted.", "Music murmured."),
    ("Rain fell.", "Rain poured.", "Rain drizzled."),
]
EXISTENCE_TRIPLES = [
    ("The seed lay there.", "The seed sprouted.", "The seed rotted."),
    ("The candle stood.", "The candle was lit.", "The candle was extinguished."),
    ("A baby lay quietly.", "A baby was born.", "A grandparent died."),
    ("The flower was a bud.", "The flower bloomed.", "The flower wilted."),
    ("The page was blank.", "She wrote a poem.", "She tore up the poem."),
    ("The cup sat empty.", "She filled the cup.", "She emptied the cup."),
    ("The room was quiet.", "Guests arrived.", "Guests departed."),
    ("The pot was cold.", "The water boiled.", "The water cooled."),
    ("Two people stood apart.", "Two people met.", "Two people parted ways."),
    ("The instrument was silent.", "Music began.", "Music ended."),
    ("The lot was empty.", "A house went up.", "The house was torn down."),
    ("Yarn lay in piles.", "She knitted a sweater.", "She unraveled the sweater."),
]
COHERENCE_TRIPLES = [
    ("She showed up.", "She showed up on time.", "She showed up at strange hours."),
    ("The cat sat.", "The cat purred peacefully.", "The cat hissed wildly."),
    ("Rain fell.", "Rain fell steadily.", "Rain fell in odd bursts."),
    ("He greeted me.", "He greeted me warmly.", "He greeted me strangely."),
    ("The river ran.", "The river flowed evenly.", "The river churned chaotically."),
    ("The clock ticked.", "The clock ticked steadily.", "The clock ticked unevenly."),
    ("Birds chirped.", "Birds chirped at dawn.", "Birds chirped at midnight."),
    ("The dog barked.", "The dog barked at strangers.", "The dog barked at nothing."),
    ("The path wound.", "The path wound smoothly.", "The path wound bizarrely."),
    ("The flame burned.", "The flame burned steadily.", "The flame flickered erratically."),
    ("She acted naturally.", "She acted as expected.", "She acted strangely."),
]
SUCCESS_TRIPLES = [
    ("She practiced.", "She nailed the routine.", "She fumbled the routine."),
    ("He fished.", "He caught a big one.", "He came back empty-handed."),
    ("The team played.", "The team won.", "The team lost."),
    ("The chef cooked.", "The dish came out perfectly.", "The dish burned."),
    ("She studied.", "She aced the test.", "She flunked the test."),
    ("He aimed.", "He hit the target.", "He missed wildly."),
    ("She climbed.", "She reached the summit.", "She fell off the cliff."),
    ("The cake baked.", "The cake rose beautifully.", "The cake collapsed."),
    ("He ran the race.", "He crossed the finish line first.", "He stumbled at the start."),
    ("She invested.", "Her stocks grew.", "Her stocks crashed."),
    ("She gardened.", "Her vegetables flourished.", "Her vegetables withered."),
]
LOSS_TRIPLES = [
    ("He had a roof.", "He bought a new house.", "He lost his home."),
    ("She has a job.", "She got promoted.", "She got fired."),
    ("The bird nested.", "The bird thrived.", "The bird starved."),
    ("The dog had owners.", "The dog had a loving family.", "The dog was abandoned."),
    ("He held the cup.", "He grasped the cup firmly.", "He dropped the cup."),
    ("The town was quiet.", "The town was peaceful.", "The town was attacked."),
    ("The neighborhood existed.", "The neighborhood prospered.", "The neighborhood declined."),
    ("She had a key.", "She kept the key safe.", "She lost the key."),
    ("He had savings.", "He multiplied his savings.", "He gambled away his savings."),
    ("She wrote a letter.", "She kept the letter.", "She burned the letter."),
    ("Coins lay there.", "He found treasure.", "He was robbed."),
]


ANCHOR_TRIPLES = {
    "VALENCE": VALENCE_TRIPLES,
    "AROUSAL": AROUSAL_TRIPLES,
    "EXISTENCE": EXISTENCE_TRIPLES,
    "COHERENCE": COHERENCE_TRIPLES,
    "SUCCESS": SUCCESS_TRIPLES,
    "LOSS": LOSS_TRIPLES,
}


# ========== W2V SENTENCE ENCODING ==========
print("Loading GloVe-300...")
wv = api.load("glove-wiki-gigaword-300")
print("Loaded.")


def encode_sentence_w2v(sentence):
    """Mean-pool word embeddings (skip OOV)."""
    # Lowercase + strip punctuation
    import re
    words = re.findall(r"[a-z]+", sentence.lower())
    vecs = [wv[w] for w in words if w in wv.key_to_index]
    if not vecs:
        return np.zeros(wv.vector_size)
    return np.mean(vecs, axis=0)


def build_w2v_axis_from_triples(triples_list):
    offsets = []
    for (b, a, c) in triples_list:
        b_v = encode_sentence_w2v(b)
        a_v = encode_sentence_w2v(a)
        c_v = encode_sentence_w2v(c)
        off_a = a_v - b_v
        off_c = c_v - b_v
        offsets.append(off_a - off_c)
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


def build_w2v_axis_from_lakoff(triples_with_domain):
    offsets = []
    for (dom, b, a, c) in triples_with_domain:
        b_v = encode_sentence_w2v(b)
        a_v = encode_sentence_w2v(a)
        c_v = encode_sentence_w2v(c)
        off_a = a_v - b_v
        off_c = c_v - b_v
        offsets.append(off_a - off_c)
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


# ========== SAE SENTENCE ENCODING ==========
LAYER = 3
hook = f"blocks.{LAYER}.hook_resid_post"
print(f"Loading SAE at layer {LAYER}...")
sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res


def encode_sentence_sae(sentence):
    with torch.no_grad():
        return sae.encode(collect(sentence, hook)).max(0).values.numpy().astype(np.float64)


def build_sae_axis_from_triples(triples_list):
    offsets = []
    for (b, a, c) in triples_list:
        b_v = encode_sentence_sae(b)
        a_v = encode_sentence_sae(a)
        c_v = encode_sentence_sae(c)
        off_a = a_v - b_v
        off_c = c_v - b_v
        offsets.append(off_a - off_c)
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


def build_sae_axis_from_lakoff(triples_with_domain):
    offsets = []
    for (dom, b, a, c) in triples_with_domain:
        b_v = encode_sentence_sae(b)
        a_v = encode_sentence_sae(a)
        c_v = encode_sentence_sae(c)
        off_a = a_v - b_v
        off_c = c_v - b_v
        offsets.append(off_a - off_c)
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


# ========== BUILD AXES IN BOTH SUBSTRATES ==========
print("\n=== Building w2v axes from matched sentences ===")
w2v_axes = {}
for name, triples in ANCHOR_TRIPLES.items():
    w2v_axes[name] = build_w2v_axis_from_triples(triples)
    print(f"  w2v: {name}")

print("\n=== Building w2v Lakoff schemas from sentences ===")
for schema, (triples, _, _) in SCHEMAS.items():
    w2v_axes[schema] = build_w2v_axis_from_lakoff(triples)
    print(f"  w2v: {schema}")

print("\n=== Building SAE axes from matched sentences ===")
sae_axes = {}
for name, triples in ANCHOR_TRIPLES.items():
    sae_axes[name] = build_sae_axis_from_triples(triples)
    print(f"  SAE: {name}")

print("\n=== Building SAE Lakoff schemas from sentences ===")
for schema, (triples, _, _) in SCHEMAS.items():
    sae_axes[schema] = build_sae_axis_from_lakoff(triples)
    print(f"  SAE: {schema}")


# ========== PCA EACH ==========
axis_names = list(w2v_axes.keys())  # should match sae_axes keys
assert set(axis_names) == set(sae_axes.keys())

print(f"\nAxes (n={len(axis_names)}): {axis_names}")

M_w2v = np.stack([w2v_axes[n] for n in axis_names])
M_sae = np.stack([sae_axes[n] for n in axis_names])

pca_w2v = PCA(n_components=min(len(axis_names), 10))
pca_sae = PCA(n_components=min(len(axis_names), 10))
pca_w2v.fit(M_w2v)
pca_sae.fit(M_sae)

print(f"\nw2v variance: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_w2v.explained_variance_ratio_)))
print(f"SAE variance: " + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_sae.explained_variance_ratio_)))


def pc_loadings(component_vec, axes_dict, names):
    pc_unit = component_vec / np.linalg.norm(component_vec)
    return np.array([float(axes_dict[n] @ pc_unit) for n in names])


w2v_pc_loadings = np.array([pc_loadings(pca_w2v.components_[i], w2v_axes, axis_names) for i in range(min(8, len(pca_w2v.components_)))])
sae_pc_loadings = np.array([pc_loadings(pca_sae.components_[i], sae_axes, axis_names) for i in range(min(8, len(pca_sae.components_)))])


print(f"\n=== w2v PC loadings (matched sentences) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in axis_names))
for i in range(min(6, len(w2v_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in w2v_pc_loadings[i]))

print(f"\n=== SAE PC loadings (matched sentences) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in axis_names))
for i in range(min(6, len(sae_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in sae_pc_loadings[i]))


# ========== CROSS-SUBSTRATE COSINE MATRIX ==========
print(f"\n=== Cross-substrate cosine similarity (PC loadings on shared axes) ===")
print(f"  {'':>10}  " + "  ".join(f"SAE-PC{j+1:>2}" for j in range(min(6, len(sae_pc_loadings)))))
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i] / np.linalg.norm(w2v_pc_loadings[i])
    row = []
    for j in range(min(6, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j] / np.linalg.norm(sae_pc_loadings[j])
        row.append(f"{float(w_vec @ s_vec):>+8.3f}")
    var_pct = pca_w2v.explained_variance_ratio_[i] * 100
    print(f"  w2v-PC{i+1} ({var_pct:>4.1f}%)  " + "  ".join(row))

print(f"\nBest match per w2v PC:")
for i in range(min(6, len(w2v_pc_loadings))):
    w_vec = w2v_pc_loadings[i] / np.linalg.norm(w2v_pc_loadings[i])
    matches = []
    for j in range(min(8, len(sae_pc_loadings))):
        s_vec = sae_pc_loadings[j] / np.linalg.norm(sae_pc_loadings[j])
        matches.append((j+1, float(w_vec @ s_vec), pca_sae.explained_variance_ratio_[j]*100))
    matches.sort(key=lambda x: abs(x[1]), reverse=True)
    best_j, best_cos, best_var = matches[0]
    print(f"  w2v-PC{i+1} → SAE-PC{best_j} (cos={best_cos:+.3f}, SAE-PC{best_j} captures {best_var:.1f}%)")


torch.save({
    "w2v_axes": w2v_axes,
    "sae_axes": sae_axes,
    "axis_names": axis_names,
    "w2v_components": pca_w2v.components_,
    "sae_components": pca_sae.components_,
    "w2v_var": pca_w2v.explained_variance_ratio_,
    "sae_var": pca_sae.explained_variance_ratio_,
}, "/Users/macn/Documents/embeddingexp/exp49_results.pt")
print("\nSaved: exp49_results.pt")
