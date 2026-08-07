"""
exp50: Matched-substrate PCA comparison using the FORMAL sentence triples
from exp43. Parallel to exp49 (which used diverse triples).

Same methodology:
  - For each sentence in each triple, encode via w2v (mean-pool word vectors)
    AND via Pythia 70m SAE (residual-stream L3)
  - Build axis offsets in each substrate
  - PCA each
  - Compare PC loadings cross-substrate

Comparing exp49 vs exp50 results tells us: does the formal-vs-diverse choice
affect the cross-substrate alignment? If one set gives cleaner alignment,
we've identified a sentence-construction sensitivity that matters for the
substrate-comparison methodology.
"""
import numpy as np
import gensim.downloader as api
import torch
from sae_lens import SAE
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp26_phase_a_measurement_gaps import model, device, collect, SCHEMAS


# FORMAL triples from exp43
VALENCE_TRIPLES = [
    ("The day was ordinary.", "The day was delightful.", "The day was awful."),
    ("The experience was unremarkable.", "The experience was pleasant.", "The experience was unpleasant."),
    ("The result was neutral.", "The result was beneficial.", "The result was harmful."),
    ("The outcome was standard.", "The outcome was wonderful.", "The outcome was terrible."),
    ("The event was unmemorable.", "The event was excellent.", "The event was dreadful."),
    ("The encounter was average.", "The encounter was satisfying.", "The encounter was frustrating."),
    ("The visit was uneventful.", "The visit was enjoyable.", "The visit was distasteful."),
    ("The arrangement was usual.", "The arrangement was favorable.", "The arrangement was unfavorable."),
    ("The decision was routine.", "The decision was agreeable.", "The decision was disagreeable."),
    ("The conversation was bland.", "The conversation was charming.", "The conversation was tedious."),
    ("The meal was passable.", "The meal was exquisite.", "The meal was repulsive."),
    ("The performance was middling.", "The performance was magnificent.", "The performance was atrocious."),
]
AROUSAL_TRIPLES = [
    ("The atmosphere was steady.", "The atmosphere was intense.", "The atmosphere was mild."),
    ("The moment was unchanged.", "The moment was urgent.", "The moment was leisurely."),
    ("The situation was stable.", "The situation was frantic.", "The situation was tranquil."),
    ("The state remained.", "The state was alert.", "The state was drowsy."),
    ("The energy was constant.", "The energy was electric.", "The energy was placid."),
    ("The mood was even.", "The mood was sharp.", "The mood was dull."),
    ("The scene was static.", "The scene was vivid.", "The scene was faint."),
    ("The crowd was settled.", "The crowd was turbulent.", "The crowd was still."),
    ("The room felt regular.", "The room felt energetic.", "The room felt lethargic."),
    ("The pace was unchanging.", "The pace was acute.", "The pace was subtle."),
]
EXISTENCE_TRIPLES = [
    ("The institution continued.", "The institution was founded.", "The institution was abolished."),
    ("The structure persisted.", "The structure was built.", "The structure was demolished."),
    ("The pattern remained.", "The pattern emerged.", "The pattern vanished."),
    ("The phenomenon persisted.", "The phenomenon arose.", "The phenomenon dissipated."),
    ("The community endured.", "The community was established.", "The community was abandoned."),
    ("The framework continued.", "The framework was created.", "The framework was destroyed."),
    ("The system stayed.", "The system was assembled.", "The system was disassembled."),
    ("The figure was present.", "The figure was born.", "The figure died."),
    ("The image stayed.", "The image appeared.", "The image disappeared."),
    ("The form remained.", "The form materialized.", "The form vanished."),
    ("The organization continued.", "The organization was originated.", "The organization was terminated."),
    ("The cycle persisted.", "The cycle began.", "The cycle ended."),
]
COHERENCE_TRIPLES = [
    ("The behavior was typical.", "The behavior was consistent.", "The behavior was inconsistent."),
    ("The result was expected.", "The result was predictable.", "The result was surprising."),
    ("The arrangement was usual.", "The arrangement was ordered.", "The arrangement was disordered."),
    ("The structure was middling.", "The structure was organized.", "The structure was disorganized."),
    ("The pattern was generic.", "The pattern was regular.", "The pattern was irregular."),
    ("The event was unremarkable.", "The event was ordinary.", "The event was anomalous."),
    ("The melody was standard.", "The melody was harmonious.", "The melody was discordant."),
    ("The system was unchanged.", "The system was aligned.", "The system was misaligned."),
    ("The text was ordinary.", "The text was coherent.", "The text was incoherent."),
    ("The state was steady.", "The state was uniform.", "The state was erratic."),
    ("The reaction was usual.", "The reaction was normal.", "The reaction was aberrant."),
]
SUCCESS_TRIPLES = [
    ("The attempt was ongoing.", "The attempt succeeded.", "The attempt failed."),
    ("The effort continued.", "The effort accomplished its goal.", "The effort fell short."),
    ("The plan was active.", "The plan triumphed.", "The plan was defeated."),
    ("The bid was pending.", "The bid won.", "The bid lost."),
    ("The trial proceeded.", "The trial was a victory.", "The trial was a defeat."),
    ("The shot was prepared.", "The shot scored.", "The shot missed."),
    ("The exam was administered.", "The exam was passed.", "The exam was failed."),
    ("The mission was running.", "The mission was accomplished.", "The mission collapsed."),
    ("The proposal was made.", "The proposal was approved.", "The proposal was rejected."),
    ("The candidacy was active.", "The candidacy was victorious.", "The candidacy was unsuccessful."),
    ("The reasoning was ongoing.", "The reasoning was correct.", "The reasoning was incorrect."),
]
LOSS_TRIPLES = [
    ("The resource was available.", "The resource was acquired.", "The resource was depleted."),
    ("The treasury was steady.", "The treasury showed profit.", "The treasury showed loss."),
    ("The supply was even.", "The supply was abundant.", "The supply was scarce."),
    ("The portfolio held.", "The portfolio gained.", "The portfolio lost."),
    ("The harvest was usual.", "The harvest was bountiful.", "The harvest failed."),
    ("The household had means.", "The household had wealth.", "The household had poverty."),
    ("The situation was safe.", "The situation was secure.", "The situation was endangered."),
    ("The area was protected.", "The area was guarded.", "The area was exposed."),
    ("The person was steady.", "The person was protected.", "The person was vulnerable."),
    ("The family was stable.", "The family was affluent.", "The family was destitute."),
    ("The stockpile was steady.", "The stockpile had surplus.", "The stockpile had deficit."),
]

ANCHOR_TRIPLES = {
    "VALENCE": VALENCE_TRIPLES,
    "AROUSAL": AROUSAL_TRIPLES,
    "EXISTENCE": EXISTENCE_TRIPLES,
    "COHERENCE": COHERENCE_TRIPLES,
    "SUCCESS": SUCCESS_TRIPLES,
    "LOSS": LOSS_TRIPLES,
}


print("Loading GloVe-300...")
wv = api.load("glove-wiki-gigaword-300")


def encode_sentence_w2v(sentence):
    import re
    words = re.findall(r"[a-z]+", sentence.lower())
    vecs = [wv[w] for w in words if w in wv.key_to_index]
    if not vecs:
        return np.zeros(wv.vector_size)
    return np.mean(vecs, axis=0)


def build_w2v_axis(triples_list, has_domain=False):
    offsets = []
    for triple in triples_list:
        if has_domain:
            _, b, a, c = triple
        else:
            b, a, c = triple
        b_v = encode_sentence_w2v(b)
        a_v = encode_sentence_w2v(a)
        c_v = encode_sentence_w2v(c)
        offsets.append((a_v - b_v) - (c_v - b_v))
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


LAYER = 3
hook = f"blocks.{LAYER}.hook_resid_post"
print(f"Loading SAE at layer {LAYER}...")
sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res


def encode_sentence_sae(sentence):
    with torch.no_grad():
        return sae.encode(collect(sentence, hook)).max(0).values.numpy().astype(np.float64)


def build_sae_axis(triples_list, has_domain=False):
    offsets = []
    for triple in triples_list:
        if has_domain:
            _, b, a, c = triple
        else:
            b, a, c = triple
        b_v = encode_sentence_sae(b)
        a_v = encode_sentence_sae(a)
        c_v = encode_sentence_sae(c)
        offsets.append((a_v - b_v) - (c_v - b_v))
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


print("\n=== Building w2v axes from FORMAL sentences ===")
w2v_axes = {}
for name, triples in ANCHOR_TRIPLES.items():
    w2v_axes[name] = build_w2v_axis(triples)
    print(f"  w2v: {name}")

print("\n=== Building w2v Lakoff schemas (same as before — these are the schema triples) ===")
for schema, (triples, _, _) in SCHEMAS.items():
    w2v_axes[schema] = build_w2v_axis(triples, has_domain=True)
    print(f"  w2v: {schema}")

print("\n=== Building SAE axes from FORMAL sentences ===")
sae_axes = {}
for name, triples in ANCHOR_TRIPLES.items():
    sae_axes[name] = build_sae_axis(triples)
    print(f"  SAE: {name}")

print("\n=== Building SAE Lakoff schemas ===")
for schema, (triples, _, _) in SCHEMAS.items():
    sae_axes[schema] = build_sae_axis(triples, has_domain=True)
    print(f"  SAE: {schema}")


axis_names = list(w2v_axes.keys())
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


print(f"\n=== w2v PC loadings (FORMAL sentences) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in axis_names))
for i in range(min(6, len(w2v_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in w2v_pc_loadings[i]))

print(f"\n=== SAE PC loadings (FORMAL sentences) ===")
print(f"  PC  " + "  ".join(f"{n[:9]:>9}" for n in axis_names))
for i in range(min(6, len(sae_pc_loadings))):
    print(f"  PC{i+1}  " + "  ".join(f"{l:>+9.3f}" for l in sae_pc_loadings[i]))


print(f"\n=== Cross-substrate cosine matrix (PC loadings) ===")
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
}, "/Users/macn/Documents/embeddingexp/exp50_results.pt")
print("\nSaved: exp50_results.pt")
