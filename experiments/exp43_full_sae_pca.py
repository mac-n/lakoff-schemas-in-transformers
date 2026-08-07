"""
exp43: Full SAE PCA with V + A + EXISTENCE + COHERENCE + SUCCESS + LOSS
sentence triples added to the existing 5 schema axes.

The point: exp41 found PC1 in SAE space = "legal-formal-register vs spatial-
origin-register" rather than a clean cognitive primitive. Niamh's hypothesis:
with only 5 schemas (UD/IO/FB/LD/LR) the variance is dominated by register
attractors because the 5 schemas don't pull strongly enough toward the
underlying cognitive primitives. Adding more axes that express the underlying
primitives directly (in non-Lakoff sentence triples) should let the cognitive
structure emerge from PCA.

We build sentence triples for:
  - VALENCE (high-V vs low-V states, no schema vocabulary)
  - AROUSAL (high-A vs low-A, valence-balanced)
  - EXISTENCE (coming-into-being vs going-out-of-being)
  - COHERENCE (expected/normal vs anomalous/aberrant)
  - SUCCESS_FAILURE (outcome-evaluation)
  - LOSS (gain vs deprivation)

Plus the existing 5 schema axes (UD/IO/FB/LD/LR) from exp22+exp25.

Total: 11 axes. PCA per layer at L2, L3, L4.

Encode all sentences through Pythia 70m residual-stream SAE.
"""
import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer


# ===================== TRIPLE DEFINITIONS =====================

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

SUCCESS_FAILURE_TRIPLES = [
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

# Original Lakoff schema triples from exp22 + exp25
# Just reload via exp26's SCHEMAS dict
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp26_phase_a_measurement_gaps import SCHEMAS, model, device, collect


# ===================== ENCODING + AXIS BUILDING =====================

def enc_via_sae(text, hook, sae):
    with torch.no_grad():
        return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)


def build_axis_from_triples(triples_list, hook, sae):
    """Build an axis from triple offsets: mean((pole_a - baseline) - (pole_c - baseline))
       = mean(pole_a - pole_c)."""
    offsets = []
    for (b, a, c) in triples_list:
        b_v = enc_via_sae(b, hook, sae)
        a_v = enc_via_sae(a, hook, sae)
        c_v = enc_via_sae(c, hook, sae)
        off_a = a_v - b_v
        off_c = c_v - b_v
        offsets.append(off_a - off_c)
    axis_raw = np.stack(offsets).mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    return axis_raw / nrm if nrm > 1e-12 else axis_raw


# ===================== MAIN LOOP =====================

# Focus on mid-layers where cluster structure was strongest in exp41
LAYERS_TO_RUN = [2, 3, 4]

per_layer_results = {}

for layer in LAYERS_TO_RUN:
    print(f"\n{'='*72}")
    print(f"LAYER {layer}")
    print(f"{'='*72}")

    hook = f"blocks.{layer}.hook_resid_post"
    sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
    sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

    axes = {}

    # Build the 5 Lakoff schema axes (just like exp41)
    for schema, (triples, name_a, name_c) in SCHEMAS.items():
        offsets_diff = []
        for (dom, b, a, c) in triples:
            b_v = enc_via_sae(b, hook, sae)
            a_v = enc_via_sae(a, hook, sae)
            c_v = enc_via_sae(c, hook, sae)
            off_a = a_v - b_v
            off_c = c_v - b_v
            offsets_diff.append(off_a - off_c)
        axis_raw = np.stack(offsets_diff).mean(axis=0)
        nrm = np.linalg.norm(axis_raw)
        axes[schema] = axis_raw / nrm if nrm > 1e-12 else axis_raw
        print(f"  built schema axis: {schema}")

    # Build the 6 additional axes (V, A, EXIST, COH, SUC, LOSS)
    for name, triples_list in [
        ("VALENCE", VALENCE_TRIPLES),
        ("AROUSAL", AROUSAL_TRIPLES),
        ("EXISTENCE", EXISTENCE_TRIPLES),
        ("COHERENCE", COHERENCE_TRIPLES),
        ("SUCCESS", SUCCESS_FAILURE_TRIPLES),
        ("LOSS", LOSS_TRIPLES),
    ]:
        axes[name] = build_axis_from_triples(triples_list, hook, sae)
        print(f"  built anchor axis: {name}")

    # PCA over the 11 axes
    axis_names = list(axes.keys())
    M = np.stack([axes[n] for n in axis_names])  # (11, 32k)

    pca = PCA(n_components=min(len(axis_names), 10))
    pca.fit(M)

    print(f"\n  Variance per PC: " + ", ".join(f"PC{i+1}={v*100:.1f}%"
                                              for i, v in enumerate(pca.explained_variance_ratio_)))

    print(f"\n  Top PC loadings:")
    for pc_idx in range(min(5, len(pca.components_))):
        pc = pca.components_[pc_idx]
        pc_unit = pc / np.linalg.norm(pc)
        loadings = [(n, float(axes[n] @ pc_unit)) for n in axis_names]
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%): " +
              ", ".join(f"{n}{l:+.2f}" for n, l in loadings[:6]))

    per_layer_results[layer] = {
        "axes": axes,
        "axis_names": axis_names,
        "pca_components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }

    del sae


# ===================== SAVE =====================
torch.save(per_layer_results, "/Users/macn/Documents/embeddingexp/exp43_results.pt")
print("\nSaved: exp43_results.pt")


# ===================== NEURONPEDIA LOOKUP =====================
print(f"\n\n{'='*72}")
print("NEURONPEDIA LOOKUP — what do the PCs semantically represent?")
print(f"{'='*72}")

NEURONPEDIA_MODEL = "pythia-70m-deduped"
API_DELAY = 0.3


def fetch_feature(np_sae, feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{np_sae}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except json.JSONDecodeError:
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            return json.loads(clean, strict=False)
    except Exception as e:
        return {"_error": str(e)}


def feat_desc(d):
    if d is None or "_error" in d:
        return "[no data]"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:160]
    return "(no description)"


def top_features(pc_vec, top_n=10):
    abs_loads = np.abs(pc_vec)
    sorted_idx = np.argsort(abs_loads)[::-1]
    positives, negatives = [], []
    for idx in sorted_idx:
        if pc_vec[idx] > 0 and len(positives) < top_n:
            positives.append((int(idx), float(pc_vec[idx])))
        elif pc_vec[idx] < 0 and len(negatives) < top_n:
            negatives.append((int(idx), float(pc_vec[idx])))
        if len(positives) >= top_n and len(negatives) >= top_n:
            break
    return positives, negatives


for layer in LAYERS_TO_RUN:
    np_sae = f"{layer}-res-sm"
    pca = per_layer_results[layer]
    print(f"\n\n--- LAYER {layer} ({np_sae}) ---")
    for pc_idx in range(min(3, len(pca["pca_components"]))):
        pc = pca["pca_components"][pc_idx]
        var = pca["explained_variance_ratio"][pc_idx] * 100
        pos, neg = top_features(pc, top_n=8)
        print(f"\n  PC{pc_idx+1} ({var:.1f}% variance)")
        print(f"  Positive pole:")
        for f_idx, ld in pos:
            d = fetch_feature(np_sae, f_idx)
            print(f"    feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
            time.sleep(API_DELAY)
        print(f"  Negative pole:")
        for f_idx, ld in neg:
            d = fetch_feature(np_sae, f_idx)
            print(f"    feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
            time.sleep(API_DELAY)

print("\nDone.")
