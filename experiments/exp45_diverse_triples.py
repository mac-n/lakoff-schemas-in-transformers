"""
exp45: SAE PCA with vocabulary-diverse sentence triples.

The exp44 polarity check showed our exp43 sentence triples activate the
formal-legal-cluster of SAE features asymmetrically, contaminating the axes
and biasing PC1 toward formal-procedural-text vocabulary even though the
underlying axis is Russell's affect diagonal.

This experiment uses redesigned triples drawing from domestic, sensory,
embodied, nature, and concrete-action domains. Less "the institution was
founded/abolished," more "the seed sprouted/rotted." The hypothesis: PC1 will
still emerge as Russell's diagonal (substrate-invariance is real) but the
SAE features expressing it will be different — less formal-legal, more
concrete-sensory or general-affective.

Just one layer (L3) for speed, where the cluster structure was strongest.
"""
import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp26_phase_a_measurement_gaps import SCHEMAS, model, device, collect


# ===================== DIVERSE-DOMAIN TRIPLES =====================
# Designed to avoid formal-passive-legal sentence structure.
# Each axis has triples spanning domestic, sensory, bodily, nature, action domains.

VALENCE_TRIPLES = [
    # Food
    ("The soup tasted ordinary.", "The soup tasted delicious.", "The soup tasted awful."),
    # Weather
    ("The day was overcast.", "The day was beautiful.", "The day was miserable."),
    # Music
    ("The song was forgettable.", "The song was wonderful.", "The song was terrible."),
    # Texture
    ("The fabric felt average.", "The fabric felt luxurious.", "The fabric felt scratchy."),
    # Smell
    ("The air smelled neutral.", "The air smelled fragrant.", "The air smelled foul."),
    # Social
    ("The party was unremarkable.", "The party was delightful.", "The party was dreadful."),
    # Color
    ("The sunset was muted.", "The sunset was gorgeous.", "The sunset was dull."),
    # Sound
    ("The music played.", "The music was lovely.", "The music was grating."),
    # Touch
    ("The pillow felt ordinary.", "The pillow felt soft.", "The pillow felt rough."),
    # Feeling
    ("She felt fine.", "She felt joyful.", "She felt miserable."),
    # Drink
    ("The wine was passable.", "The wine was exquisite.", "The wine was vile."),
    # Place
    ("The garden was usual.", "The garden was charming.", "The garden was dismal."),
]

AROUSAL_TRIPLES = [
    # Heart
    ("Her heart beat normally.", "Her heart raced.", "Her heart slowed."),
    # Nature
    ("The river flowed.", "The river raged.", "The river trickled."),
    # Fire
    ("The fire burned.", "The fire roared.", "The fire smoldered."),
    # Breath
    ("She breathed evenly.", "She breathed sharply.", "She breathed softly."),
    # Crowd
    ("The crowd murmured.", "The crowd erupted.", "The crowd hushed."),
    # Wind
    ("The wind blew.", "The wind howled.", "The wind whispered."),
    # Motion
    ("He walked.", "He bolted.", "He ambled."),
    # Animal
    ("The dog watched.", "The dog charged.", "The dog dozed."),
    # Drink
    ("She drank coffee.", "She gulped coffee.", "She sipped coffee."),
    # Sound
    ("Music played.", "Music blasted.", "Music murmured."),
    # Weather
    ("Rain fell.", "Rain poured.", "Rain drizzled."),
]

EXISTENCE_TRIPLES = [
    # Seed
    ("The seed lay there.", "The seed sprouted.", "The seed rotted."),
    # Candle
    ("The candle stood.", "The candle was lit.", "The candle was extinguished."),
    # Life
    ("A baby lay quietly.", "A baby was born.", "A grandparent died."),
    # Flower
    ("The flower was a bud.", "The flower bloomed.", "The flower wilted."),
    # Writing
    ("The page was blank.", "She wrote a poem.", "She tore up the poem."),
    # Cup
    ("The cup sat empty.", "She filled the cup.", "She emptied the cup."),
    # Guests
    ("The room was quiet.", "Guests arrived.", "Guests departed."),
    # Cooking
    ("The pot was cold.", "The water boiled.", "The water cooled."),
    # Friendship
    ("Two people stood apart.", "Two people met.", "Two people parted ways."),
    # Music
    ("The instrument was silent.", "Music began.", "Music ended."),
    # Building
    ("The lot was empty.", "A house went up.", "The house was torn down."),
    # Fabric
    ("Yarn lay in piles.", "She knitted a sweater.", "She unraveled the sweater."),
]

COHERENCE_TRIPLES = [
    # Action
    ("She showed up.", "She showed up on time.", "She showed up at strange hours."),
    # Cat
    ("The cat sat.", "The cat purred peacefully.", "The cat hissed wildly."),
    # Weather
    ("Rain fell.", "Rain fell steadily.", "Rain fell in odd bursts."),
    # Greeting
    ("He greeted me.", "He greeted me warmly.", "He greeted me strangely."),
    # River
    ("The river ran.", "The river flowed evenly.", "The river churned chaotically."),
    # Clock
    ("The clock ticked.", "The clock ticked steadily.", "The clock ticked unevenly."),
    # Birds
    ("Birds chirped.", "Birds chirped at dawn.", "Birds chirped at midnight."),
    # Dog
    ("The dog barked.", "The dog barked at strangers.", "The dog barked at nothing."),
    # Path
    ("The path wound.", "The path wound smoothly.", "The path wound bizarrely."),
    # Flame
    ("The flame burned.", "The flame burned steadily.", "The flame flickered erratically."),
    # Behavior
    ("She acted naturally.", "She acted as expected.", "She acted strangely."),
]

SUCCESS_FAILURE_TRIPLES = [
    # Practice
    ("She practiced.", "She nailed the routine.", "She fumbled the routine."),
    # Fishing
    ("He fished.", "He caught a big one.", "He came back empty-handed."),
    # Game
    ("The team played.", "The team won.", "The team lost."),
    # Cooking
    ("The chef cooked.", "The dish came out perfectly.", "The dish burned."),
    # Study
    ("She studied.", "She aced the test.", "She flunked the test."),
    # Aim
    ("He aimed.", "He hit the target.", "He missed wildly."),
    # Climb
    ("She climbed.", "She reached the summit.", "She fell off the cliff."),
    # Baking
    ("The cake baked.", "The cake rose beautifully.", "The cake collapsed."),
    # Race
    ("He ran the race.", "He crossed the finish line first.", "He stumbled at the start."),
    # Investment
    ("She invested.", "Her stocks grew.", "Her stocks crashed."),
    # Garden
    ("She gardened.", "Her vegetables flourished.", "Her vegetables withered."),
]

LOSS_TRIPLES = [
    # Home
    ("He had a roof.", "He bought a new house.", "He lost his home."),
    # Job
    ("She has a job.", "She got promoted.", "She got fired."),
    # Bird
    ("The bird nested.", "The bird thrived.", "The bird starved."),
    # Dog
    ("The dog had owners.", "The dog had a loving family.", "The dog was abandoned."),
    # Object
    ("He held the cup.", "He grasped the cup firmly.", "He dropped the cup."),
    # Town
    ("The town was quiet.", "The town was peaceful.", "The town was attacked."),
    # Neighborhood
    ("The neighborhood existed.", "The neighborhood prospered.", "The neighborhood declined."),
    # Key
    ("She had a key.", "She kept the key safe.", "She lost the key."),
    # Savings
    ("He had savings.", "He multiplied his savings.", "He gambled away his savings."),
    # Letter
    ("She wrote a letter.", "She kept the letter.", "She burned the letter."),
    # Treasure
    ("Coins lay there.", "He found treasure.", "He was robbed."),
]


# ===================== ENCODING =====================
def enc_via_sae(text, hook, sae):
    with torch.no_grad():
        return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)


def build_axis_from_triples(triples_list, hook, sae):
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


LAYER = 3
hook = f"blocks.{LAYER}.hook_resid_post"
print(f"Loading SAE for layer {LAYER}...")
sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res

axes = {}

# Build the 5 Lakoff schema axes (same as before)
print("\n=== Building schema axes ===")
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
    print(f"  built schema: {schema}")

# Build the 6 candidate axes from diverse triples
print("\n=== Building candidate axes from DIVERSE triples ===")
for name, triples_list in [
    ("VALENCE", VALENCE_TRIPLES),
    ("AROUSAL", AROUSAL_TRIPLES),
    ("EXISTENCE", EXISTENCE_TRIPLES),
    ("COHERENCE", COHERENCE_TRIPLES),
    ("SUCCESS", SUCCESS_FAILURE_TRIPLES),
    ("LOSS", LOSS_TRIPLES),
]:
    axes[name] = build_axis_from_triples(triples_list, hook, sae)
    print(f"  built anchor: {name} ({len(triples_list)} triples)")


# ===================== PCA =====================
axis_names = list(axes.keys())
M = np.stack([axes[n] for n in axis_names])

pca = PCA(n_components=min(len(axis_names), 10))
pca.fit(M)

print(f"\n=== Variance per PC ===")
print(", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca.explained_variance_ratio_)))

print(f"\n=== Top PC loadings ===")
for pc_idx in range(min(5, len(pca.components_))):
    pc = pca.components_[pc_idx]
    pc_unit = pc / np.linalg.norm(pc)
    loadings = [(n, float(axes[n] @ pc_unit)) for n in axis_names]
    loadings.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%): " +
          ", ".join(f"{n}{l:+.2f}" for n, l in loadings[:6]))


# ===================== NEURONPEDIA LOOKUP =====================
NEURONPEDIA_MODEL = "pythia-70m-deduped"
np_sae = f"{LAYER}-res-sm"


def fetch_feature(feat_idx):
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


def top_features(vec, top_n=10):
    abs_loads = np.abs(vec)
    sorted_idx = np.argsort(abs_loads)[::-1]
    positives, negatives = [], []
    for idx in sorted_idx:
        if vec[idx] > 0 and len(positives) < top_n:
            positives.append((int(idx), float(vec[idx])))
        elif vec[idx] < 0 and len(negatives) < top_n:
            negatives.append((int(idx), float(vec[idx])))
        if len(positives) >= top_n and len(negatives) >= top_n:
            break
    return positives, negatives


print(f"\n=== Neuronpedia lookup on PC1-3 ===")
for pc_idx in range(min(3, len(pca.components_))):
    pc = pca.components_[pc_idx]
    var = pca.explained_variance_ratio_[pc_idx] * 100
    pos, neg = top_features(pc, top_n=10)
    print(f"\n--- PC{pc_idx+1} ({var:.1f}% variance) ---")
    print(f"Positive pole:")
    for f_idx, ld in pos:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
        time.sleep(0.3)
    print(f"Negative pole:")
    for f_idx, ld in neg:
        d = fetch_feature(f_idx)
        print(f"  feat {f_idx:>5d} ({ld:+.3f}): {feat_desc(d)}")
        time.sleep(0.3)


torch.save({"axes": axes, "pca_components": pca.components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "axis_names": axis_names},
           "/Users/macn/Documents/embeddingexp/exp45_results.pt")
print("\nSaved: exp45_results.pt")
