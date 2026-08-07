"""
exp47: Measure what fraction of the SAE feature space each triple set
activates. Compares exp43 (formal triples) vs exp45 (diverse triples).

The question (Niamh): did the diverse triples remove a big chunk of the
model's representational space by avoiding formal-passive-legal-procedural
sentence structures, or did they activate similar but different features?

For each triple set:
  - encode all sentences (baseline + pole_a + pole_c) at L3
  - get active features per sentence (max-pooled over tokens, nonzero)
  - aggregate distinct features across the set
  - compare overlap / disjoint features
  - look up the features that fired in formal but NOT diverse (what we lost)
  - look up features that fired in diverse but NOT formal (what we gained)
"""
import json
import re
import time

import numpy as np
import requests
import torch
from sae_lens import SAE

import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from exp26_phase_a_measurement_gaps import model, device, collect


# ===================== TRIPLE DEFINITIONS =====================
# Formal triples (exp43)
FORMAL = {
    "VALENCE": [
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
    ],
    "AROUSAL": [
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
    ],
    "EXISTENCE": [
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
    ],
    "COHERENCE": [
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
    ],
    "SUCCESS": [
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
    ],
    "LOSS": [
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
    ],
}

# Diverse triples (exp45)
DIVERSE = {
    "VALENCE": [
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
    ],
    "AROUSAL": [
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
    ],
    "EXISTENCE": [
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
    ],
    "COHERENCE": [
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
    ],
    "SUCCESS": [
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
    ],
    "LOSS": [
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
    ],
}


# ===================== ENCODING =====================
LAYER = 3
hook = f"blocks.{LAYER}.hook_resid_post"
print(f"Loading SAE at layer {LAYER}...")
sae_res = SAE.from_pretrained(release="pythia-70m-deduped-res-sm", sae_id=hook, device="cpu")
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res


def enc(text):
    with torch.no_grad():
        return sae.encode(collect(text, hook)).max(0).values.numpy().astype(np.float64)


def features_above_threshold(activation_vec, thresh=0.01):
    """Return set of feature indices that activated above threshold."""
    return set(np.where(activation_vec > thresh)[0].tolist())


def all_features_from_triples(triple_dict):
    """For each axis, collect the union of all features that fire across the
    triples (baseline + pole_a + pole_c). Returns dict of {axis: set of features}.
    Also tracks total activation magnitude per feature."""
    per_axis = {}
    activation_sums = np.zeros(32768)
    for axis, triples in triple_dict.items():
        feats = set()
        for (b, a, c) in triples:
            for sent in [b, a, c]:
                v = enc(sent)
                feats |= features_above_threshold(v, thresh=0.01)
                activation_sums += v
        per_axis[axis] = feats
    all_active = set().union(*per_axis.values())
    return per_axis, all_active, activation_sums


print("\n=== Encoding FORMAL triples ===")
formal_per_axis, formal_all, formal_act_sums = all_features_from_triples(FORMAL)

print("\n=== Encoding DIVERSE triples ===")
diverse_per_axis, diverse_all, diverse_act_sums = all_features_from_triples(DIVERSE)


# ===================== COMPARISON =====================
print(f"\n\n{'='*72}")
print("SUBSTRATE COVERAGE COMPARISON")
print(f"{'='*72}")

print(f"\nTotal SAE features: 32768")
print(f"\nDistinct features activated (threshold > 0.01):")
print(f"  FORMAL triples:  {len(formal_all):>6d}  ({100*len(formal_all)/32768:.2f}% of SAE features)")
print(f"  DIVERSE triples: {len(diverse_all):>6d}  ({100*len(diverse_all)/32768:.2f}% of SAE features)")

overlap = formal_all & diverse_all
formal_only = formal_all - diverse_all
diverse_only = diverse_all - formal_all

print(f"\nSet relationships:")
print(f"  Overlap (both):     {len(overlap):>6d}  ({100*len(overlap)/len(formal_all|diverse_all):.1f}% of union)")
print(f"  FORMAL only:        {len(formal_only):>6d}  ({100*len(formal_only)/len(formal_all):.1f}% of FORMAL set)")
print(f"  DIVERSE only:       {len(diverse_only):>6d}  ({100*len(diverse_only)/len(diverse_all):.1f}% of DIVERSE set)")

print(f"\nPer-axis feature counts:")
print(f"  {'axis':>12}  {'FORMAL':>8}  {'DIVERSE':>8}  {'overlap':>8}  {'F-only':>7}  {'D-only':>7}")
for axis in FORMAL:
    f_set = formal_per_axis[axis]
    d_set = diverse_per_axis[axis]
    o = f_set & d_set
    print(f"  {axis:>12}  {len(f_set):>8}  {len(d_set):>8}  {len(o):>8}  {len(f_set-d_set):>7}  {len(d_set-f_set):>7}")


# ===================== TOP DIFFERENTIAL FEATURES =====================
# Features that fired in FORMAL but NOT DIVERSE (with highest activation magnitude in formal)
print(f"\n\n{'='*72}")
print("TOP FEATURES ONLY IN FORMAL (what we 'lost' by switching to diverse triples)")
print(f"{'='*72}")

formal_only_sorted = sorted(formal_only, key=lambda i: formal_act_sums[i], reverse=True)[:20]


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


for f_idx in formal_only_sorted:
    act = formal_act_sums[f_idx]
    d = fetch_feature(int(f_idx))
    print(f"  feat {int(f_idx):>5d} (total act={act:.2f}): {feat_desc(d)}")
    time.sleep(0.3)


print(f"\n\n{'='*72}")
print("TOP FEATURES ONLY IN DIVERSE (what we 'gained' by switching)")
print(f"{'='*72}")

diverse_only_sorted = sorted(diverse_only, key=lambda i: diverse_act_sums[i], reverse=True)[:20]

for f_idx in diverse_only_sorted:
    act = diverse_act_sums[f_idx]
    d = fetch_feature(int(f_idx))
    print(f"  feat {int(f_idx):>5d} (total act={act:.2f}): {feat_desc(d)}")
    time.sleep(0.3)


# ===================== CHECK SPECIFIC LEGAL-CLUSTER FEATURES =====================
print(f"\n\n{'='*72}")
print("DID THE LEGAL-CLUSTER FEATURES FIRE IN DIVERSE TRIPLES?")
print(f"{'='*72}")

LEGAL_CLUSTER = {
    21809: "expressions of gratitude and friendship",
    5355:  "legal terminology related to liability and warranties",
    1812:  "complex legal terms and organizational identifiers",
    22622: "instances of line breaks or empty sections in the text",
    3637:  "key terms related to legal proceedings and opinions",
}

print(f"\n  {'feat':>6}  {'desc':>50}  {'in FORMAL?':>10}  {'in DIVERSE?':>11}  {'F act':>8}  {'D act':>8}")
for f_idx, desc in LEGAL_CLUSTER.items():
    in_f = f_idx in formal_all
    in_d = f_idx in diverse_all
    f_act = formal_act_sums[f_idx]
    d_act = diverse_act_sums[f_idx]
    print(f"  {f_idx:>6}  {desc[:50]:>50}  {str(in_f):>10}  {str(in_d):>11}  {f_act:>8.2f}  {d_act:>8.2f}")

print("\nDone.")
