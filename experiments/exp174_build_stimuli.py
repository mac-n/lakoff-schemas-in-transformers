"""
exp174_build_stimuli.py — build and verify the frozen stimulus file for
exp174 Part B (the happy / incoherent dissociation). See PREREG_exp174.md.

No model weights are loaded: tokenizer only.

Writes exp174_stimuli.json (sentences + frozen shuffle permutations) and
prints its sha256, which goes into the prereg before the model loads.

Checks, all hard failures:
  - every sentence is lowercase, letters and spaces only
  - 12..16 words; lengths within a matched triple differ by at most 1
  - no word from the UP-DOWN anchor list (LAKOFF_SCHEMAS_MML), the literal
    UP/DOWN list of exp141, or a supplementary height / posture list,
    including simple inflections
  - tokenising word-by-word with a leading space reproduces the tokenisation
    of the whole sentence with a leading space (so shuffling word groups
    keeps exactly the same tokens)
  - no shuffle permutation is the identity
"""

import hashlib
import json
import random
import re
import sys

sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

OUT = "/Users/macn/Documents/embeddingexp/exp174_stimuli.json"
N_SHUFFLES = 5
MIN_WORDS, MAX_WORDS = 12, 16

# exp141's literal spatial lists, copied verbatim
LAKOFF_UP = ["up", "rise", "rose", "rising", "ascend", "raise", "climb", "lift",
             "above", "over", "top", "high", "higher", "upward"]
LAKOFF_DOWN = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
               "below", "under", "bottom", "low", "lower", "downward"]

# Supplementary height / posture / vertical-motion vocabulary
SUPPLEMENT = """tall short sky skies mountain hill tower ceiling floor ground
peak summit valley upstairs downstairs height elevated stairs ladder jump
stand stood upright sat sit lay lie climbed lifted raised dropped sank sunk
fallen risen deep beneath underneath overhead aloft descended ascended
sunrise sunset bounce bounded skip skipped hang hung stack stacked bury
buried dive dived plunge plunged soar tumble tumbled collapse collapsed
uphill downhill highest lowest happiness sadness worsened""".split()

# (happy, sad, neutral) matched triples
TRIPLES = [
 ("the children laughed all afternoon while their grandmother baked warm bread in the kitchen",
  "the children cried all afternoon while their grandmother packed her things in the kitchen",
  "the children read all afternoon while their grandmother sorted old papers in the kitchen"),
 ("she opened the letter and smiled because her sister was finally coming home for the summer",
  "she opened the letter and wept because her sister was never coming home from the war",
  "she opened the letter and noted that her sister was changing her address for the summer"),
 ("after years of practice he won the prize and his whole family cheered with delight",
  "after years of practice he lost the prize and his whole family left in silence",
  "after years of practice he entered the contest and his whole family watched from home"),
 ("the puppy raced across the garden and licked the face of the delighted little boy",
  "the puppy stopped breathing in the garden and the little boy could not stop sobbing",
  "the puppy walked across the garden and sniffed the shoes of the little boy there"),
 ("we danced at the wedding until midnight and everyone said it was a wonderful celebration",
  "we waited at the funeral until midnight and everyone said it was a terrible loss",
  "we stayed at the meeting until midnight and everyone said it was a long discussion"),
 ("the doctor told them the treatment had worked and their daughter was completely cured",
  "the doctor told them the treatment had failed and their daughter would not recover",
  "the doctor told them the treatment had started and their daughter was now resting"),
 ("my friends surprised me with a cake and we sang together late into the night",
  "my friends forgot my birthday entirely and i stayed home alone late into the night",
  "my friends brought me a small package and we talked together late into the night"),
 ("the old man smiled warmly as his grandchildren ran to greet him at the gate",
  "the old man wept quietly as his grandchildren walked away from him at the gate",
  "the old man nodded briefly as his grandchildren walked slowly past him at the gate"),
 ("they finally bought their first house and celebrated with champagne on the sunny porch",
  "they finally lost their first house and packed their boxes on the empty porch",
  "they finally painted their first house and moved their boxes on the front porch"),
 ("the teacher praised her essay and the whole class applauded her wonderful achievement",
  "the teacher mocked her essay and the whole class laughed at her mistake",
  "the teacher collected her essay and the whole class opened the next chapter"),
 ("he hugged his mother at the airport after ten long years apart and laughed with joy",
  "he lost his mother at the hospital after ten years of pain and wept with grief",
  "he met his mother at the airport after ten long years abroad and talked about work"),
 ("the garden was full of flowers and the whole family enjoyed a lovely picnic there",
  "the garden was full of weeds and the whole family mourned their ruined harvest there",
  "the garden was full of tools and the whole family sorted the wooden crates there"),
 ("her new job was a dream and she loved every single day at the office",
  "her new job was a nightmare and she dreaded every single day at the office",
  "her new job was in accounting and she worked every single day at the office"),
 ("the team won the final match and the entire town celebrated in the streets all night",
  "the team lost the final match and the entire town grieved in the streets all night",
  "the team played the final match and the entire town watched on television all night"),
 ("he received a generous gift from his uncle and thanked him with a grateful smile",
  "he received a cruel insult from his uncle and answered him with a bitter scowl",
  "he received a brief message from his uncle and answered him with a typed note"),
 ("the baby giggled when her father tickled her toes and kissed her tiny hands",
  "the baby screamed when her father shouted at her and slammed the heavy door",
  "the baby blinked when her father changed her socks and closed the bedroom door"),
 ("spring arrived at last and the birds sang sweetly in the bright morning sunshine",
  "winter arrived at last and the birds froze silently in the bitter morning frost",
  "autumn arrived at last and the birds gathered quietly in the grey morning light"),
 ("the soldiers returned safely to their families and the village rejoiced for three whole days",
  "the soldiers never returned to their families and the village mourned for three whole days",
  "the soldiers marched slowly past their barracks and the village watched for three whole days"),
 ("she passed every exam with honours and her proud parents threw a wonderful party",
  "she failed every exam that term and her furious parents refused to speak to her",
  "she finished every exam that term and her parents drove to the station for her"),
 ("the kitten purred in my lap while the fire crackled and we felt completely content",
  "the kitten died in my lap while the fire faded and we felt completely heartbroken",
  "the kitten slept in my lap while the fire burned and we watched the evening news"),
 ("our neighbours welcomed us with fresh pie and friendly smiles on our very first evening",
  "our neighbours greeted us with angry shouts and hostile stares on our very first evening",
  "our neighbours passed us with shopping bags and car keys on our very first evening"),
 ("the artist finished her painting and the gallery crowd gasped with admiration and delight",
  "the artist destroyed her painting and the gallery crowd watched with horror and dismay",
  "the artist framed her painting and the gallery crowd moved along to the next room"),
 ("they found the missing child safe and unharmed and the mother wept with relief and joy",
  "they found the missing child cold and lifeless and the mother screamed with grief and despair",
  "they found the missing child at the library and the mother walked there to collect him"),
 ("the scientist discovered a cure and thousands of grateful patients sent her letters of thanks",
  "the scientist abandoned the cure and thousands of desperate patients sent her letters of rage",
  "the scientist described the method and thousands of interested students sent her letters with questions"),
 ("we watched the dawn from the beach and felt peaceful and glad to be alive",
  "we watched the wreckage from the beach and felt hopeless and afraid for the survivors",
  "we watched the ferry from the beach and counted the cars waiting at the terminal"),
 ("his brother forgave him at last and the two of them embraced and laughed together",
  "his brother betrayed him at last and the two of them never spoke again afterwards",
  "his brother phoned him at last and the two of them discussed the schedule together"),
 ("the festival filled the square with music and laughter and everybody danced until dawn",
  "the riot filled the square with smoke and screaming and everybody fled before dawn",
  "the market filled the square with stalls and customers and everybody traded until noon"),
 ("she adored her gentle grandfather who told wonderful stories and always made her laugh",
  "she feared her violent grandfather who told horrible stories and always made her cry",
  "she visited her retired grandfather who kept detailed records and always wore a hat"),
 ("the rescue dog found a loving home and wagged his tail all the way there",
  "the rescue dog lost his only home and whimpered softly all the way there",
  "the rescue dog entered a different kennel and looked around all the way there"),
 ("after the long drought the rain came and the farmers cheered in their green fields",
  "after the long drought the fire came and the farmers wept in their burned fields",
  "after the long drought the survey came and the farmers answered in their own words"),
 ("he proposed by the lake and she said yes with tears of pure joy",
  "he left her by the lake and she said nothing through tears of pure misery",
  "he parked by the lake and she said they should check the map again"),
 ("the choir sang beautifully and the audience left the hall smiling and humming the tunes",
  "the choir sang dreadfully and the audience left the hall frowning and cursing the tickets",
  "the choir sang twice and the audience left the hall carrying their coats and programmes"),
 ("my grandmother recovered fully and now she laughs and gardens every morning with her friends",
  "my grandmother deteriorated quickly and now she suffers and struggles every morning with her pain",
  "my grandmother moved recently and now she shops and cooks every morning with her neighbour"),
 ("the students cheered when the principal announced a holiday and free ice cream for everyone",
  "the students groaned when the principal announced a punishment and extra homework for everyone",
  "the students listened when the principal announced a timetable and new lockers for everyone"),
 ("it was the loveliest evening of the year and we felt lucky and truly grateful",
  "it was the cruellest evening of the year and we felt lonely and truly ashamed",
  "it was the final evening of the year and we felt ready and reasonably prepared"),
 ("the orphan was adopted by a kind couple who cherished and protected him always",
  "the orphan was abandoned by a cruel couple who neglected and frightened him always",
  "the orphan was registered by a local couple who recorded and reported his details"),
 ("her painting won first place and she beamed with pride as the crowd applauded",
  "her painting was torn apart and she trembled with shame as the crowd jeered",
  "her painting was shown third and she waited with others as the crowd entered"),
 ("the fishermen came home with a wonderful catch and the harbour rang with merry songs",
  "the fishermen came home with empty nets and the harbour echoed with bitter complaints",
  "the fishermen came home with their usual catch and the harbour filled with delivery vans"),
 ("we laughed so hard at his jokes that our sides ached and tears streamed freely",
  "we cried so hard at his funeral that our throats ached and tears streamed freely",
  "we talked so long at his office that our coffee cooled and the clock struck four"),
 ("the little girl hugged her new teddy bear and danced merrily along the sunny lane",
  "the little girl clutched her broken teddy bear and limped miserably along the muddy lane",
  "the little girl carried her brown teddy bear and walked steadily along the narrow lane"),
]
VALENCES = ["happy", "sad", "neutral"]


def inflections(w):
    forms = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er",
             w + "est", w + "ly", w + "ness"}
    if w.endswith("e"):
        forms |= {w[:-1] + "ing", w[:-1] + "ed", w + "r", w + "st"}
    if w.endswith("y"):
        forms |= {w[:-1] + "ier", w[:-1] + "iest", w[:-1] + "ily", w[:-1] + "iness"}
    return forms


def banned_set():
    base = set(LAKOFF_UP + LAKOFF_DOWN + SUPPLEMENT)
    for p, n in LAKOFF_SCHEMAS_MML["UP-DOWN"]:
        base.add(p); base.add(n)
    out = set()
    for w in base:
        out |= inflections(w)
    return out


def main():
    banned = banned_set()
    problems = []
    assert len(TRIPLES) == 40, len(TRIPLES)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

    items = []
    for ti, triple in enumerate(TRIPLES):
        lens = []
        for val, sent in zip(VALENCES, triple):
            if not re.fullmatch(r"[a-z ]+", sent) or "  " in sent or sent != sent.strip():
                problems.append(f"triple {ti} {val}: bad characters: {sent!r}")
            words = sent.split(" ")
            lens.append(len(words))
            if not (MIN_WORDS <= len(words) <= MAX_WORDS):
                problems.append(f"triple {ti} {val}: {len(words)} words")
            hits = [w for w in words if w in banned]
            if hits:
                problems.append(f"triple {ti} {val}: banned {hits}")
            groups = [tok.encode(" " + w, add_special_tokens=False) for w in words]
            flat = [t for g in groups for t in g]
            whole = tok.encode(" " + sent, add_special_tokens=False)
            if flat != whole:
                problems.append(f"triple {ti} {val}: word-wise tokenisation differs from whole")
            perms = []
            rng = random.Random(174000 + ti * 10 + VALENCES.index(val))
            while len(perms) < N_SHUFFLES:
                p = list(range(len(words)))
                rng.shuffle(p)
                # reject identity and anything keeping more than a third in place
                if sum(1 for i, j in enumerate(p) if i == j) > len(words) // 3:
                    continue
                if p in perms:
                    continue
                perms.append(p)
            items.append({"triple": ti, "valence": val, "sentence": sent,
                          "n_words": len(words), "n_tokens": len(flat),
                          "shuffles": perms})
        if max(lens) - min(lens) > 1:
            problems.append(f"triple {ti}: word counts {lens} differ by more than 1")

    if problems:
        print("STIMULI INVALID:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    payload = {"experiment": "exp174", "n_shuffles": N_SHUFFLES,
               "tokenizer": "EleutherAI/pythia-410m",
               "note": "each word tokenised with a leading space; shuffles are "
                       "permutations of word indices",
               "items": items}
    blob = json.dumps(payload, indent=1, sort_keys=True).encode()
    with open(OUT, "wb") as f:
        f.write(blob)
    ntok = [it["n_tokens"] for it in items]
    nw = [it["n_words"] for it in items]
    for v in VALENCES:
        sub = [it for it in items if it["valence"] == v]
        print(f"{v:8s} n={len(sub)}  mean words {sum(i['n_words'] for i in sub)/len(sub):.2f}"
              f"  mean tokens {sum(i['n_tokens'] for i in sub)/len(sub):.2f}")
    print(f"words {min(nw)}..{max(nw)}  tokens {min(ntok)}..{max(ntok)}")
    print("STIMULI VALID:", len(items), "sentences")
    print("sha256:", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
