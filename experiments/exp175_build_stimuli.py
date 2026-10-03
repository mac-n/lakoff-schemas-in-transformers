"""
exp175_build_stimuli.py — build and verify the frozen stimulus file for
exp175 (is UP grounded in "holding against gravity / effecting"?).
See PREREG_exp175.md. Tokenizer only; no model weights are loaded.

Produces exp175_stimuli.json and prints its sha256.

Contents:
  - a larger purely spatial UP / DOWN word list (candidates below; a word is
    kept only if it is ONE token when written with a leading space)
  - valence word lists (same filter)
  - the pole words of the lab's other seven schemas (same filter; words that
    are also on a spatial list are dropped from the schema lists)
  - three neutral "mention" frames that hold one word each
  - 40 NEW matched happy / sad / neutral sentence triples (none reused from
    exp174) with 3 frozen scramble orders each
"""

import hashlib
import json
import random
import re
import sys

sys.path.insert(0, "/Users/macn/Documents/embeddingexp")
from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

OUT = "/Users/macn/Documents/embeddingexp/exp175_stimuli.json"
N_SHUFFLES = 3
MIN_WORDS, MAX_WORDS = 12, 16
MIN_POLE = 25

UP_CAND = """up above over top high higher highest upward rise rising rose raise
raised lift lifted climb climbing climbed ascend peak summit upper tall taller
sky ceiling roof overhead elevated upstairs atop soar aloft uphill height
heights tower mount crest skyward""".split()
DOWN_CAND = """down below under bottom low lower lowest downward fall falling fell
fallen drop dropped sink sinking sank sunk descend beneath underneath floor
ground base basement downstairs depth depths deep deeper pit valley plunge dive
downhill collapse slump cellar underground buried""".split()
VALPOS_CAND = """happy joy glad cheerful delighted pleased smile laugh love
wonderful lovely merry bliss proud hope kind gentle calm lucky grateful
pleasant beautiful""".split()
VALNEG_CAND = """sad grief sorrow miserable gloomy upset cry weep hate terrible
awful bitter despair ashamed fear cruel harsh lonely unlucky angry unpleasant
ugly""".split()
SCHEMA_NAMES = ["IN-OUT_CLEAN", "FORWARD-BACK", "PATH-MOTION", "LIGHT-DARK",
                "FORCE", "BALANCE", "DIFFICULTY-BURDEN"]
COMMON = ["the", "of", "and", "to", "in", "is", "it", "you", "that", "he",
          "was", "for", "on", "are", "with", "as", "his", "they", "at", "be"]
RARE = ["serendipity", "ostracize", "perspicacity", "obfuscate", "sycophant"]

FRAMES = [
    "the next word on the list is {w} and then we move on to the rest of the page",
    "she wrote the word {w} on the board before the lesson began that morning",
    "in the first column he typed {w} and saved the file to the shared folder",
]

# literal lists of exp141 + supplementary height / posture vocabulary (as exp174)
LAKOFF_UP = ["up", "rise", "rose", "rising", "ascend", "raise", "climb", "lift",
             "above", "over", "top", "high", "higher", "upward"]
LAKOFF_DOWN = ["down", "fall", "fell", "falling", "descend", "drop", "sink",
               "below", "under", "bottom", "low", "lower", "downward"]
SUPPLEMENT = """tall short sky skies mountain hill tower ceiling floor ground
peak summit valley upstairs downstairs height elevated stairs ladder jump
stand stood upright sat sit lay lie climbed lifted raised dropped sank sunk
fallen risen deep beneath underneath overhead aloft descended ascended
sunrise sunset bounce bounded skip skipped hang hung stack stacked bury
buried dive dived plunge plunged soar tumble tumbled collapse collapsed
uphill downhill highest lowest happiness sadness worsened roof atop crest
mount cellar basement pit depth depths underground slump""".split()

TRIPLES = [
 ("the bakery gave away free cakes and the whole street came out smiling that morning",
  "the bakery burned to ashes and the whole street came out weeping that morning",
  "the bakery changed its hours and the whole street came out earlier that morning"),
 ("my sister had twins last night and the nurses said both babies are doing wonderfully",
  "my sister lost her twins last night and the nurses said nothing could be done",
  "my sister worked a shift last night and the nurses said the schedule would change"),
 ("the refugees were welcomed with blankets and soup and the children finally slept without fear",
  "the refugees were turned away without blankets or soup and the children shivered with fear",
  "the refugees were counted and given numbered cards and the children waited beside the fence"),
 ("he opened the envelope and shouted with joy because the university had accepted him",
  "he opened the envelope and groaned with despair because the university had rejected him",
  "he opened the envelope and read the form because the university had requested it"),
 ("the old friends met again after decades and talked and laughed until the cafe closed",
  "the old friends quarrelled again after decades and parted in anger before the cafe closed",
  "the old friends met again on tuesday and compared their notes until the cafe closed"),
 ("her garden won the village prize and the neighbours brought flowers to congratulate her",
  "her garden was wrecked by vandals and the neighbours brought nothing to comfort her",
  "her garden was measured by surveyors and the neighbours brought papers to show her"),
 ("the surgeon smiled and said the operation was a complete success and he could go home",
  "the surgeon sighed and said the operation was a complete failure and he would not wake",
  "the surgeon nodded and said the operation was scheduled for friday and he could wait"),
 ("we adopted a gentle old dog and now the house is full of warmth and laughter",
  "we lost our gentle old dog and now the house is full of silence and sorrow",
  "we borrowed a quiet old dog and now the house is full of leads and bowls"),
 ("the rain stopped before the parade and the crowd cheered as the band began",
  "the storm struck before the parade and the crowd screamed as the stage gave way",
  "the trucks arrived before the parade and the crowd waited as the band prepared"),
 ("grandfather told his funniest story and all the cousins laughed until their faces hurt",
  "grandfather told us he was dying and all the cousins cried until their faces hurt",
  "grandfather told his usual story and all the cousins listened until their tea cooled"),
 ("the lost cat came home after a month and the little boy hugged her and laughed",
  "the lost cat was found dead that month and the little boy held her and sobbed",
  "the lost cat was seen nearby that month and the little boy noted it and waited"),
 ("she got the job she always wanted and celebrated with her friends by the river",
  "she lost the job she always wanted and wandered alone for hours by the river",
  "she described the job she had applied for and walked with her friends by the river"),
 ("the village school reopened with new books and the teachers greeted every child with delight",
  "the village school closed forever that week and the teachers left every classroom in tears",
  "the village school reopened on monday and the teachers handed every child a timetable"),
 ("his first novel was published at last and readers wrote to say they loved it",
  "his first novel was rejected again and editors wrote to say they hated it",
  "his first novel was catalogued at last and librarians wrote to say they shelved it"),
 ("the harvest was the richest in years and the farmers feasted and sang all evening",
  "the harvest was the poorest in years and the farmers starved and despaired all winter",
  "the harvest was the earliest in years and the farmers sorted and weighed all evening"),
 ("they repaired the old bridge together and the two villages celebrated with a shared feast",
  "they destroyed the old bridge in anger and the two villages never spoke again",
  "they inspected the old bridge together and the two villages received a written report"),
 ("the nurse brought wonderful news and my mother laughed for the first time in weeks",
  "the nurse brought dreadful news and my mother wept for the first time in years",
  "the nurse brought the forms and my mother signed for the first time that week"),
 ("a stranger returned my wallet with every coin and i thanked her with a grateful hug",
  "a stranger stole my wallet with every coin and i cursed him with a helpless rage",
  "a stranger examined my wallet with every coin and i answered him with a brief reply"),
 ("the youngest daughter sang at the concert and her parents glowed with pride and affection",
  "the youngest daughter fainted at the concert and her parents trembled with dread and panic",
  "the youngest daughter arrived at the concert and her parents waited with coats and tickets"),
 ("after the long war the brothers embraced and promised never to be parted again",
  "after the long war the brothers were dead and their mother never smiled again",
  "after the long war the brothers were registered and their mother signed the forms"),
 ("the bride laughed as the guests threw petals and the musicians played her favourite song",
  "the bride sobbed as the guests whispered cruelly and the musicians packed their instruments early",
  "the bride waited as the guests found seats and the musicians tuned their instruments again"),
 ("our team finished the project early and the manager rewarded everyone with a generous bonus",
  "our team ruined the project completely and the manager punished everyone with a brutal lecture",
  "our team discussed the project briefly and the manager emailed everyone with a revised agenda"),
 ("the feverish child woke smiling and asked for breakfast and the whole ward rejoiced",
  "the feverish child stopped breathing before breakfast and the whole ward grieved in silence",
  "the feverish child woke quietly and asked for water and the whole ward continued working"),
 ("he planted an orchard for his grandchildren and lived to taste the sweetest apples",
  "he planted an orchard for his grandchildren and died before a single tree bloomed",
  "he planted an orchard for his employer and recorded the date of every tree"),
 ("the sailors spotted land at dawn and wept with relief after months of thirst",
  "the sailors spotted rocks at dawn and screamed in terror as the hull split apart",
  "the sailors spotted gulls at dawn and wrote the time in the logbook as usual"),
 ("she forgave her father at his bedside and they held hands and smiled at each other",
  "she cursed her father at his bedside and they glared and said nothing to each other",
  "she visited her father at his bedside and they watched television and spoke about traffic"),
 ("the puppies played in the hay and the farm children squealed with delight all afternoon",
  "the puppies starved in the hay and the farm children wept with guilt all afternoon",
  "the puppies slept in the hay and the farm children swept the yard all afternoon"),
 ("her letter arrived with wonderful news and i read it aloud to everyone at dinner",
  "her letter arrived with terrible news and i read it alone in the dark kitchen",
  "her letter arrived with the invoices and i read it later at the kitchen table"),
 ("the choir of children sang for the patients and every face in the ward brightened",
  "the news of the accident reached the patients and every face in the ward darkened",
  "the cart of clean linen reached the patients and every bed in the ward was changed"),
 ("we reached the cottage at dusk and found a warm fire and a generous supper waiting",
  "we reached the cottage at dusk and found broken windows and all our belongings stolen",
  "we reached the cottage at dusk and found the key and the instructions as described"),
 ("the mayor thanked the volunteers warmly and the whole hall applauded their kindness and courage",
  "the mayor blamed the volunteers bitterly and the whole hall jeered at their clumsy mistakes",
  "the mayor listed the volunteers alphabetically and the whole hall checked their names and numbers"),
 ("my brother beat the disease at last and danced at his own birthday party",
  "my brother lost to the disease at last and never saw his own birthday party",
  "my brother described the disease at length and then planned his own birthday party"),
 ("the fox cubs played safely by the stream while their mother watched with calm contentment",
  "the fox cubs died slowly by the stream while their mother whimpered with helpless misery",
  "the fox cubs moved slowly by the stream while their mother watched from the far bank"),
 ("she tasted the soup her grandmother made and smiled at the lovely familiar flavour",
  "she tasted the soup her grandmother made and wept because the old woman was gone",
  "she tasted the soup her grandmother made and added salt before serving the guests"),
 ("the fireworks lit the harbour and the families cheered and hugged on the crowded pier",
  "the explosion wrecked the harbour and the families screamed and fled from the burning pier",
  "the lanterns marked the harbour and the families queued and boarded from the wooden pier"),
 ("he finally paid his debts and slept peacefully for the first time in years",
  "he finally lost his savings and slept poorly for the rest of his years",
  "he finally filed his receipts and slept as usual for the rest of the week"),
 ("the students surprised their teacher with a gift and she cried from pure gratitude",
  "the students tormented their teacher with cruel jokes and she cried from pure humiliation",
  "the students presented their teacher with the forms and she signed from pure habit"),
 ("our village won the football cup and the bells rang while everyone danced in the square",
  "our village lost its only doctor and the bells tolled while everyone mourned in the square",
  "our village replaced its only bus and the bells rang while everyone shopped in the square"),
 ("the kind stranger paid for our meal and wished us a wonderful journey with a smile",
  "the cruel stranger spat at our meal and wished us a miserable journey with a sneer",
  "the other stranger asked about our meal and wished us a quick journey with a nod"),
 ("the twins blew out the candles together and the whole room burst into delighted applause",
  "the twins fought about the candles bitterly and the whole room filled with awkward silence",
  "the twins counted out the candles together and the whole room waited for the photograph"),
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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    problems = []

    def single(w):
        return len(tok.encode(" " + w, add_special_tokens=False)) == 1

    def keep(cands):
        seen, out = set(), []
        for w in cands:
            if w not in seen and single(w):
                out.append(w)
            seen.add(w)
        return out

    up, down = keep(UP_CAND), keep(DOWN_CAND)
    if set(up) & set(down):
        problems.append(f"spatial overlap {set(up) & set(down)}")
    if len(up) < MIN_POLE or len(down) < MIN_POLE:
        problems.append(f"spatial list too small: {len(up)} up, {len(down)} down")
    spatial = set(up) | set(down)
    valpos, valneg = keep(VALPOS_CAND), keep(VALNEG_CAND)
    valpos = [w for w in valpos if w not in spatial]
    valneg = [w for w in valneg if w not in spatial]
    schemas = {}
    for name in SCHEMA_NAMES:
        pairs = LAKOFF_SCHEMAS_MML[name]
        pos = keep(sorted(set(p[0] for p in pairs)))
        neg = keep(sorted(set(p[1] for p in pairs)))
        pos = [w for w in pos if w not in spatial and re.fullmatch(r"[a-z]+", w)]
        neg = [w for w in neg if w not in spatial and re.fullmatch(r"[a-z]+", w)]
        both = set(pos) & set(neg)
        pos = [w for w in pos if w not in both]; neg = [w for w in neg if w not in both]
        schemas[name] = {"pos": pos, "neg": neg}

    # frames: one slot, no spatial / banned word, lowercase
    banned_base = set(LAKOFF_UP + LAKOFF_DOWN + SUPPLEMENT) | spatial
    for p, n in LAKOFF_SCHEMAS_MML["UP-DOWN"]:
        banned_base.add(p); banned_base.add(n)
    banned = set()
    for w in banned_base:
        banned |= inflections(w)
    for fr in FRAMES:
        words = fr.replace("{w}", "SLOT").split(" ")
        if words.count("SLOT") != 1:
            problems.append(f"frame slot count: {fr}")
        hits = [w for w in words if w in banned]
        if hits:
            problems.append(f"frame has banned {hits}: {fr}")
        if len(words) - 1 - words.index("SLOT") < 8:
            problems.append(f"frame has fewer than 8 words after the slot: {fr}")

    old = json.load(open("/Users/macn/Documents/embeddingexp/exp174_stimuli.json"))
    old_sents = {it["sentence"] for it in old["items"]}
    items = []
    assert len(TRIPLES) == 40, len(TRIPLES)
    for ti, triple in enumerate(TRIPLES):
        lens = []
        for val, sent in zip(VALENCES, triple):
            if sent in old_sents:
                problems.append(f"triple {ti} {val}: reused from exp174")
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
            if flat != tok.encode(" " + sent, add_special_tokens=False):
                problems.append(f"triple {ti} {val}: word-wise tokenisation differs from whole")
            perms = []
            rng = random.Random(175000 + ti * 10 + VALENCES.index(val))
            while len(perms) < N_SHUFFLES:
                p = list(range(len(words)))
                rng.shuffle(p)
                if sum(1 for i, j in enumerate(p) if i == j) > len(words) // 3:
                    continue
                if p in perms:
                    continue
                perms.append(p)
            items.append({"triple": ti, "valence": val, "sentence": sent,
                          "n_words": len(words), "n_tokens": len(flat), "shuffles": perms})
        if max(lens) - min(lens) > 1:
            problems.append(f"triple {ti}: word counts {lens} differ by more than 1")

    if problems:
        print("STIMULI INVALID:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    payload = {"experiment": "exp175", "tokenizer": "EleutherAI/pythia-410m",
               "n_shuffles": N_SHUFFLES, "frames": FRAMES,
               "words": {"UP": up, "DOWN": down, "VALPOS": valpos, "VALNEG": valneg,
                         "schemas": schemas, "COMMON": COMMON, "RARE": RARE},
               "items": items}
    blob = json.dumps(payload, indent=1, sort_keys=True).encode()
    open(OUT, "wb").write(blob)
    print(f"spatial UP {len(up)}: {' '.join(up)}")
    print(f"spatial DOWN {len(down)}: {' '.join(down)}")
    print(f"dropped (not one token): UP {[w for w in UP_CAND if w not in up]}  DOWN {[w for w in DOWN_CAND if w not in down]}")
    print(f"valence + {len(valpos)}  - {len(valneg)}")
    for name in SCHEMA_NAMES:
        print(f"  {name}: {len(schemas[name]['pos'])} / {len(schemas[name]['neg'])}")
    for v in VALENCES:
        sub = [it for it in items if it["valence"] == v]
        print(f"{v:8s} n={len(sub)}  mean words {sum(i['n_words'] for i in sub)/len(sub):.2f}"
              f"  mean tokens {sum(i['n_tokens'] for i in sub)/len(sub):.2f}")
    print("STIMULI VALID:", len(items), "sentences")
    print("sha256:", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
