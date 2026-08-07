"""
exp10_corpus_design.py - build vocabulary lists for the probe corpus.

Schemas: UP/DOWN, IN/OUT (CONTAINER), and BEVERAGE (sham control).
Children per schema chosen for multi-child decorrelation testing.

For each (schema, child) cell, this script:
  1. Starts from a small seed vocabulary curated by hand
  2. Uses WordNet to expand each seed into its synonym set
  3. Flags overlapping vocabulary across children within the same schema
     (overlap = decorrelation failure → swap or drop)
  4. Outputs structured vocabulary JSON for sentence construction

This is iterative. First pass: see what comes out, prune overlap by hand,
re-run. Goal: each cell ends with ~15-25 words whose vocabulary doesn't
appear in any other cell of the same schema.

Output: corpus_vocabulary.json + a readable .md report.
"""

import json
from collections import defaultdict

from nltk.corpus import wordnet as wn

# ---- Schema definitions ----
# Each cell: list of (seed_lemma, pos, restrict_to_synsets_or_None)
# restrict_to_synsets: if given, only pull from these synset names (string list)
#                     to filter out unwanted senses for polysemous words.

SCHEMAS = {
    "UP_DOWN": {
        "UP_LITERAL": [
            ("ascend", "v", None),
            ("climb", "v", ["climb.v.01"]),       # the "go upward physically" sense
            ("soar", "v", ["soar.v.01"]),
            ("lift", "v", ["lift.v.01"]),         # raise/elevate physically
            ("hoist", "v", None),
        ],
        "UP_HAPPY": [
            ("cheerful", "a", None),
            ("elated", "a", None),
            ("jubilant", "a", None),
            ("gleeful", "a", None),
            ("joyful", "a", None),
            ("ecstatic", "a", None),
        ],
        "UP_MORE": [
            ("increase", "v", ["increase.v.01"]),
            ("grow", "v", ["grow.v.01"]),
            ("augment", "v", None),
            ("expand", "v", ["expand.v.02"]),     # become larger in quantity
            ("accrue", "v", None),
        ],
        "UP_STATUS": [
            ("promote", "v", ["promote.v.03"]),   # give a higher position
            ("distinguished", "a", None),
            ("prominent", "a", None),
            ("esteemed", "a", None),
        ],
        "UP_HEALTH": [
            ("thrive", "v", None),
            ("convalesce", "v", None),
            ("recuperate", "v", None),
            ("vigorous", "a", None),
            ("robust", "a", None),
        ],
        "DOWN_LITERAL": [
            ("descend", "v", ["descend.v.01"]),
            ("fall", "v", ["fall.v.01"]),         # move downward
            ("plunge", "v", ["plunge.v.01"]),
            ("plummet", "v", None),
            ("sink", "v", ["sink.v.01"]),
        ],
        "DOWN_SAD": [
            ("dejected", "a", None),
            ("morose", "a", None),
            ("glum", "a", None),
            ("melancholy", "a", None),
            ("forlorn", "a", None),
        ],
        "DOWN_LESS": [
            ("decrease", "v", ["decrease.v.01"]),
            ("shrink", "v", ["shrink.v.01"]),
            ("dwindle", "v", None),
            ("decline", "v", ["decline.v.04"]),
            ("diminish", "v", None),
        ],
        "DOWN_STATUS": [
            ("demote", "v", None),
            ("depose", "v", None),
            ("oust", "v", None),
            ("disgrace", "v", None),
        ],
        "DOWN_SICK": [
            ("ail", "v", None),
            ("deteriorate", "v", None),
            ("languish", "v", None),
            ("sickly", "a", None),
            ("frail", "a", None),
        ],
    },
    "IN_OUT": {
        "IN_LITERAL": [
            ("enclosed", "a", None),
            ("sealed", "a", None),
            ("contained", "a", None),
            ("encased", "a", None),
        ],
        "IN_MIND": [
            ("harbor", "v", ["harbor.v.02"]),     # hold (a thought) in mind
            ("ponder", "v", None),
            ("recall", "v", ["recall.v.04"]),     # bring to mind
            ("conceive", "v", ["conceive.v.01"]), # form (a thought)
        ],
        "IN_RELATIONSHIP": [
            ("married", "a", None),
            ("engaged", "a", None),
            ("committed", "a", None),
            ("partnered", "a", None),
        ],
        "IN_TIME": [
            ("within", "r", None),
            ("during", "r", None),
            ("amid", "r", None),
        ],
        "IN_DIFFICULTY": [
            ("mired", "a", None),
            ("stranded", "a", None),
            ("trapped", "a", None),
            ("ensnared", "a", None),
        ],
        "OUT_LITERAL": [
            ("released", "a", None),
            ("ejected", "a", None),
            ("extracted", "a", None),
            ("expelled", "a", None),
        ],
        "OUT_MIND": [
            ("forget", "v", ["forget.v.01"]),
            ("dismiss", "v", ["dismiss.v.04"]),   # cease to consider
            ("shed", "v", ["shed.v.02"]),         # cast off
        ],
        "OUT_RELATIONSHIP": [
            ("divorced", "a", None),
            ("separated", "a", None),
            ("estranged", "a", None),
            ("single", "a", ["single.s.01"]),     # not married/partnered
        ],
        "OUT_TIME": [
            ("expired", "a", None),
            ("outdated", "a", None),
            ("obsolete", "a", None),
            ("bygone", "a", None),
        ],
        "OUT_DIFFICULTY": [
            ("extricated", "a", None),
            ("rescued", "a", None),
            ("freed", "a", None),
            ("relieved", "a", None),
        ],
    },
    "BEVERAGE_sham": {
        "COFFEE": [
            ("espresso", "n", None),
            ("latte", "n", None),
            ("cappuccino", "n", None),
            ("mocha", "n", None),
            ("americano", "n", None),
        ],
        "ALCOHOL": [
            ("wine", "n", ["wine.n.01"]),
            ("beer", "n", None),
            ("whiskey", "n", None),
            ("vodka", "n", None),
            ("cocktail", "n", None),
        ],
        "TEA": [
            ("tea", "n", ["tea.n.01"]),
            ("matcha", "n", None),
            ("chai", "n", None),
            ("oolong", "n", None),
        ],
        "JUICE_WATER": [
            ("juice", "n", ["juice.n.01"]),
            ("lemonade", "n", None),
            ("smoothie", "n", None),
            ("water", "n", ["water.n.01"]),
        ],
    },
}


def expand_with_wordnet(seed, pos, restrict_synsets=None):
    """Return synonyms for a seed word, optionally filtered to specific synsets."""
    lemmas = set()
    if restrict_synsets:
        synsets = [wn.synset(name) for name in restrict_synsets]
    else:
        synsets = wn.synsets(seed, pos=pos)
    for s in synsets:
        for l in s.lemmas():
            name = l.name().replace("_", " ").lower()
            if " " in name:  # skip multi-word for now
                continue
            lemmas.add(name)
    return lemmas


def build_cell_vocab(cell_seeds):
    vocab = set()
    seed_provenance = defaultdict(list)  # word -> which seeds it came from
    for seed, pos, restrict in cell_seeds:
        syns = expand_with_wordnet(seed, pos, restrict)
        vocab.update(syns)
        for w in syns:
            seed_provenance[w].append(seed)
    return vocab, seed_provenance


# ---- Build all cell vocabularies ----
all_vocab = {}     # (schema, cell) -> set of words
all_provenance = {}
for schema, cells in SCHEMAS.items():
    all_vocab[schema] = {}
    all_provenance[schema] = {}
    for cell, seeds in cells.items():
        vocab, prov = build_cell_vocab(seeds)
        all_vocab[schema][cell] = vocab
        all_provenance[schema][cell] = prov


# ---- Check overlap within each schema ----
overlap_report = {}  # schema -> dict of overlap-pair -> set of overlapping words
for schema, cells in all_vocab.items():
    cell_names = list(cells.keys())
    overlap_report[schema] = {}
    for i, c1 in enumerate(cell_names):
        for c2 in cell_names[i+1:]:
            shared = cells[c1] & cells[c2]
            if shared:
                overlap_report[schema][f"{c1} ↔ {c2}"] = shared


# ---- Output ----
report_path = "/Users/macn/Documents/embeddingexp/corpus_vocabulary_report.md"
json_path = "/Users/macn/Documents/embeddingexp/corpus_vocabulary.json"

with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("=" * 100)
    out("Probe corpus vocabulary — WordNet-expanded from hand-curated seeds")
    out("=" * 100)
    out()

    for schema, cells in all_vocab.items():
        out("#" * 100)
        out(f"# {schema}")
        out("#" * 100)
        out()
        for cell, vocab in cells.items():
            out(f"## {cell}  ({len(vocab)} words)")
            sorted_words = sorted(vocab)
            # Group ~8 per line for readability
            for i in range(0, len(sorted_words), 8):
                out("  " + ", ".join(sorted_words[i:i+8]))
            out()

        out(f"### Cross-cell overlap within {schema}:")
        if not overlap_report[schema]:
            out("  (none — all cells decorrelated) ✓")
        else:
            for pair, shared in overlap_report[schema].items():
                out(f"  {pair}: {sorted(shared)}")
        out()
        out()

# Save the structured JSON
serializable = {
    schema: {cell: sorted(vocab) for cell, vocab in cells.items()}
    for schema, cells in all_vocab.items()
}
with open(json_path, "w") as f:
    json.dump(serializable, f, indent=2)

print(f"\nReport: {report_path}")
print(f"JSON:   {json_path}")
