"""
exp11_vocabulary_curated.py — hand-curated vocabulary per (schema, child).

WordNet helped us bootstrap (see exp10) but polysemy leaks meant we needed
manual curation. This is the curated result, treating each cell's vocabulary
as a deliberate methodological choice rather than auto-expansion.

Decorrelation rules enforced:
  - Within a schema: no word appears in multiple children's vocabularies
  - Across schemas: ideally no shared words (so cross-schema fingerprint
    overlap reflects real schema-relations, not lexical confounds)

Lexical-bleed avoided: words that literally contain the schema's pole-word
("upbeat" containing "up", "downcast" containing "down") are excluded —
they'd test schema-lexical-priming rather than schema-deployment.

Output: corpus_vocabulary_curated.json + readable .md report.
"""

import json
from collections import defaultdict


# ---- The curated vocabulary ----
# Each cell: list of canonical lexical items. Sentence construction will use
# natural morphology (climbed, climbing, etc) — the list is the conceptual
# core, not the exact strings in sentences.

VOCAB = {
    "UP_DOWN": {
        # === UP children ===
        "UP_LITERAL": [
            # physical upward motion of an object
            "ascend", "ascended", "ascending",
            "climb", "climbed", "climbing",
            "soar", "soared", "soaring",
            "hoist", "hoisted",
            "tower",
        ],
        "UP_HAPPY": [
            # mood improvement / positive affect
            # NOT included: lifted, uplifted (lexical bleed with literal pole)
            "elated", "ecstatic",
            "cheerful", "jubilant",
            "joyful", "gleeful",
            "exultant", "rhapsodic",
            "beaming", "radiant",
        ],
        "UP_MORE": [
            # quantity increase
            # NOT included: rose/rising (polysemous with literal), surged (also physical)
            "increase", "increased", "increasing",
            "grew", "growth", "growing",
            "augment", "augmented",
            "expand", "expanded",
            "accrue", "accrued",
            "multiplied",
        ],
        "UP_STATUS": [
            # social rank / prestige elevation
            # NOT included: elevated (also literal), advanced (also forward-motion)
            "promote", "promoted", "promotion",
            "esteem", "esteemed",
            "distinguish", "distinguished",
            "prominent", "prestigious",
            "honored", "eminent",
            "accolade",
        ],
        "UP_HEALTH": [
            # vitality / health improvement / recovery
            # NOT included: expanded (overlaps with MORE), boomed (also commerce)
            "thrive", "thriving",
            "convalesce", "convalescing",
            "recuperate", "recuperating",
            "vigor", "vigorous",
            "robust", "vitality",
        ],
        # === DOWN children ===
        "DOWN_LITERAL": [
            # physical downward motion
            "descend", "descended", "descending",
            "plunge", "plunged", "plunging",
            "plummet", "plummeted",
            "sink", "sank", "sinking",
            "tumble", "tumbled",
        ],
        "DOWN_SAD": [
            # mood deterioration / negative affect
            # NOT included: low, down (lexical bleed), sour (taste)
            "dejected", "morose",
            "glum", "melancholy",
            "forlorn", "saturnine",
            "somber", "sullen",
            "doleful", "woebegone",
        ],
        "DOWN_LESS": [
            # quantity decrease
            # NOT included: fell (overlaps with literal), dropped (polysemous)
            "decrease", "decreased",
            "shrink", "shrank", "shrinking",
            "dwindle", "dwindled",
            "diminish", "diminished",
            "decline", "declined",
            "wane", "waned",
        ],
        "DOWN_STATUS": [
            # social rank / prestige decline
            "demote", "demoted",
            "disgrace", "disgraced",
            "oust", "ousted",
            "discredit", "discredited",
            "depose", "deposed",
            "dethrone", "dethroned",
            "ostracize", "ostracized",
        ],
        "DOWN_SICK": [
            # illness / health deterioration
            # NOT included: dropped (literal), declined (quantity)
            "ail", "ailing",
            "deteriorate", "deteriorating",
            "languish", "languishing",
            "sickly", "frail",
            "enfeeble", "enfeebled",
            "infirm", "feeble",
        ],
    },
    "IN_OUT": {
        # === IN children ===
        "IN_LITERAL": [
            # physical containment
            "contain", "contained",
            "enclose", "enclosed",
            "seal", "sealed",
            "bottle", "bottled",
            "encase", "encased",
            "wrap", "wrapped",
        ],
        "IN_MIND": [
            # thoughts/memory inside mind
            "ponder", "pondered", "pondering",
            "contemplate", "contemplated",
            "meditate", "meditated",
            "ruminate", "ruminating",
            "harbor", "harbored",  # in the mental sense
            "recall", "recalled",
        ],
        "IN_RELATIONSHIP": [
            # romantic/social bonds
            "married", "marriage",
            "engaged", "engagement",
            "committed", "commitment",
            "partnered", "partnership",
            "betrothed",
        ],
        "IN_TIME": [
            # within a temporal period (mostly prepositional)
            "within", "during", "throughout",
            "amid", "amidst",
            "spans", "spanned", "spanning",
            "encompass", "encompassed",
        ],
        "IN_DIFFICULTY": [
            # caught in trouble
            # NOT included: trapped, stuck, caught (overlap with IN_LITERAL)
            "mired", "miring",
            "stranded",
            "ensnared", "snared",
            "embroiled", "beleaguered",
            "predicament",
            "quagmire",
        ],
        # === OUT children ===
        "OUT_LITERAL": [
            # emerged from physical containment
            "escape", "escaped", "escaping",
            "emerge", "emerged", "emerging",
            "eject", "ejected",
            "extract", "extracted",
            "release", "released",
            "expel", "expelled",
        ],
        "OUT_MIND": [
            # thought released from mind
            # NOT included: shed (overlaps), abandoned (also relationship)
            "forget", "forgot", "forgotten",
            "dismiss", "dismissed",
            "discard", "discarded",
            "banish", "banished",  # from mind
            "purge", "purged",  # from memory
        ],
        "OUT_RELATIONSHIP": [
            # relationship dissolution
            "divorce", "divorced",
            "separate", "separated", "separation",
            "estrange", "estranged",
            "split-up", "broke-up",
            "single", "unmarried",
        ],
        "OUT_TIME": [
            # outside the period / in the past
            "expire", "expired",
            "bygone",
            "outdated",
            "obsolete", "obsolescent",
            "elapsed",
            "lapse", "lapsed",
        ],
        "OUT_DIFFICULTY": [
            # extracted from difficulty
            # NOT included: freed (overlaps with literal), released (overlaps with literal)
            "extricate", "extricated",
            "rescue", "rescued",
            "liberate", "liberated",
            "relief", "relieved",
            "salvage", "salvaged",
        ],
    },
    "BEVERAGE_sham": {
        "COFFEE": [
            "coffee", "espresso", "latte",
            "cappuccino", "mocha", "americano",
            "decaf", "barista", "brewed",
            "beans",
        ],
        "ALCOHOL": [
            # NOT included: spirits (overlaps with UP_HAPPY's "spirits")
            "wine", "beer", "whiskey",
            "vodka", "cocktail", "gin",
            "rum", "bourbon", "ale",
            "champagne", "liquor",
        ],
        "TEA": [
            # WordNet-poor cell — adding manually
            "tea", "matcha", "chai",
            "oolong", "kombucha", "herbal",
            "earl-grey", "jasmine", "rooibos",
        ],
        "JUICE_WATER": [
            # non-alcoholic, non-coffee, non-tea
            "juice", "lemonade", "smoothie",
            "water", "sparkling", "mineral",
            "soda", "fizzy",
        ],
    },
}


# ---- Check overlaps ----
def all_words(vocab_dict):
    """Yield (schema, cell, word) for every word in every cell."""
    for schema, cells in vocab_dict.items():
        for cell, words in cells.items():
            for w in words:
                yield schema, cell, w.lower()


word_to_cells = defaultdict(list)
for schema, cell, word in all_words(VOCAB):
    word_to_cells[word].append((schema, cell))

within_schema_overlap = defaultdict(list)  # schema -> list of (word, cells)
cross_schema_overlap = []                    # list of (word, cells)
for word, cells in word_to_cells.items():
    if len(cells) > 1:
        schemas_with_word = {c[0] for c in cells}
        if len(schemas_with_word) == 1:
            # within-schema
            schema = next(iter(schemas_with_word))
            within_schema_overlap[schema].append((word, cells))
        else:
            cross_schema_overlap.append((word, cells))


# ---- Output ----
report_path = "/Users/macn/Documents/embeddingexp/corpus_vocabulary_curated_report.md"
json_path = "/Users/macn/Documents/embeddingexp/corpus_vocabulary_curated.json"

with open(report_path, "w") as f:
    def out(s=""):
        print(s)
        f.write(s + "\n")

    out("# Curated probe-corpus vocabulary")
    out()
    out("Hand-curated after exp10's WordNet expansion surfaced polysemy issues.")
    out("Decorrelation rules: no within-schema word duplication; no cross-schema duplication.")
    out("Lexical schema-bleed (words containing the literal pole) avoided.")
    out()

    for schema, cells in VOCAB.items():
        out(f"## {schema}")
        out()
        for cell, words in cells.items():
            out(f"### {cell} ({len(words)} words)")
            sorted_words = sorted(set(w.lower() for w in words))
            for i in range(0, len(sorted_words), 6):
                out("  " + ", ".join(sorted_words[i:i+6]))
            out()

    out("---")
    out()
    out("## Overlap check")
    out()
    out("### Within-schema overlaps:")
    if not within_schema_overlap:
        out("  (none) ✓")
    else:
        for schema, overlaps in within_schema_overlap.items():
            for word, cells in overlaps:
                cell_names = ", ".join(c[1] for c in cells)
                out(f"  - **{word}** appears in {schema}: {cell_names}")
    out()
    out("### Cross-schema overlaps:")
    if not cross_schema_overlap:
        out("  (none) ✓")
    else:
        for word, cells in cross_schema_overlap:
            locs = ", ".join(f"{c[0]}/{c[1]}" for c in cells)
            out(f"  - **{word}** appears in: {locs}")

    out()
    out("### Cell sizes:")
    for schema, cells in VOCAB.items():
        for cell, words in cells.items():
            n = len(set(w.lower() for w in words))
            marker = " ⚠" if n < 8 else ""
            out(f"  {schema}.{cell}: {n} words{marker}")

# Save the structured JSON
serializable = {
    schema: {cell: sorted(set(w.lower() for w in words)) for cell, words in cells.items()}
    for schema, cells in VOCAB.items()
}
with open(json_path, "w") as f:
    json.dump(serializable, f, indent=2)

print(f"\nReport: {report_path}")
print(f"JSON:   {json_path}")
