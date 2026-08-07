"""exp166_prompt_verify.py — authored 2026-06-13.

Automated overlap check for the exp166 THIRD prompt set, per
PREREG_exp166.md "Integrity checks". Gate to freezing the prompts:
  (i)   no token matches any axis-defining word of ANY of the 8 active
        schemas;
  (ii)  zero verbatim overlap AND zero high-overlap near-dupe (token-set
        Jaccard < 0.5) with exp160's 40 (attn_entropy_lib.PROMPTS) or
        exp161's 40 (exp161....FRESH_PROMPTS);
  (iii) exactly 40 prompts, all unique.

The checker's own logic is tested at the bottom (exp161 v1.1 lesson:
the first overlap checker was buggy and silently checked the wrong
region). Run: ./lakoff/bin/python3 exp166_prompt_verify.py
"""
import re
from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML as S
from markedness_norm_protocol import SCHEMA_NAMES
from attn_entropy_lib import PROMPTS as EXP160_PROMPTS
from exp161_balance_entropy_prereg import FRESH_PROMPTS as EXP161_PROMPTS

# ----------------------------------------------------------------------
# CANDIDATE third prompt set (40). Everyday scenes, varied register,
# no balance/stability vocabulary, no schema axis words, novel topics.
# ----------------------------------------------------------------------
CANDIDATE = [
    "The barista frothed the milk and dusted cocoa across the foam.",
    "A magpie landed on the fence and eyed the picnic table.",
    "The mechanic drained the oil and wiped his hands on a rag.",
    "Her phone buzzed twice and then went silent in her pocket.",
    "The chef plated the fish and garnished it with dill.",
    "Children traced chalk animals on the playground tarmac.",
    "The cobbler stitched the sole and tapped in fresh nails.",
    "Rain freckled the windscreen as the wipers squeaked.",
    "The cashier counted the float and zipped the bag shut.",
    "A pot of soup simmered while she chopped parsley.",
    "The electrician labelled each wire and taped the panel shut.",
    "Geese crossed the meadow in a ragged honking row.",
    "The potter wet the clay and the wheel began to spin.",
    "She knitted two rows and miscounted the stitches.",
    "The fishmonger packed the crab in crushed ice.",
    "A drone skimmed the wheat and snapped photographs.",
    "The waiter recited the specials and refilled their water.",
    "Frost coated the lawn and the engine grumbled to life.",
    "The tailor pinned the hem and chalked a curve at the waist.",
    "Bees worked the lavender while the afternoon hummed.",
    "The plumber bled the radiator and the knocking finally quit.",
    "She sorted the laundry into colours and folded the towels.",
    "The butcher boned the lamb and tied it with kitchen twine.",
    "A toddler smeared yogurt across the tray and giggled.",
    "The florist clipped the stems and bound them with raffia.",
    "Thunder rolled somewhere distant and the dog whined.",
    "He greased the baking tin and cracked four eggs into the bowl.",
    "The usher tore the tickets and gestured toward the seats.",
    "Snow muffled the street and the ploughs began their rounds.",
    "The vet listened to the cat's chest and frowned a little.",
    "A busker strummed a chord and nodded at the gathering crowd.",
    "The cleaner mopped the lobby and roped off the wet tiles.",
    "Wasps had built a grey nest against the garden shed.",
    "The barman polished the tumblers and stacked them by size.",
    "A courier left the parcel with the neighbour at number nine.",
    "The hikers boiled snowmelt and shared a bar of chocolate.",
    "She defrosted the freezer and sponged the puddle dry.",
    "The referee blew the whistle and waved play on.",
    "Pigeons squabbled around a crust on the pavement.",
    "The seamstress threaded the machine and ran a test seam.",
]

# ----------------------------------------------------------------------
AXIS_WORDS = set()
for _sn in SCHEMA_NAMES:
    for _p, _n in S[_sn]:
        AXIS_WORDS.add(_p.lower()); AXIS_WORDS.add(_n.lower())


def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def check(prompts, axis_words=AXIS_WORDS, prior_sets=None):
    """Returns (ok, report dict). Pure; reused by the self-test."""
    prior_sets = prior_sets or {}
    rep = {"axis_hits": {}, "verbatim": [], "near_dupe": [], "n": len(prompts),
           "n_unique": len(set(prompts))}
    # (i) axis words
    for i, p in enumerate(prompts):
        bad = toks(p) & axis_words
        if bad:
            rep["axis_hits"][i] = (p, sorted(bad))
    # (ii) overlap with prior sets
    prior_all = [(name, q) for name, lst in prior_sets.items() for q in lst]
    prior_strings = {q for _, q in prior_all}
    for i, p in enumerate(prompts):
        if p in prior_strings:
            rep["verbatim"].append((i, p))
        for name, q in prior_all:
            j = jaccard(p, q)
            if j >= 0.5:
                rep["near_dupe"].append((i, round(j, 3), name, p, q))
    ok = (not rep["axis_hits"] and not rep["verbatim"] and not rep["near_dupe"]
          and rep["n"] == 40 and rep["n_unique"] == 40)
    return ok, rep


def _selftest():
    """The checker must FLAG known-bad inputs (exp161 v1.1 lesson)."""
    # planted axis word "balanced"
    ok, _ = check(["The acrobat balanced on the wire and grinned."] * 40)
    assert not ok, "selftest A: should flag axis word 'balanced'"
    # planted verbatim dupe of an exp161 prompt
    dup = EXP161_PROMPTS[0]
    ok, rep = check([dup] + ["unique scene number %d here today" % k for k in range(39)],
                    prior_sets={"exp161": EXP161_PROMPTS})
    assert rep["verbatim"], "selftest B: should flag verbatim dupe"
    # planted near-dupe (high Jaccard, not verbatim)
    base = "The carpenter measured twice and the shelf fit on the first try."
    near = "The carpenter measured twice and the shelf fit on the second try."
    ok, rep = check([near] + ["filler scene token %d apple orange grape" % k for k in range(39)],
                    prior_sets={"exp161": [base]})
    assert rep["near_dupe"], "selftest C: should flag near-dupe"
    # a clean set passes its own checks (no axis words, distinct)
    clean = ["clean scene alpha %d apple orange grape mango" % k for k in range(40)]
    ok, rep = check(clean, prior_sets={})
    assert ok, f"selftest D: clean set should pass, got {rep}"
    print("  self-test: PASS (checker flags axis words, verbatim, near-dupe; passes clean)")


if __name__ == "__main__":
    print("exp166 prompt-set overlap check")
    print("=" * 60)
    print(f"active schemas: {len(SCHEMA_NAMES)} | forbidden axis words: {len(AXIS_WORDS)}")
    _selftest()
    print("-" * 60)
    ok, rep = check(CANDIDATE, prior_sets={"exp160": EXP160_PROMPTS,
                                           "exp161": EXP161_PROMPTS})
    print(f"n prompts: {rep['n']}  unique: {rep['n_unique']}")
    if rep["axis_hits"]:
        print("\nAXIS-WORD HITS (must be empty):")
        for i, (p, bad) in rep["axis_hits"].items():
            print(f"  [{i}] {bad}  in: {p}")
    else:
        print("axis-word hits: NONE")
    if rep["verbatim"]:
        print("\nVERBATIM OVERLAP (must be empty):")
        for i, p in rep["verbatim"]:
            print(f"  [{i}] {p}")
    else:
        print("verbatim overlap with exp160/exp161: NONE")
    if rep["near_dupe"]:
        print("\nNEAR-DUPES (Jaccard >= 0.5, must be empty):")
        for i, j, name, p, q in rep["near_dupe"]:
            print(f"  [{i}] J={j} vs {name}: {p}  ||  {q}")
    else:
        print("near-dupes (Jaccard >= 0.5): NONE")
    print("-" * 60)
    print("RESULT:", "PASS — safe to freeze" if ok else "FAIL — fix before freezing")
    if ok:
        import hashlib
        h = hashlib.sha256("\n".join(CANDIDATE).encode()).hexdigest()[:16]
        print(f"prompt-set checksum (sha256[:16]): {h}")
