"""
Canonical Lakoff image-schema vocabulary extracted from the Master Metaphor List.

Source: Lakoff, G., Espenson, J., & Schwartz, A. (1991). Master Metaphor List.
Second Edition. Cognitive Linguistics Group, University of California at
Berkeley. PDF copy at `/Users/macn/Documents/embeddingexp/METAPHORLIST.pdf`,
215 pages, organized by target domain (not by source-domain image schema).

Methodology note: The MML is organized by what gets *explained* metaphorically
(target domain), not by image-schema source. So extracting "all UP-DOWN
vocabulary" requires scanning many files where UP-DOWN appears as a recurring
source-domain vocabulary. The extraction below was done by six subagents in
parallel, each focused on one schema cluster. Page citations in comments refer
to the document-internal page numbers printed at the bottom of each PDF page.

Where Lakoff does not have a named primary image-schema for a concept we care
about (e.g., EXISTENCE/NON-EXISTENCE — the becoming/unbecoming axis), we
compose vocabulary from Lakoff's adjacent attested metaphors (STATES ARE
LOCATIONS + CREATING IS BIRTHING + EXISTENCE IS LIFE etc.) and flag this
explicitly. The composed-axis status is documented per-list below.

This file is intended as a citable canonical vocabulary source for word2vec
axis construction experiments. Use it instead of hand-curated word pairs when
you want defensible "drawn from Lakoff (1991) MML" methodology.

For Master Metaphor List citation suggestion:
Lakoff, G., Espenson, J., & Schwartz, A. (1991). Master Metaphor List, 2nd ed.
Cognitive Linguistics Group, UC Berkeley.
"""

# ============================================================================
# UP-DOWN — verticality
# ----------------------------------------------------------------------------
# Lakoff's verticality schema appears as source-domain vocabulary across many
# target-domain files. Key MML attestations:
#   - MORE IS UP / AMOUNT IS VERTICALITY (p 14, Properties)
#   - HIGH STATUS IS UP / STATUS IS POSITION (pp 58-59, Comparison)
#   - CONTROL IS UP (p 67, Competition)
#   - WELL-BEING IS UP / HARMING IS LOWERING (p 50, Harm)
#   - EUPHORIC STATES ARE UP — cross-ref to HAPPY IS UP (p 177, Intoxication)
#   - MORAL IS UP / IMMORAL IS DOWN (p 187, Morality)
#   - GOOD IS UP / BAD IS DOWN (p 187, Morality)
#   - FORESEEABLE FUTURE EVENTS ARE UP (p 76, Time)
#
# Methodological caveat: HAPPY IS UP, RATIONAL IS UP, CONSCIOUS IS UP, DIVINE IS
# UP are canonical in *Metaphors We Live By* Ch.4 but NOT boxed as standalone
# entries in this 1991 second-draft MML. Where included below, sourced from
# MWLB rather than MML; flagged inline.
# ============================================================================
UP_DOWN_MML = [
    # Literal vertical motion (cross-file attestations)
    ("up", "down"),
    ("rise", "fall"),
    ("rose", "fell"),
    ("rising", "falling"),
    ("ascend", "descend"),
    ("raise", "lower"),
    ("climb", "drop"),
    ("lift", "sink"),
    ("above", "below"),
    ("over", "under"),
    ("top", "bottom"),
    ("high", "low"),
    ("higher", "lower"),
    ("upward", "downward"),
    # MORE IS UP / AMOUNT IS VERTICALITY (Properties p 14; Comparison p 62)
    ("more", "less"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("grew", "shrank"),
    ("growing", "shrinking"),
    # HIGH STATUS IS UP / STATUS IS POSITION (Comparison pp 58-59)
    ("upper", "lower"),
    ("promoted", "demoted"),
    ("promotion", "demotion"),
    ("superior", "inferior"),
    # CONTROL IS UP (Competition p 67)
    ("dominant", "submissive"),
    ("dominate", "submit"),
    ("commanding", "subordinate"),
    # GOOD IS UP (Morality p 187)
    ("good", "bad"),
    ("better", "worse"),
    ("best", "worst"),
    # MORAL IS UP (Morality p 187)
    ("moral", "immoral"),
    ("upright", "depraved"),
    ("upstanding", "underhanded"),
    ("noble", "base"),
    ("lofty", "lowly"),
    ("honorable", "ignoble"),
    # WELL-BEING IS UP / HEALTHY IS UP (Harm p 50)
    ("healthy", "sick"),
    ("well", "ill"),
    ("fit", "ailing"),
    ("strong", "weak"),
    ("uplifting", "degrading"),
    ("uplifted", "debased"),
    # HAPPY IS UP (MWLB Ch.4; cross-ref via EUPHORIC STATES ARE UP, Intoxication p 177)
    ("happy", "sad"),
    ("joyful", "depressed"),
    ("cheerful", "gloomy"),
    ("elated", "dejected"),
    ("euphoric", "despondent"),
    ("buoyant", "downcast"),
    ("soaring", "sinking"),
    # CONSCIOUS IS UP (MWLB Ch.4; supported by "Warm milk puts him to sleep" p 20)
    ("awake", "asleep"),
    ("alert", "drowsy"),
    ("conscious", "unconscious"),
    # RATIONAL IS UP (MWLB Ch.4; supported by "high-minded" p 187, "level-headed" p 147)
    ("rational", "emotional"),
    ("sober", "drunk"),
    # DIVINE IS UP (MWLB Ch.4; "heaven/hell" implicit)
    ("heaven", "hell"),
    ("heavenly", "infernal"),
    ("sacred", "profane"),
]


# ============================================================================
# IN-OUT — containment / CONTAINER schema
# ----------------------------------------------------------------------------
# Lakoff's CONTAINER schema is one of his most pervasive. MML attestations
# across 10+ target-domain children:
#   - STATES ARE LOCATIONS (p 8, States) — "He is in love"
#   - EXISTENCE IS LOCATION OUT OF CONTAINER (p 70, Existence)
#   - CONTAINMENT IN A PRESCRIBED SHAPE (p 10, States) — group membership
#   - DIFFICULTIES ARE CONTAINERS (p 75, Difficulties) — "in hot water"
#   - (BOUNDED) TIME IS A CONTAINER (p 78, Time) — "in three minutes"
#   - THE MIND IS A CONTAINER FOR OBJECTS (p 94, MentalObj)
#   - MEMORY IS A CONTAINER FOR OBJECTS (p 95, MentalObj)
#   - IDEAS ARE LOCATIONS + Constrained Thought (p 90, MentalLoc)
#   - EMOTIONS ARE ENTITIES WITHIN A PERSON / BODY IS CONTAINER (p 140, Emotion)
#   - EMOTIONS ARE LOCATIONS (p 144, Emotion)
#   - ANGER IS HOT FLUID IN A CONTAINER (p 149, Anger)
#   - A PROBLEM IS A LOCKED CONTAINER FOR ITS SOLUTION (p 195, Problems)
#
# CRITICAL METHODOLOGICAL NOTE: IN-OUT is valence-ambivalent across Lakoff
# children. IN=good (in love, included, remembered, married). IN=bad (trapped,
# mired, in trouble). A single IN-OUT axis built from mixed-valence pairs will
# entangle containment-topology with valence. For valence-clean IO testing,
# use only the "literal containment" pairs at the top (inside-outside,
# contained-released, etc.). For full Lakoff coverage, use all of them.
# ============================================================================
IN_OUT_MML = [
    # Literal containment (valence-neutral structurally)
    ("inside", "outside"),
    ("interior", "exterior"),
    ("within", "without"),
    ("enter", "exit"),
    ("entered", "exited"),
    ("contained", "released"),
    ("enclosed", "exposed"),
    ("sealed", "opened"),
    ("imported", "exported"),
    ("inhaled", "exhaled"),
    # MIND IS A CONTAINER / MEMORY IS A CONTAINER (pp 94-95) — IN=good
    ("remembered", "forgotten"),
    ("retained", "dismissed"),
    ("recalled", "forgotten"),
    ("memorized", "forgotten"),
    ("stored", "discarded"),
    # DIFFICULTIES ARE CONTAINERS (p 75) — IN=bad (inverted valence)
    ("trapped", "escaped"),
    ("stuck", "freed"),
    ("mired", "extricated"),
    ("stranded", "rescued"),
    ("imprisoned", "liberated"),
    ("ensnared", "released"),
    ("entangled", "disentangled"),
    # EMOTIONS ARE ENTITIES WITHIN A PERSON / ANGER IS HOT FLUID (pp 140, 149)
    ("contained", "vented"),
    ("bottled", "erupted"),
    ("suppressed", "expressed"),
    ("repressed", "released"),
    ("harbored", "banished"),
    # EMOTIONS ARE LOCATIONS / STATES ARE LOCATIONS (pp 8, 144) — mixed valence
    ("enamored", "estranged"),
    ("absorbed", "detached"),
    ("immersed", "withdrawn"),
    ("engrossed", "disengaged"),
    # RELATIONSHIPS ARE CONTAINERS — IN=good
    ("married", "divorced"),
    ("engaged", "estranged"),
    ("partnered", "separated"),
    ("united", "separated"),
    ("bonded", "estranged"),
    # GROUPS/CATEGORIES ARE CONTAINERS (Containment in Prescribed Shape, p 10) — IN=good
    ("included", "excluded"),
    ("admitted", "expelled"),
    ("accepted", "rejected"),
    ("enrolled", "dismissed"),
    ("inducted", "ousted"),
    ("welcomed", "banished"),
    # BOUNDED TIME IS A CONTAINER (p 78) — structural, no valence
    ("during", "after"),
    # A PROBLEM IS A LOCKED CONTAINER / EXISTENCE IS LOCATION OUT OF CONTAINER
    ("buried", "uncovered"),
    ("hidden", "revealed"),
    ("concealed", "exposed"),
    ("submerged", "emerged"),
    # IDEAS ARE LOCATIONS / Constrained Thought (p 90)
    ("entrapped", "liberated"),
    ("confined", "freed"),
    ("constrained", "unconstrained"),
]

# Subset of IN_OUT_MML for valence-clean testing (direction-only, no membership
# or difficulty valence). Use this when measuring containment-topology cleanly.
IN_OUT_MML_CLEAN = [
    ("inside", "outside"),
    ("interior", "exterior"),
    ("within", "without"),
    ("enter", "exit"),
    ("entered", "exited"),
    ("contained", "released"),
    ("enclosed", "exposed"),
    ("sealed", "opened"),
    ("inhaled", "exhaled"),
    ("during", "after"),
]


# ============================================================================
# FORWARD-BACK — directional motion / temporal direction
# ----------------------------------------------------------------------------
# Lakoff's FB schema is part of the SOURCE-PATH-GOAL / motion family. MML
# attestations:
#   - PROGRESS IS FORWARD MOTION (pp 16, 28, Action)
#   - NEGATIVE PROGRESS IS BACKWARD MOVEMENT (p 29)
#   - TIME IS SOMETHING MOVING TOWARD YOU (p 76, Time)
#   - FORESEEABLE FUTURE EVENTS ARE UP / AHEAD (p 76)
#   - LIFE / LOVE / CAREER IS A JOURNEY (pp 36-38)
#   - MORALITY IS A STRAIGHT PATH (p 185, Morality)
#   - LINEAR SCALES ARE PATHS (p 64) / Being Farther Along on a Path (p 57)
# ============================================================================
FORWARD_BACK_MML = [
    # Literal motion direction (PROGRESS IS FORWARD MOTION, pp 16, 28)
    ("forward", "backward"),
    ("ahead", "behind"),
    ("advance", "retreat"),
    ("onward", "back"),
    # Progress vs regress (NEGATIVE PROGRESS IS BACKWARD MOVEMENT, p 29)
    ("progress", "regress"),
    ("progressing", "regressing"),
    ("advancing", "retreating"),
    ("advanced", "retreated"),
    ("gain", "lose"),
    # Time (TIME IS MOTION; FUTURE IS AHEAD, p 76)
    ("future", "past"),
    ("upcoming", "former"),
    ("next", "previous"),
    ("later", "earlier"),
    ("after", "before"),
    ("approaching", "passing"),
    # Journey (LIFE/LOVE/CAREER IS A JOURNEY, pp 36-38)
    ("departing", "returning"),
    ("arrive", "depart"),
    # Development (CHANGE IS MOTION + PROGRESS, pp 15, 28-29)
    ("develop", "regress"),
    ("evolve", "devolve"),
    # Morality-as-path (MORALITY IS A STRAIGHT PATH, p 185)
    ("straight", "crooked"),
    ("upright", "deviant"),
    # Linear-scale-as-path (LINEAR SCALES ARE PATHS, p 64)
    ("leading", "trailing"),
]


# ============================================================================
# PATH-MOTION — motion vs stasis (the SOURCE-PATH-GOAL schema's motion polarity)
# ----------------------------------------------------------------------------
# SOURCE-PATH-GOAL is fundamentally about motion through stages; the polar
# axis is motion-vs-stasis. MML attestations:
#   - ACTION IS SELF-PROPELLED MOTION (p 27)
#   - LACK OF PROGRESS IS LACK OF MOTION (p 30) — "I got stuck"
#   - Caused Inability to Act is Prevention of Motion (p 32)
#   - Flow of Events is Flow of Water (p 41)
#   - Speed of Progress is Speed of Motion (p 29)
# ============================================================================
PATH_MOTION_MML = [
    # Core motion/stasis (ACTION IS MOTION, p 27)
    ("moving", "stationary"),
    ("moving", "still"),
    ("motion", "stillness"),
    ("going", "stopping"),
    ("going", "staying"),
    # Travel/journey verbs (LIFE IS A JOURNEY, p 36)
    ("traveled", "remained"),
    ("journeyed", "stayed"),
    ("walking", "standing"),
    ("walked", "stood"),
    ("departed", "stayed"),
    # Progress vs standstill (PROGRESS IS FORWARD MOTION, pp 16, 28)
    ("proceeding", "stuck"),
    ("advancing", "stalled"),
    ("flowing", "frozen"),
    # Speed (Speed of Progress is Speed of Motion, p 29)
    ("accelerating", "halting"),
    ("running", "resting"),
    # Path-traversal
    ("crossing", "lingering"),
    ("pursuing", "abandoning"),
]


# ============================================================================
# LIGHT-DARK — illumination, knowing, hope, goodness
# ----------------------------------------------------------------------------
# Lakoff's LIGHT-DARK schema appears in a dedicated file AND cross-cuts
# multiple target domains via canonical metaphors:
#   - LIGHT IS A FLUID / DARKNESS IS A SOLID / DARKNESS IS A COVER / LIGHT
#     IS A LINE (pp 179-180, LightDark file)
#   - IDEAS ARE LIGHT SOURCES (p 100, MentalImage)
#   - UNDERSTANDING IS SEEING (pp 86-87, Mental) — "shed light," "in the dark"
#   - HOPE IS LIGHT (p 152, Hope) — "bright hopes," "glimmer of hope"
#   - STRONG EMOTION IS BLINDING (p 147, Emotion)
#   - GOODNESS IS LIGHT / BADNESS IS DARKNESS (p 190, Morality)
#   - GOODNESS IS WHITE / BADNESS IS BLACK (p 190, Morality)
#
# Notable Lakoff observation: "Darkness is light's elder brother" (p 202) —
# attested as a kinship-asymmetry metaphor with traditional ordering.
# ============================================================================
LIGHT_DARK_MML = [
    # Core LightDark file (pp 179-180)
    ("light", "darkness"),
    ("bright", "dark"),
    ("sunlight", "darkness"),
    ("sunny", "shadowy"),
    ("dawn", "dusk"),
    ("daylight", "night"),
    ("day", "night"),
    # IDEAS ARE LIGHT SOURCES + UNDERSTANDING IS SEEING (pp 86-87, 100)
    ("bright", "dim"),
    ("illuminate", "obscure"),
    ("illuminated", "obscured"),
    ("clarify", "cloud"),
    ("clear", "murky"),
    ("clear", "cloudy"),
    ("sighted", "blind"),
    ("see", "blind"),
    # HOPE IS LIGHT (p 152) / despair pole
    ("glimmer", "gloom"),
    ("bright", "bleak"),
    ("radiant", "gloomy"),
    ("hopeful", "hopeless"),
    # STRONG EMOTION IS BLINDING (p 147)
    ("dazzle", "daze"),
    # GOODNESS IS LIGHT / BADNESS IS DARKNESS (p 190)
    ("light", "dark"),
    ("lightness", "darkness"),
    ("brightness", "blackness"),
    ("white", "black"),
    # Knowledge / truth (UNDERSTANDING IS SEEING + DARKNESS IS A COVER)
    ("revealed", "buried"),
    ("visible", "hidden"),
    ("transparent", "opaque"),
    ("shine", "darken"),
]


# ============================================================================
# EXISTENCE — coming-into-being vs going-out-of-being (becoming/unbecoming)
# ----------------------------------------------------------------------------
# METHODOLOGICAL CAVEAT: Lakoff does NOT posit a primary image-schema for
# EXISTENCE/NON-EXISTENCE. The vocabulary below is composed from Lakoff's
# canonical constellation of existence-change metaphors:
#   - EXISTENCE IS A LOCATION (HERE) (p 70, Existence)
#   - EXISTENCE IS LOCATION UP HERE (p 70)
#   - EXISTENCE IS LOCATION OUT OF CONTAINER (p 70)
#   - EXISTENCE IS LIFE (p 71)
#   - EXISTENCE IS VISIBILITY (p 71)
#   - EXISTENCE IS HAVING A FORM (p 71)
#   - CREATING IS MOVING TO A LOCATION (p 72, Creation)
#   - CREATING IS BIRTHING / CAUSATION IS PROGENERATION (p 72)
#   - CREATION IS CULTIVATION (p 73)
#   - CREATING IS MAKING (p 73)
#   - ACTIVE IS ALIVE (p 26, Action)
#   - FAILURE IS DEATH / SUCCESS IS LIFE (p 67, Competition)
#   - SOCIETY IS A BODY: birth → inception, death → collapse (p 211, Society)
#
# This is a composed axis, not a Lakoff-canonical primary schema. Cite
# accordingly in writeups.
# ============================================================================
EXISTENCE_MML = [
    # EXISTENCE IS A LOCATION (HERE) (p 70)
    ("arrived", "departed"),
    ("came", "went"),
    ("entered", "exited"),
    ("present", "absent"),
    ("here", "gone"),
    ("stayed", "left"),
    # EXISTENCE IS LOCATION OUT OF CONTAINER (p 70)
    ("emerged", "submerged"),
    ("arose", "sank"),
    ("surfaced", "sank"),
    ("rose", "fell"),
    # EXISTENCE IS VISIBILITY / EXISTENCE IS HAVING A FORM (p 71)
    ("appeared", "disappeared"),
    ("appeared", "vanished"),
    ("materialized", "vanished"),
    ("visible", "invisible"),
    ("appeared", "faded"),
    # EXISTENCE IS LIFE / FAILURE IS DEATH / SOCIETY IS A BODY (pp 71, 67, 211)
    ("born", "died"),
    ("birth", "death"),
    ("alive", "dead"),
    ("living", "dying"),
    # CREATING IS MAKING (Creation pp 72-73)
    ("created", "destroyed"),
    ("made", "unmade"),
    ("built", "demolished"),
    ("built", "destroyed"),
    ("constructed", "demolished"),
    ("formed", "dissolved"),
    ("shaped", "shattered"),
    ("generated", "annihilated"),
    ("produced", "destroyed"),
    ("forged", "broken"),
    # CREATING IS BIRTHING (p 72)
    ("conceived", "killed"),
    # CREATION IS CULTIVATION (p 73)
    ("planted", "uprooted"),
    ("grew", "withered"),
    ("sprouted", "withered"),
    ("flourished", "perished"),
    # Boundary / inception-collapse (SOCIETY IS A BODY p 211; aspectual)
    ("began", "ended"),
    ("started", "finished"),
    ("originated", "terminated"),
    ("founded", "abandoned"),
    ("established", "abolished"),
    ("rose", "collapsed"),
]


# ============================================================================
# FORCE — physical/causal/psychological force
# ----------------------------------------------------------------------------
# Lakoff's FORCE schema appears across MML files:
#   - CAUSED CHANGE IS FORCED MOTION (p 20, Causation)
#   - CAUSES ARE FORCES (p 23, Causation)
#   - FORCE IS A SUBSTANCE DIRECTED AT AN AFFECTED PARTY (p 170, Force)
#   - A FORCE IS A MOVING OBJECT (p 172, Force)
#   - PSYCHOLOGICAL FORCES ARE PHYSICAL FORCES (p 131, PsychologicalForces)
#   - LOGIC IS A FORCE THAT MOVES A MIND (p 125, Logic)
#   - LOGIC IS GRAVITY (p 126, Logic)
#   - OBLIGATIONS ARE FORCES (p 207, Responsibilities)
# ============================================================================
FORCE_MML = [
    ("push", "pull"),
    ("force", "yield"),
    ("compel", "release"),
    ("drive", "restrain"),
    ("press", "ease"),
    ("forceful", "yielding"),
    ("strong", "weak"),
    ("impel", "deter"),
    ("exert", "relax"),
    ("resist", "submit"),
    ("oppose", "comply"),
    ("propel", "halt"),
    ("coerce", "permit"),
    ("attract", "repel"),
    ("dominate", "submit"),
]


# ============================================================================
# BALANCE — equilibrium / stability
# ----------------------------------------------------------------------------
# Lakoff's BALANCE schema:
#   - Comparison of Importance is Weighing (p 62, Comparison)
#   - MENTAL ACCOUNTING / Results are Net Balances (p 136)
#   - MORAL ACCOUNTING (p 188, Morality)
#   - EMOTIONAL STABILITY IS BALANCE (implicit via topple/collapse, p 126)
#   - Compliance is Tightness vs Slackness (p 208)
#   - Theories Need Support / collapse (p 119, Debate)
# ============================================================================
BALANCE_MML = [
    ("balanced", "unbalanced"),
    ("stable", "unstable"),
    ("steady", "wobbly"),
    ("level", "tilted"),
    ("even", "uneven"),
    ("equal", "unequal"),
    ("upright", "fallen"),
    ("firm", "shaky"),
    ("solid", "wobbling"),
    ("settled", "tipping"),
    ("centered", "lopsided"),
    ("symmetric", "skewed"),
    ("aligned", "askew"),
    ("tight", "slack"),
    ("poised", "teetering"),
]


# ============================================================================
# DIFFICULTY-BURDEN — heaviness / obstacle / support
# ----------------------------------------------------------------------------
# Lakoff's DIFFICULTIES-AS-BURDEN schema:
#   - OBLIGATIONS ARE BURDENS (p 204, Responsibilities) — "weighed down,"
#     "heavy load," "shouldered the task"
#   - RESPONSIBILITIES ARE BURDENS (p 206)
#   - Difficulty is Hardness of Object (p 195, Problems)
#   - Obstacles to Action are Obstacles to Motion (p 31, Action)
#   - Easy Action is Easy Motion (p 31)
#   - Difficulties are Impediments to Travel (p 37)
#
# NOTE: "heavy" / "light" here is WEIGHT, not illumination. Don't confuse with
# LIGHT_DARK_MML's "light" / "dark" which is illumination.
# ============================================================================
DIFFICULTY_BURDEN_MML = [
    ("heavy", "light"),  # WEIGHT, not illumination
    ("burdensome", "effortless"),
    ("weighty", "trifling"),
    ("loaded", "unloaded"),
    ("encumbered", "unencumbered"),
    ("pressed", "relieved"),
    ("obstructed", "clear"),
    ("blocked", "open"),
    ("hindered", "aided"),
    ("stuck", "free"),
    ("hard", "easy"),
    ("rough", "smooth"),
    ("strenuous", "easy"),
    ("supported", "unsupported"),
    ("propped", "collapsed"),
]


# ============================================================================
# Convenience: registry of all canonical schema axes
# ============================================================================
LAKOFF_SCHEMAS_MML = {
    "UP-DOWN":          UP_DOWN_MML,
    "IN-OUT":           IN_OUT_MML,
    "IN-OUT_CLEAN":     IN_OUT_MML_CLEAN,
    "FORWARD-BACK":     FORWARD_BACK_MML,
    "PATH-MOTION":      PATH_MOTION_MML,
    "LIGHT-DARK":       LIGHT_DARK_MML,
    "EXISTENCE":        EXISTENCE_MML,   # composed, not a Lakoff primary schema
    "FORCE":            FORCE_MML,
    "BALANCE":          BALANCE_MML,
    "DIFFICULTY-BURDEN": DIFFICULTY_BURDEN_MML,
}


# Quick verification when run as script
if __name__ == "__main__":
    total_pairs = 0
    for name, pairs in LAKOFF_SCHEMAS_MML.items():
        print(f"  {name:>18}:  {len(pairs):>3} pairs")
        total_pairs += len(pairs)
    print(f"  {'TOTAL':>18}:  {total_pairs}")
