"""
project_axis_vocabulary.py

Consolidated anchor vocabularies for all project-constructed target axes.
Previously scattered across exp52, exp53, exp54, exp56, exp59, exp63, exp64,
exp66, exp69. Pulled together 2026-05-27 end-of-session.

Substrate throughout: glove-wiki-gigaword-300.

Reading order:
  1. CURRENT (post-exp69) basis axes — use these going forward
  2. ORIGINAL (pre-exp69) basis axes — kept for reproducibility but deprecated
  3. Auxiliary / derived constructs
  4. Lakoff schemas (see lakoff_canonical_vocabulary.py for those)

For axis-construction convention see exp60_final_basis.py:build_axis —
  mean of (w[a] - w[c]) over in-vocab pairs, unit-normalized.
"""

# ============================================================================
# CURRENT BASIS — 12 axes post-exp80
# ============================================================================
# C, W, ATTENTION, INTENTION, R, D, IO_CLEAN, DV, MB, EPISTEMIC_VALUE,
# ABSTRACT_CONCRETE, REAL_IMAGINARY
#
# A (RUSSELL_DIAGONAL) replaced by ATTENTION_CLEAN — see exp69 lab notebook.
# G (GOAL_DIRECTED) replaced by INTENTION_CLEAN — see exp69 lab notebook.
# DV (DECISION_VERDICT) was originally built as SELECTION in exp63 and
#   renamed through exp64 A3 once we saw the pole vocabulary.
# MB (MARKOV_BLANKET) built in exp64 Part B.
# EPISTEMIC_VALUE built in exp73 (Thread NEW, post-exp70 refinement).
#   Completes the EFE = C − W − EV decomposition at the basis level.
# ABSTRACT_CONCRETE built in exp76 (proposed by Niamh as missing primitive
#   that Lakoff-bootstrapped construction couldn't access). Confirmed
#   independent in exp81 (82% of variance outside the 11-axis basis).
# REAL_IMAGINARY (originally MODAL_STATUS) built in exp76 + refined in exp80.
#   Probe test (exp80) on out-of-anchor word pairs: 10/10 consistent.
#   Renamed from MOD because the probe test confirmed it captures the
#   real/imaginary distinction specifically, not the broader logician's modal
#   status. Largest off-diagonal in 12-axis basis: cos(ABS, REAL_IMAGINARY)
#   = +0.32 (substrate-real overlap; imagined content is more abstract).

# ----------------------------------------------------------------------------
# C — Integrated reward / wellbeing (exp54)
# ----------------------------------------------------------------------------
# Active-inference reading: value-pole of expected free energy (pragmatic value)
TARGET_REWARD_COMPOSITE_PAIRS = [
    ("flourishing",  "suffering"),
    ("thriving",     "struggling"),
    ("prospering",   "declining"),
    ("blessed",      "cursed"),
    ("fortunate",    "unfortunate"),
    ("fulfilled",    "ruined"),
    ("privileged",   "oppressed"),
    ("graced",       "plagued"),
    ("charmed",      "hexed"),
    ("favored",      "disfavored"),
    ("lucky",        "unlucky"),
    ("wholesome",    "broken"),
]

# ----------------------------------------------------------------------------
# W — Weight / cost (exp59)
# ----------------------------------------------------------------------------
# Active-inference reading: cost-pole of expected free energy
# OUTSTANDING: target_COST construction (Thread 2) to test whether
# computational-cost vocabulary recovers same axis as somatic-weight.
TARGET_WEIGHT_PAIRS = [
    ("heavy",          "weightless"),
    ("weighty",        "airy"),
    ("ponderous",      "buoyant"),
    ("burdensome",     "effortless"),
    ("laden",          "unburdened"),
    ("cumbersome",     "nimble"),
    ("leaden",         "feathery"),
    ("dense",          "wispy"),
    ("encumbered",     "unencumbered"),
    ("heavyweight",    "featherweight"),
    ("massive",        "delicate"),
    ("oppressive",     "lighthearted"),
]

# ----------------------------------------------------------------------------
# ATTENTION_CLEAN (exp69) — replaces A
# ----------------------------------------------------------------------------
# Active-inference reading: γ (perceptual precision / gain on prediction errors)
# Screened to avoid intentional-content bleed.
# Was: A (RUSSELL_DIAGONAL — exp52) which included focused/directed/attentive,
# all of which carry implicit intentional content; that A leaked into G's
# domain and produced the +0.56 entanglement.
ATTENTION_CLEAN_PAIRS = [
    ("noticing",    "missing"),
    ("perceiving",  "overlooking"),
    ("sensing",     "missing"),
    ("detecting",   "missing"),
    ("spotting",    "missing"),
    ("recognizing", "overlooking"),
    ("seeing",      "missing"),
    ("hearing",     "missing"),
    ("registering", "ignoring"),
    ("witnessing",  "missing"),
    ("observing",   "missing"),
    ("aware",       "unaware"),
]

# ----------------------------------------------------------------------------
# INTENTION_CLEAN (exp69) — replaces G
# ----------------------------------------------------------------------------
# Active-inference reading: π (policy precision / commitment to policy)
# Screened to avoid attentional-content bleed.
# Was: G (GOAL_DIRECTED — exp53) which included oriented/targeted, both of
# which carry attentional content.
INTENTION_CLEAN_PAIRS = [
    ("intending",  "drifting"),
    ("planning",   "improvising"),
    ("deciding",   "deferring"),
    ("committing", "hedging"),
    ("choosing",   "defaulting"),
    ("designing",  "improvising"),
    ("resolving",  "postponing"),
    ("scheduling", "winging"),
    ("plotting",   "freelancing"),
    ("aiming",     "drifting"),    # "aiming" leans attentional — flagged
    ("intending",  "stumbling"),
    ("plan",       "improvise"),
]

# ----------------------------------------------------------------------------
# R — Perceptual precision / regulability (exp52, V+A-residualized in exp55+)
# ----------------------------------------------------------------------------
# Active-inference reading: PC3 in Lakoff space; partial alignment with γ
# but distinct from ATTENTION_CLEAN.
TARGET_EQUILIBRIUM_RUNAWAY_PAIRS = [
    ("correcting",     "escalating"),
    ("adjusting",      "cascading"),
    ("recalibrating",  "snowballing"),
    ("righting",       "spiraling"),
    ("stabilizing",    "mushrooming"),
    ("regulating",     "ballooning"),
    ("moderating",     "surging"),
    ("tempering",      "propagating"),
    ("dampening",      "amplifying"),
    ("restraining",    "intensifying"),
    ("restoring",      "exploding"),
    ("atoning",        "raging"),
    ("mending",        "festering"),
    ("reconciling",    "ravaging"),
]

# ----------------------------------------------------------------------------
# D — Compression / surprisal / predictability (exp54)
# ----------------------------------------------------------------------------
TARGET_SURPRISAL_PAIRS = [
    ("familiar",      "unfamiliar"),
    ("routine",       "novel"),
    ("recognized",    "unrecognized"),
    ("anticipated",   "unanticipated"),
    ("foreseen",      "unforeseen"),
    ("commonplace",   "extraordinary"),
    ("mundane",       "astonishing"),
    ("typical",       "atypical"),
    ("customary",     "unprecedented"),
    ("rote",          "startling"),
    ("habitual",      "jarring"),
    ("everyday",      "shocking"),
]

# ----------------------------------------------------------------------------
# IO_CLEAN — Container-topology / spatial in-out
# ----------------------------------------------------------------------------
# See lakoff_canonical_vocabulary.py:IN_OUT_MML_CLEAN for the full list.
# Captures spatial-containment only; does NOT capture abstract self/other
# (MB does that — see below).

# ----------------------------------------------------------------------------
# DV — Decision-verdict (exp63, named in exp64 A3) — 8th basis axis
# ----------------------------------------------------------------------------
# Was originally constructed as target_SELECTION expecting it to subsume G
# and reframe IO. Refuted on both counts. Re-interpreted as the linguistic
# axis of evaluation OUTCOMES (chosen vs refuted, validated vs denied).
# Distinct from C (91.6% non-C residual). Distinct from IO_CLEAN (+0.017).
# Distinct from gating-PROCESS (which lives in G+A / ATT+INT).
TARGET_DECISION_VERDICT_PAIRS = [
    ("selected",   "rejected"),
    ("chose",      "refused"),
    ("picked",     "discarded"),
    ("admitted",   "denied"),
    ("accepted",   "declined"),
    ("kept",       "removed"),
    ("chosen",     "eliminated"),
    ("preferred",  "overlooked"),
    ("favored",    "excluded"),
    ("designated", "omitted"),
    ("highlighted", "neglected"),
]
# Originally included ("singled-out", "ignored") — hyphenated form dropped.

# ----------------------------------------------------------------------------
# MB — Markov-blanket / substrate self-other (exp64 Part B) — 9th basis axis
# ----------------------------------------------------------------------------
# Sub-thread C2 from original Entry 25. Picks up abstract self/other content
# (self, ego, soul, identity, autonomy) that IO_CLEAN structurally missed.
# Max |cos| with rest of basis = +0.137 (with IO_CLEAN). Independent.
TARGET_MARKOV_BLANKET_PAIRS = [
    ("self",          "other"),
    ("agent",         "environment"),
    ("internal",      "external"),
    ("mine",          "theirs"),
    ("own",           "foreign"),
    ("private",       "public"),
    ("subjective",    "objective"),
    ("introspection", "perception"),
    ("autonomous",    "dependent"),
    ("individual",    "collective"),
    ("personal",      "impersonal"),
    ("intrinsic",     "extrinsic"),
    ("endogenous",    "exogenous"),
]

# ----------------------------------------------------------------------------
# EPISTEMIC_VALUE (exp73) — 10th basis axis
# ----------------------------------------------------------------------------
# Active-inference reading: the epistemic-value term of EFE — the drive
# toward gathering information about hidden states / model parameters.
# Completes the basis-level EFE decomposition: EFE = C − W − EV.
#
# State-based anchors per exp70's refinement: pure curiosity-state vocabulary,
# NO activity-verbs (investigate / explore / probe / inquire / seek — those
# load on INT_CLEAN, see exp70). The state/activity split is empirically
# confirmed: state-curiosity-words load 0.34–0.55 on EV; activity-verbs load
# 0.30–0.40 on INT and ≤0.16 on EV.
#
# Cleanest member of basis: max |cos| with rest of basis = +0.154 (with DV).
# The DV coupling comes from EV's negative pole ("already-decided / authorized
# / customary / dismissed") sharing register with DV's rejection-side
# vocabulary. Not anchor-bias — substrate-real overlap between incurious-state
# and verdict-rendered-state.
#
# Falsifiers cleared: max |cos| < 0.40 (derived state ruled out),
# cos with INT = −0.137 (activity-verb leakage ruled out).
TARGET_EPISTEMIC_VALUE_PAIRS = [
    ("curious",     "indifferent"),
    ("intrigued",   "dismissive"),
    ("fascinated",  "bored"),
    ("inquisitive", "incurious"),
    ("puzzled",     "settled"),
    ("wondering",   "knowing"),
    ("marveling",   "dismissing"),
    ("awestruck",   "jaded"),
    ("mystified",   "certain"),
    ("engaged",     "blase"),   # blasé without diacritic (GloVe stripping)
]

# ----------------------------------------------------------------------------
# ABSTRACT_CONCRETE (exp76) — 11th basis axis
# ----------------------------------------------------------------------------
# Proposed by Niamh after exp75's modest coverage finding suggested Lakoff-
# bootstrapped construction was missing major operators. Added in exp76 with
# +5.0pp marginal coverage across all categories (largest gains on
# ABSTRACT_FORMAL +8.4pp and MODAL_ACTUAL +7.3pp as predicted, broad lift
# elsewhere). Cross-bleed verified in exp77: max |cos| with 10-axis basis
# = 0.273 (with W).
#
# Confirmed as independent primitive in exp81: regressing ABS onto the full
# 11-axis basis (with REAL_IMAGINARY) gives R² = 18.4% — ~82% of ABS's
# variance is outside everything else we have. ABS captures structure that
# no combination of the other 11 primitives recovers.
#
# Pole vocabulary positive (abstract side): kantian, propounded, conceptions,
# posits, idealist, materialist, epistemology, hegelian, metaphysics. Clean
# philosophical-abstract cluster.
ABSTRACT_CONCRETE_PAIRS = [
    ("abstract",     "concrete"),
    ("theoretical",  "practical"),
    ("conceptual",   "physical"),
    ("general",      "specific"),
    ("idea",         "object"),
    ("principle",    "instance"),
    ("intangible",   "tangible"),
    ("notion",       "thing"),
    ("categorical",  "particular"),
    ("ideal",        "material"),
]

# ----------------------------------------------------------------------------
# REAL_IMAGINARY (exp76 → exp80 refinement) — 12th basis axis
# ----------------------------------------------------------------------------
# Originally constructed in exp76 as MODAL_STATUS with 10 pairs. Refined in
# exp80 to: (a) remove broad-distribution function words (could/can/is/
# established) that pulled compound-token register noise, (b) eliminate
# duplicate words (actual ×2, real ×2), (c) remove "theoretical" which
# cross-bled with ABSTRACT_CONCRETE, (d) add (imaginary, real) per Niamh's
# insight that this might be the foundational form of the primitive.
#
# Renamed MOD → REAL_IMAGINARY in exp80 because the probe test confirmed
# the axis captures the real/imaginary distinction specifically, not the
# broader logician's modal status. Probe test: 10 real/imaginary word pairs
# OUTSIDE the anchor list (imagination/perception, fantasy/memory, dream/
# experience, fiction/history, myth/fact, speculation/observation, vision/
# witness, supposition/evidence, conjecture/data, rumor/report) — all 10
# consistent (imagined-side loads more positively than real-side), magnitudes
# up to +0.56.
#
# Pole vocabulary asymmetric: negative pole (real side) is clean evidence/
# verification cluster (demonstrated, confirmed, showed, verified, noted,
# achieved, reported, documented, concluded). Positive pole (imaginary
# side) has clean signal (imaginary, conjectural, daydreams, sword-and-
# sorcery, post-modern, steampunk) mixed with register noise.
#
# Cos with ABSTRACT_CONCRETE = +0.32 — substrate-real overlap (imagined
# content is more abstract than real content). Largest off-diagonal in
# 12-axis basis. Still under 0.35 threshold.
#
# Rotation-imagining finding (exp81): the mean delta vector v(imagined_i) −
# v(real_i) across 14 probe pairs has cos = +0.60 with REAL_IMAGINARY, but
# also +0.42 with ABS, −0.31 with W, −0.26 with INT. So "imagining" as a
# computational operation is a multi-axis rotation in word-vector space,
# not a single-axis transformation through REAL_IMAGINARY alone.
REAL_IMAGINARY_PAIRS = [
    ("hypothetical",   "actual"),
    ("imagined",       "observed"),
    ("imaginary",      "real"),
    ("fictional",      "factual"),
    ("counterfactual", "demonstrated"),
    ("speculative",    "confirmed"),
    ("conjectural",    "verified"),
    ("presumed",       "proven"),
    ("notional",       "materialized"),
    ("alleged",        "documented"),
]


# ============================================================================
# DEPRECATED basis axes — replaced by ATTENTION_CLEAN and INTENTION_CLEAN
# Kept for reproducibility of exp52–exp68.
# ============================================================================

# ----------------------------------------------------------------------------
# A_RUSSELL_DIAGONAL (DEPRECATED — replaced by ATTENTION_CLEAN, see exp69)
# ----------------------------------------------------------------------------
# Originally constructed in exp52 as "valence-balanced salience" anchors;
# turned out to land on Russell's V×A diagonal rather than salience.
# Subsequently used as A_aff in exp54 onward.
# DEPRECATED because cross-bleed analysis (exp69) showed these anchors
# carry substantial intentional content (focused/directed/attentive imply
# agentive engagement, not pure perception). Use ATTENTION_CLEAN instead.
TARGET_SALIENCE_PAIRS = [  # = A_RUSSELL_DIAGONAL
    ("important",     "unimportant"),
    ("urgent",        "idle"),
    ("salient",       "irrelevant"),
    ("attentive",     "inattentive"),
    ("focused",       "unfocused"),
    ("directed",      "diffuse"),
    ("prominent",     "inconspicuous"),
    ("noticeable",    "unnoticeable"),
    ("foregrounded",  "backgrounded"),  # backgrounded was OOV in GloVe
    ("highlighted",   "overlooked"),
    ("conspicuous",   "unobtrusive"),
    ("pronounced",    "muted"),
]

# ----------------------------------------------------------------------------
# G_GOAL_DIRECTED (DEPRECATED — replaced by INTENTION_CLEAN, see exp69)
# ----------------------------------------------------------------------------
# Originally constructed in exp53 to operationalize policy precision.
# Reasonable axis but its anchors include attentional vocabulary
# (oriented/targeted) which leaks into A's domain. Use INTENTION_CLEAN
# instead.
TARGET_GOAL_DIRECTED_PAIRS = [
    ("pursuing",     "idling"),
    ("aiming",       "wandering"),
    ("purposeful",   "aimless"),
    ("deliberate",   "accidental"),
    ("motivated",    "unmotivated"),
    ("intentional",  "unintentional"),
    ("resolute",     "hesitant"),
    ("committed",    "uncommitted"),
    ("driven",       "becalmed"),
    ("oriented",     "disoriented"),     # attentional bleed
    ("targeted",     "untargeted"),       # attentional bleed
    ("decided",      "undecided"),
    ("chasing",      "dawdling"),
    ("ambitious",    "complacent"),
]


# ============================================================================
# AUXILIARY / DERIVED CONSTRUCTS — explored but not basis members
# ============================================================================

# ----------------------------------------------------------------------------
# B_VALUE_PURE (exp54) — lost the PC1 comparator
# ----------------------------------------------------------------------------
# Pure preference vocabulary; C beat it as PC1 candidate (+0.32 vs +0.54).
TARGET_VALUE_PURE_PAIRS = [
    ("preferred",    "dispreferred"),   # dispreferred was OOV
    ("wanted",       "unwanted"),
    ("sought",       "shunned"),
    ("cherished",    "loathed"),
    ("loved",        "hated"),
    ("treasured",    "abhorred"),
    ("valued",       "devalued"),
    ("welcomed",     "rebuffed"),
    ("embraced",     "rejected"),
    ("approached",   "avoided"),
    ("adored",       "detested"),
    ("admired",      "scorned"),
]

# ----------------------------------------------------------------------------
# EE_EXPLOIT_EXPLORE (exp56) — failed PC2 alignment, dropped
# ----------------------------------------------------------------------------
# Was at wrong abstraction level (corporate-optimization-vs-investigation
# register). Niamh's gathering-information primitive is target_EPISTEMIC_VALUE
# (queued, see BASIS_TESTS_TODO.md), not EE.
TARGET_EE_PAIRS = [
    ("exploit",       "explore"),
    ("harvest",       "forage"),
    ("specialize",    "diversify"),
    ("entrench",      "venture"),
    ("optimize",      "experiment"),
    ("routine",       "novelty"),
    ("consolidate",   "branch"),
    ("refined",       "exploratory"),
    ("perfecting",    "probing"),
    ("capitalize",    "prospect"),
    ("rehearsing",    "discovering"),
    ("mastery",       "investigation"),
]

# ----------------------------------------------------------------------------
# PROGRESS (exp66) — Lakoff-schema-level, not basis-level
# ----------------------------------------------------------------------------
# Lives in FB-G plane (cos with FB = +0.59, cos with G = +0.46). Lakoff
# PROGRESS-IS-FORWARD-MOTION confirmed. Best single-axis disentangler of
# A-G (Δ = −0.07) — meaningful but the bigger explanation was anchor bias
# (exp69). PROGRESS is a derived construct, not a basis primitive.
TARGET_PROGRESS_PAIRS = [
    ("advancing",    "regressing"),
    ("progressing",  "stalling"),
    ("gaining",      "losing"),
    ("nearing",      "distancing"),
    ("closing",      "receding"),
    ("improving",    "deteriorating"),
    ("mounting",     "dwindling"),
    ("accelerating", "decelerating"),
    ("proceeding",   "halting"),
    ("developing",   "declining"),
]

# ----------------------------------------------------------------------------
# TIME_PROTO (exp61b) — quick-pass construction for Thread 2.5
# ----------------------------------------------------------------------------
# Confirmed orthogonal to R and to the shared anisotropy direction; cos
# with proto-TIME = 0.015 for shared-pole-direction. Full target_TIME
# construction queued (Thread 2.5) with cross-bleed screening against
# INTENTION_CLEAN (intention is mildly future-leaning per exp68).
TARGET_TIME_PROTO_PAIRS = [
    ("past",         "future"),
    ("yesterday",    "tomorrow"),
    ("before",       "after"),
    ("earlier",      "later"),
    ("ancient",      "modern"),
    ("old",          "new"),
    ("remembered",   "anticipated"),
    ("begun",        "pending"),
    ("completed",    "planned"),     # planned overlaps INT_CLEAN — flagged
    ("precedes",     "follows"),
    ("historical",   "upcoming"),
]


# ============================================================================
# Test-of-attempts to construct subject/object-of-intentional-acts (exp69)
# ============================================================================
# RESULT: construction failed face-validity. Past-participle anchors
# (observer/observed etc.) created severe function-word and proper-noun
# anisotropy. Subject/object grammatical roles may not be cleanly
# recoverable in distributional semantics. Kept for documentation.
INT_STRUCT_PAIRS_FAILED = [
    ("observer",    "observed"),
    ("subject",     "object"),
    ("witness",     "witnessed"),
    ("speaker",     "spoken"),
    ("writer",      "written"),
    ("teacher",     "taught"),
    ("giver",       "given"),
    ("lover",       "loved"),
    ("helper",      "helped"),
    ("perceiver",   "perceived"),
    ("interpreter", "interpreted"),
    ("knower",      "known"),
]
