# Lab Notebook v5 — Substrate Primitives, Cross-Layer Mappings, and Architectural Correspondences

Started 2026-06-08. Continues from v4 (which ended 2026-06-06 with the
v5-territory framing about substrate-of-computation as the orienting
question).

---

## Morning summary for Niamh (read this first)

> v4 closed on the question "how does the substrate of computation determine
> what counts as a unit of meaning?" v5 covers exp127 through exp143, which
> turn out to address that question in several distinct ways. The headline,
> stated plainly:
>
> **In Pythia 410M, Lakoff-style organisational structure exists in three
> empirically distinct geometric forms, each captured by a different
> measurement, all present simultaneously.** Conflating them was the
> source of confusion across v3 and v4.
>
> The three geometries:
>
> 1. **Within-layer schema cluster structure** (exp123 already, refined
>    by exp138). Lakoff schemas are recoverable as a coordinated
>    relational system at every layer. Morphology is positioned on these
>    schemas at conceptually-plausible locations (our extrapolation, not
>    Lakoff's direct prediction). Captured by per-layer cos tests.
>
> 2. **Cross-layer metaphorical mappings** (exp143, dovetailing with
>    exp116's behavioural findings). UP-steering at L_inject produces
>    large, dose-responsive shifts in VAL and MAG projections at
>    downstream layers, far above a random-direction control. The model
>    has learned canonical Lakoff mappings (HAPPY-IS-UP, MORE-IS-UP) as
>    cross-layer transformations executed by attention and MLPs.
>    UP→VAL is ~4-5× stronger than UP→MAG.
>
> 3. **Substrate-architectural correspondences** (exp141 + exp142). One
>    Lakoff schema — BALANCE — turns out to ride a specific
>    architectural feature (LayerNorm's variance normalisation):
>    inflected suffixes have lower residual norms than their bases, and
>    norm-delta correlates r=0.85–0.97 with BALANCE projection. This is
>    *specific* to BALANCE; FORWARD-BACK does not ride positional
>    encoding (tested in exp142), LIGHT-DARK does not ride attention
>    (tested in exp142). Most schemas are learned-semantic; one is
>    architecturally anchored.
>
> Also documented here:
>
> - **Substrate-specific content encoding** (exp140): Pythia encodes
>   extent content words (deep, abyss, tall) preferentially on a
>   MAGNITUDE axis; GloVe encodes them on a DIRECTIONAL axis. The
>   abstract Lakoff UP-DOWN schema lives on the directional axis in both
>   substrates. Substrate-specific signature in *content-word
>   placement*, not in the abstract schema axis itself.
> - **Anisotropy contamination of morphology vectors** (exp137 → exp138):
>   pair-difference morphology directions have |cos with anisotropy|
>   ≈ 0.55–0.59, which collapsed all suffix directions into
>   superficial similarity. Per-layer anisotropy stripping recovers
>   differentiated suffix-specific schema mappings.
> - **Representational systems vs folk categories** (exp136 from v4
>   handoff, recapped here): Lakoff schemas, quantifiers, and
>   determiners cross-layer-preserve as coordinated systems; logical
>   operators do not (folk category, not a representational system).
> - **Massive-activation effect in middle layers** (exp141 TEST 0):
>   at L8–L20, raw bare-word pair-difference axes saturate to ±0.998
>   cos with anisotropy. This is the Sun et al massive-activations
>   phenomenon and confounds raw-axis tests in those layers.
>
> **What v5 does NOT claim**:
> - We have NOT shown that LAKOFF_UP is "magnitude" within-layer.
>   cos(LAKOFF_UP_clean, MAG_clean) ≈ 0 at every layer (exp141).
>   The MORE-IS-UP claim survives as a cross-layer transformation
>   (exp143), not a within-layer axis alignment.
> - We have NOT shown that the substrate-architectural correspondence
>   is general. It works for BALANCE. It does NOT work for FORWARD-BACK
>   or LIGHT-DARK. The "schemas ride architecture" hypothesis is
>   specific to LayerNorm-equilibrium and probably won't generalise.
> - We have NOT separated Lakoff's direct predictions from our
>   extrapolations carefully enough in earlier writeups. v5 attempts
>   that separation explicitly (see Entry 8).
>
> **Reframe**: Niamh's primary interest is *model embodiment*, not
> Lakoff per se. The substrate-embodiment thesis these three geometries
> jointly support is: Pythia recapitulates the *structural form* of
> embodied-cognitive organisation — substrate-given primitives as
> metaphorical source domains, cross-domain mappings as learned
> transformations, substrate-architectural features as anchoring axes
> — through statistical learning from text alone. The Lakoff framework
> is the analytical tool that lets us recognise these patterns; the
> deeper claim is about embodiment in non-bodily substrates.

---

## The orienting question (continued from v4)

v4 ended on:
> "How does the substrate of computation determine what counts as a
> unit of meaning?"

v5's contribution: the substrate determines this in *multiple ways
simultaneously*. There isn't one organising principle; there's a
stratified structure:

- **Architectural primitives** (LayerNorm, residual stream, causal
  attention): some directly become representational axes (BALANCE);
  most don't (FB, LD).
- **Mathematical primitives** of vector spaces (magnitude as norm):
  become content-encoding axes for substrate-natural content
  (extent/quantity words on MAG).
- **Learned cluster structure** (Lakoff schemas as relational system):
  emerges from statistical learning, layer-stable, interpretable as
  embodied-cognitive organisation.
- **Learned cross-domain transformations** (canonical Lakoff
  metaphors): live as multi-layer composition functions, not single
  directions.

The substrate provides certain features for free; the rest is learned
on top. The empirical question per Lakoff schema is which kind of
structure it is — and the answer is heterogeneous.

---

## Entry 1 — exp127–128: substrate-specific content-word encoding

### Background

v3 / v4 work on the cleaned-UP steering vector kept producing magnitude
effects (pool depth co-shifts up with patient height under steering,
exp116). At the same time, exp127 traced Pythia's per-layer projection
of "deep" / "deeper" / "deepest" on the UP-DOWN axis: small but
positive projections, not strongly directional, growing through middle
layers.

### exp127 design

Build clean UP-DOWN at every layer. Project the words {deep, deeper,
deepest, abyss, shallow} on UP per layer. Track per-layer trajectory.

### exp127 result

The projections are non-trivially positive on UP for "deep" and
"deeper" — that is, "deep" projects toward UP, not toward DOWN. This
is anti-spatial-direction (deep is conventionally downward) and
suggested that the model's "UP" content captures something other than
bipolar verticality. The cluster reading (UP = MORE/HIGH-STATUS/HAPPY
bundle) became the working hypothesis.

### exp128

Comparative suffix (-ER) has its own systematic projection on the UP
axis across layers — positive, layer-stable, increasing through middle
layers. The morphological operator is positioned on UP at a location
consistent with "more-of-the-base." This was the seed of the
morphology-on-schemas hypothesis later tested in exp130–138.

### Caveat

Not a clean isolation between "magnitude" and "direction" content at
this point; that took until exp140 to disentangle.

---

## Entry 2 — exp136: representational systems vs folk categories

### Setup

Per-axis cross-layer stability test for several clusters of
hand-curated word axes. Question: which clusters of "operators"
actually form coordinated representational systems in the model, vs
which are folk categories that don't correspond to representational
reality?

Clusters tested:
- Lakoff schemas (8 axes)
- Quantifiers (some, all, many, few, etc.)
- Determiners (the, a, this, that, etc.)
- Logical operators (and, or, not, if, then, etc.)
- Modal operators (must, can, may, might, etc.)

### Result

Lakoff schemas, quantifiers, and determiners show cross-layer
preservation of their inter-axis matrix structure (per-cluster
mean off-diagonal cos > weak null baseline). Logical operators do
NOT — they pass individually (each operator's axis is layer-stable)
but their cross-layer matrix preservation is at the null level.

### Interpretation

"Logical operators" is a folk category from formal logic; it does not
correspond to a representational system in Pythia. AND and OR are
individually represented, but the geometric *relationships* between
them aren't preserved across layers in the way schemas' relationships
are. Caveat to flag for paper: cross-layer matrix preservation tests
can pass for individual stability while failing for system-level
structure — and the distinction matters.

### Caveat (from exp139, see Entry 4)

Cross-layer matrix preservation tests need careful nulls. Generic
word-contrast matrices preserve their shape across layers regardless
of specific structure — so a positive cross-layer matrix preservation
result isn't on its own evidence of *specific* representational
structure unless the null is calibrated correctly. We did this for
Lakoff vs logical-operator comparison, but it's a limitation of the
methodology more broadly.

---

## Entry 3 — exp137 → exp138: morphology anisotropy contamination

### What we expected to find (exp130 era)

We had built morphology axes as pair-differences:
  `direction(-ING) = mean(walking - walk, running - run, ...)`

The expectation: each suffix would land at a distinct location on the
Lakoff schemas, reflecting its grammatical-conceptual content.

### What we actually found (exp130 raw)

All suffixes pointed in approximately the same direction. Suffix-suffix
cosine mean was +0.77. The heatmap looked uninformative — every suffix
projecting modestly on every schema.

### exp137: discovering the confound

Checked |cos with anisotropy| for each pair-difference suffix
direction. Found values 0.55–0.59 — substantially anisotropy-aligned.
Pair-difference morphology directions are NOT anisotropy-clean,
contrary to the schema axes (which exp123 verified clean to <0.03).

Possible mechanism: inflected forms (longer, more complex tokens) and
their bases have systematically different residual norms (foreshadowing
the BALANCE-as-norm finding in exp141). The difference vector picks up
this norm asymmetry as a substrate-anisotropy component, which then
dominates the morphology axis.

### exp138: stripped morphology heatmap

Re-built suffix directions with explicit per-layer anisotropy strip
PLUS frequency strip. Suffix-suffix cosine drops from +0.77 to ~+0.20
on average. Differentiated suffix-specific schema mappings emerge:

| Suffix | Mapping | Mean cos |
|---|---|---|
| -ING progressive | PATH-MOTION positive | ~+0.20 |
| -ED past tense | FORWARD-BACK negative | ~-0.18 |
| un- negation | LIGHT-DARK negative | ~-0.15 |
| re- repetition | FORWARD-BACK negative | ~-0.16 |
| -S plural | (no strong specific schema) | — |
| -ER comparative | (mild positive on many) | — |
| -EST superlative | (similar to ER) | — |

All inflected suffixes share a BALANCE-negative component (mean across
suffixes around -0.10 to -0.20). This was initially read as Lakoffian
"inflection = departure from equilibrium" but exp141 (Entry 6) shows
it's literally the architectural BALANCE-as-norm signal.

### Honest framing for the paper

These suffix-schema mappings are **our extrapolation from Lakoffian
metaphor theory**, not Lakoff's direct predictions. He has
TIME-IS-MOTION-FORWARD and PAST-IS-BEHIND as discussed metaphors;
applying these to English morphology specifically is an extension we
made. The finding is novel and consistent with Lakoffian principles;
it should not be presented as "predicted by cognitive linguistics."

---

## Entry 4 — exp139: cross-layer matrix preservation caveat

### Test

Take the suffix × Lakoff-schema matrix at L4 and at L20. Compute their
similarity (cos between flattened matrices, or Frobenius normalised
inner product). Compare to a null: build a matrix of arbitrary
word-pair differences against the same schemas, same dimensions,
compare cross-layer.

### Result

Both real (morphology × schema) and null (arbitrary word-diff × schema)
matrices preserve their shape across layers well. The real
preservation is somewhat higher but not categorically so. **Passing
the cross-layer matrix preservation test is not sufficient evidence of
a coordinated representational structure** — generic word-contrast
matrices preserve their shape too.

### Implication

The morphology-on-schemas claim (exp138) should be made on the
basis of the *specific predicted suffix-schema mappings holding at
specific locations*, not on the basis of cross-layer matrix
preservation as such. The matrix-preservation tests in v4 (exp136)
for cluster classification need to be read with this caveat too —
they're informative when comparing specific clusters to each other,
not as standalone tests of "is this a coordinated system."

---

## Entry 5 — exp140: MAGNITUDE vs DIRECTIONAL substrate primitive

### Setup

Build two clean axes from purified word lists, anisotropy + freq
stripped at each layer:
- MAGNITUDE axis: {huge, big, large, enormous, vast, ...} minus
  {tiny, small, little, minute, ...}
- DIRECTIONAL axis: {above, over, atop, upward, upper, ...} minus
  {below, under, underneath, downward, lower, ...}

Compare to Lakoff UP-DOWN axis (mixed direction + content terms).
Test where extent content words (deep, deeper, abyss, tall, shallow)
project preferentially.

### Result (L12, Pythia 410M)

Test words from the "depth_extent" category project on MAG with
|cos| 1.6–4× larger than on DIR.

| Word | on MAG | on DIR | winner |
|---|---|---|---|
| deep | +0.085 | +0.029 | MAG (3×) |
| deeper | +0.156 | +0.039 | MAG (4×) |
| deepest | +0.183 | +0.048 | MAG (4×) |
| abyss | +0.111 | +0.035 | MAG (3×) |
| tall | +0.072 | +0.046 | MAG (1.6×) |

The Lakoff UP-DOWN axis itself aligns more with DIR than MAG in both
Pythia and GloVe (consistent with UP-DOWN being a direction-anchored
schema). But content words for *extent* land on MAG in Pythia.

### GloVe comparison

Same test words projected on GloVe-built MAG and DIR axes show the
opposite pattern: extent words project on DIR with |cos| 2–4× larger
than on MAG. GloVe encodes "deeper" as ~negative on DIR (downward
direction) — bipolar verticality. Pythia doesn't.

### Interpretation

Pythia's substrate (transformer residual stream with normed vectors
+ LayerNorm) makes magnitude a substrate-intrinsic primitive in a way
GloVe's substrate doesn't. The model preferentially uses magnitude to
encode extent content. This is the substrate-primitive signature in
content-word placement.

Important honest qualifier: the abstract Lakoff UP-DOWN axis is not
particularly magnitude-aligned in Pythia (cos with MAG ~ 0 across all
layers per exp141). The substrate-primitive finding is specifically
about content-word encoding, not about how the abstract schema axis
is positioned. These are two different claims.

---

## Entry 6 — exp141: substrate primitives, the UP decomposition, and BALANCE-as-norm

### Setup

Five tests:
- TEST 0: is VALENCE riding the anisotropy direction? (raw axes, no
  stripping)
- TEST 1: layer trajectory of clean LAKOFF_UP — cos with MAG, DIR, VAL
- TEST 2: BALANCE as norm-deviation primitive (Δ‖residual‖ vs BALANCE
  projection per suffix pair)
- TEST 3: cross-layer stability of all axes
- TEST 4: orthonormalised Gram-Schmidt decomposition of LAKOFF_UP onto
  {MAG, DIR, VAL} basis, with random-axis control

### TEST 0 result: valence-as-anisotropy is confounded by massive activations

Raw axis-anisotropy alignment per layer:

| Layer | MAG·aniso | DIR·aniso | VAL·aniso | LAK·aniso | BAL·aniso |
|---|---|---|---|---|---|
| L0 | -0.071 | -0.044 | +0.199 | +0.067 | -0.019 |
| L4 | -0.286 | -0.035 | +0.537 | +0.191 | +0.484 |
| L8 | -0.998 | +0.209 | +0.999 | +0.999 | +0.999 |
| L12 | -0.999 | +0.209 | +0.999 | +0.999 | +0.999 |
| L20 | -0.991 | +0.110 | +0.994 | +0.993 | +0.993 |
| L23 | +0.540 | -0.416 | -0.249 | -0.317 | -0.786 |

The L8–L20 saturation is the massive-activations effect (Sun et al
2024): the anisotropy direction is so dominant that every raw
pair-difference axis is collinear with it (sign indicating which side
the centroid sits on). This means "VAL rides anisotropy" cannot be
cleanly tested in middle layers — *every* axis does, including ones
that exp123 verified were clean after stripping.

At L0 and L4 where the signal differentiates: VAL has the highest
|cos with anisotropy| (+0.537 at L4), with BALANCE close (+0.484).
Niamh's intuition that valence is substrate-anisotropy-rooted gets
partial support at L4 only.

### TEST 1 result: LAKOFF_UP shifts DIR → VAL across layers; MAG ~ null throughout

| Layer | LAK·MAG | LAK·DIR | LAK·VAL | winner |
|---|---|---|---|---|
| L0 | +0.040 | +0.307 | +0.140 | DIR |
| L4 | +0.031 | +0.411 | +0.275 | DIR |
| L8 | +0.004 | +0.375 | +0.316 | DIR |
| L12 | +0.028 | +0.370 | +0.257 | DIR |
| L16 | +0.010 | +0.281 | +0.251 | DIR |
| L18 | +0.010 | +0.233 | +0.240 | VAL |
| L20 | -0.013 | +0.117 | +0.215 | VAL |
| L22 | -0.008 | +0.006 | +0.185 | VAL |
| L23 | +0.107 | -0.038 | +0.052 | MAG |

**LAKOFF_UP is essentially never magnitude-aligned within-layer.**
LAK·MAG fluctuates around 0 (-0.013 to +0.107) at every layer. The
shift in LAKOFF_UP content across layers is DIR (early-mid, fading)
→ VAL (taking over by L18, peaking around L20, then fading by L22).

This **contradicts** the naive expectation that LAKOFF_UP rides
magnitude. Magnitude is a substrate-primitive for *content words
about extent* (Entry 5), not for the abstract schema axis itself.

The DIR→VAL shift was originally read as "embodied metacognition
emerges late in the network" — and that reading survives in the
weak sense that VAL alignment grows mid-network. The stronger DIR→MAG
prediction failed.

### TEST 2 result: BALANCE is a norm-deviation signal — *strong* confirmation

Niamh's prediction that BALANCE corresponds to LayerNorm. Tested by
two things:

(a) Do inflected forms have systematically different residual norms
than their bases? **Yes, massively.** Mean Δ‖r‖ at L12:

| Suffix | mean Δ‖r‖ | SE |
|---|---|---|
| -ER comparative | -595 | 136 |
| -EST superlative | -753 | 142 |
| -ING progressive | -403 | 144 |
| -ED past | -616 | 148 |
| un- negation | -770 | 128 |
| re- repetition | -849 | 121 |

Inflected forms have ~400–850 SMALLER residual norms than their
bases in middle layers. This is the massive-activations effect:
certain base tokens (like "happy", "do", "big") sit in the
high-norm regime; their inflected derivatives don't.

(b) Does per-pair Δ‖r‖ correlate with the pair's BALANCE projection?
**Yes, very highly.** Correlations across pairs within each suffix
family, at each layer:

| Layer | suffix family | r(Δ‖r‖, BALANCE proj) |
|---|---|---|
| L4 | ER, EST, ING, ED, UN | +0.86 to +0.97 |
| L8 | all | +0.81 to +0.97 |
| L12 | all (RE weaker) | +0.54 to +0.95 |
| L16 | all (RE weaker) | +0.48 to +0.96 |
| L20 | most (RE drops to +0.14) | +0.86 to +0.95 mostly |

Norm-deviation correlates near-perfectly with BALANCE projection.
Pythia's BALANCE schema in large part captures "this token has the
kind of activation magnitude LayerNorm has to renormalise hard."
The structural pattern (canonical-equilibrium vs deviation) is
preserved; the *content* is substrate-architectural.

### TEST 3 result: VAL doesn't behave as substrate-primitive by cross-layer stability

| Axis | Cross-layer stab (mean off-diag cos, L4-L22) |
|---|---|
| MAGNITUDE | +0.7742 |
| DIRECTIONAL | +0.7384 |
| VALENCE | +0.7547 |
| LAKOFF UP | +0.8602 |
| BALANCE | +0.7393 |

VAL sits between MAG and DIR. Not distinctively stable. LAKOFF is
the most stable, probably because it's averaged over the largest
anchor-word set.

### TEST 4 result: MAG contributes ~null to LAKOFF_UP; residual is enormous

Orthonormalised Gram-Schmidt decomposition of LAKOFF_UP onto
{MAG, DIR, VAL}. Three orderings tested (MAG-first, DIR-first,
VAL-first); coefficients are stable across orderings because the
basis is near-orthogonal (max axis-axis cos -0.27 at L8 for MAG-VAL).

At L12:
- |MAG coef| = 0.028 (null mean 0.023) — at null
- |DIR coef| = 0.369 (null mean 0.025) — 15× null
- |VAL coef| = 0.251 (null mean 0.025) — 10× null
- residual ‖·‖ = 0.894 — 80% of LAKOFF_UP variance unaccounted

The residual stays huge (0.87–0.99) across all layers. {MAG, DIR, VAL}
together explain a small slice of LAKOFF_UP. Most of the schema's
geometric structure is in *something else* — candidates include
STATUS-IS-UP, CONTROL-IS-UP, IMPORTANCE-IS-UP, VIRTUE-IS-UP (the
metaphorical-extension cluster Lakoff identified, minus the
specific axes we tested), and probably POS-tag effects from anchor
words.

### What exp141 settles

- BALANCE schema in Pythia is substantially a substrate-architectural
  artifact (norm-deviation signal). The most important finding here.
- VAL is not a substrate-primitive by the stability metric. Valence
  IS prominent in late-layer LAKOFF_UP content (TEST 1) but its
  ride on anisotropy can't be cleanly tested in middle layers
  because of massive activations.
- MAG is not within-layer aligned with LAKOFF_UP. The MORE-IS-UP
  claim has to live somewhere else if it lives at all (resolved by
  exp143).
- Most of LAKOFF_UP is unaccounted-for by our three primitive axes;
  the schema is richer than this basis captures.

---

## Entry 7 — exp142: FORWARD-BACK is NOT positional, LIGHT-DARK is NOT attentional

### Setup

Hypothesis (Niamh): if BALANCE rides LayerNorm, then maybe
FORWARD-BACK rides positional encoding (causal attention gives
transformers a hardcoded forward/backward asymmetry), and LIGHT-DARK
rides attention itself (KNOWING-IS-SEEING in Lakoff arises from
visual attention being the substrate-given epistemic primitive;
attention is the transformer's analogue).

### Method

Build a POSITION axis from natural prompts: mean(late-half residuals)
minus mean(early-half residuals), per layer, anisotropy + freq
stripped. Build an ATTENTION axis: mean(top-quartile-attention-received
residuals) minus mean(bottom-quartile-attention-received residuals),
per layer, similarly stripped. 10 natural prompts, 25–30 tokens each.

Test cos with all 8 Lakoff schemas (clean) per layer.

### Result for POSITION (FORWARD-BACK predicted)

Mean |cos| across L8–L20, ordered:

| Schema | mean |cos| with POSITION |
|---|---|
| BALANCE | +0.086 |
| FORCE | +0.074 |
| FORWARD-BACK | +0.071 ← predicted winner, actually 3rd |
| UP-DOWN | +0.068 |
| PATH-MOTION | +0.047 |
| DIFFICULTY-BURDEN | +0.043 |
| LIGHT-DARK | +0.021 |
| IN-OUT_CLEAN | +0.019 |

FB does NOT preferentially ride POSITION. All numbers are small
(< 0.1). No schema strongly rides the position axis.

### Result for ATTENTION (LIGHT-DARK predicted)

Mean |cos| across L8–L20:

| Schema | mean |cos| with ATTENTION |
|---|---|
| PATH-MOTION | +0.076 ← surprising, modest |
| FORWARD-BACK | +0.058 |
| UP-DOWN | +0.045 |
| FORCE | +0.043 |
| DIFFICULTY-BURDEN | +0.037 |
| BALANCE | +0.036 |
| IN-OUT_CLEAN | +0.030 |
| LIGHT-DARK | +0.026 ← predicted winner, actually 7th of 8 |

LIGHT-DARK does NOT preferentially ride ATTENTION. PATH-MOTION
modestly wins; mechanism plausible (attention IS a kind of
information-flow) but speculative and small.

The ATTENTION axis itself has weak cross-layer stability (+0.42 vs
+0.74+ for the others) and partial anisotropy correlation pre-strip
(+0.35 at L16). Crude measurement; summing attention across heads
and queries loses too much. Per-head analysis might salvage LIGHT-DARK
but the burden of proof is now on a more refined measurement.

### What this settles

The substrate-architectural-correspondence hypothesis is **specific
to BALANCE, not general**. LayerNorm is the only architectural
operation that runs on every token at every layer uniformly — it's
substrate-pervasive in a way attention and positional encoding aren't.
Other Lakoff schemas appear to be learned-semantic, riding distributional
co-occurrence structure rather than architectural features.

This *strengthens* the BALANCE-LayerNorm finding by making it specific.
It also tightens the substrate-embodiment story: most schemas live in
learned representation space, but one specific schema is anchored to
the substrate's architecture, and that anchoring is interpretable.

---

## Entry 8 — exp143: cross-layer metaphorical mappings (the resolution)

### The contradiction we were facing

- exp116 behavioural: clean-UP steering at L_inject produces magnitude
  effects in output (pool depth shifts up with patient height, etc.)
- exp141 cos test: cos(UP_clean[L], MAG_clean[L]) ≈ 0 at every layer

Both apparently true, but if UP and MAG are orthogonal within-layer,
how does UP-steering cause magnitude effects in output? Niamh
correctly identified this as a methodology problem and asked whether
we were measuring the wrong thing.

### Hypothesis

The cos-at-single-layer test measures within-layer co-activation /
shared linear structure. Lakoff's metaphorical mappings are
*cross-layer transformations* — UP-IS-MORE doesn't claim UP and
MORE are the same direction; it claims the source domain UP maps
onto the target domain MORE. In transformer geometry, a mapping is
a learned transformation executed by attention and MLPs across
layers, NOT direction-equality at a single layer.

Test: steer on clean-UP at L_inject, read residual at downstream
layers, compute projection on clean-MAG (and clean-VAL, clean-DIR)
at downstream layers. Compare to a random-direction-of-same-norm
steering control.

### Protocol

- L_inject ∈ {8, 12, 16}
- α ∈ {0, 1, 2, 4, 8}
- 10 natural completion-style prompts
- Steer at L_inject (last-token residual hook), read at all downstream
  layers
- ΔMAG = projection-under-steering minus projection-at-baseline,
  averaged across prompts
- Random direction: gaussian unit vector orthogonal to UP_clean at
  L_inject

### Result (L_inject = 12, α = 4, headline)

| L_readout | ΔMAG (UP) | ΔMAG (RAND) | ΔVAL (UP) | ΔVAL (RAND) | ΔDIR (UP) | ΔDIR (RAND) |
|---|---|---|---|---|---|---|
| L14 | +0.139 | +0.061 | +0.692 | -0.025 | +1.063 | +0.017 |
| L16 | +0.154 | +0.023 | +0.587 | -0.136 | +0.791 | +0.038 |
| L18 | +0.214 | +0.020 | +0.553 | -0.248 | +0.490 | +0.154 |
| L23 | +0.455 | -0.134 | +0.455 | -0.480 | -0.276 | +0.935 |

UP-steering produces:
- Modest but consistent ΔMAG at every downstream layer (UP-vs-RAND
  gap +0.08 to +0.59 depending on layer)
- LARGE ΔVAL throughout, growing with α, with RAND going negative
  (UP-vs-RAND gap +0.70 to +0.94 at α=4)
- LARGE ΔDIR early after injection, fading to negative by L23
  (within-layer alignment carried forward by residual stream, then
  transformed)

Dose-response approximately linear in α from α=1 to α=8 for all axes.
Effect persists across all three L_inject values.

### What this settles

1. **The contradiction dissolves.** Within-layer orthogonality and
   cross-layer mapping are different geometric facts; both real.
2. **UP→VAL is the strongest learned mapping by far** — roughly 4-5×
   stronger than UP→MAG. The model's most robust metaphor on UP is
   the valence one (HAPPY-IS-UP, GOOD-IS-UP), not the magnitude one.
3. **UP→MAG exists but is moderate.** The MORE-IS-UP mapping is
   present as a cross-layer transformation; exp116's magnitude
   effects under UP-steering are explained.
4. **Methodology lesson**: testing Lakoffian metaphor structure
   requires measuring cross-layer transformations, not just
   within-layer axis alignment. The static cluster structure (within-
   layer) and the dynamic mapping structure (cross-layer) are
   complementary, both Lakoffian, both real.

### Theoretical refinement

The model has the *structural form* of Lakoffian metaphor execution
— cross-domain transformations implementable as multi-layer
computations — without any embodied source for the mappings.
HAPPY-IS-UP is the strongest cross-layer mapping (~5× MORE-IS-UP)
and is also the most phenomenologically embodied in Lakoff's
framework. Its presence in Pythia without a phenomenological
substrate is suggestive evidence that the structural form arises
in non-embodied systems through statistical learning from text.

### Follow-up: exp145 result — both interpretations partially hold

Tested by steering on three non-Lakoffian directions at L_inject=12,
α=4, reading ΔMAG/ΔVAL/ΔDIR downstream:

| Direction | sanity (cos with VAL at L12) | ΔVAL at L16 |
|---|---|---|
| UP (Lakoffian reference) | n/a | +0.591 |
| NOUN_VERB (syntactic, clean) | +0.037 (orthogonal) | +0.284 |
| CONC_ABS (semantic, partly VAL-loaded) | +0.180 | +0.237 |
| ANIM_INAN (semantic, clean) | +0.052 (orthogonal) | -0.081 |
| RAND | 0 (constructed orthogonal) | +0.089 |

**The result is genuinely mixed**:

- UP produces the **largest** ΔVAL by a clear margin (~2× the next
  meaningful direction). Specific UP→VAL learned mapping is real.
- NOUN_VERB (cleanly orthogonal to VAL at L12) produces ΔVAL of
  +0.284 at L16 — ~3× the RAND control's +0.089. Meaningful
  steering inputs DO get pulled toward VAL by late layers above
  noise levels. General late-VAL-routing is also real.
- CONC_ABS isn't a fully clean test (had +0.180 baseline VAL
  content at L12).
- ANIM_INAN going NEGATIVE is unexpected and unexplained.

**So both interpretations partially hold**. The model has:
1. **A structural late-layer pull toward valence** (~+0.20 to
   +0.30 ΔVAL for orthogonal meaningful directions above RAND
   baseline) — substrate-level / architecture-plus-objective finding.
2. **A specific learned UP→VAL mapping on top** (~+0.30 *additional*
   ΔVAL beyond the structural pull) — classical Lakoffian
   metaphor as cross-layer transformation.

The total observed UP→VAL effect decomposes roughly as:
specific-learned-mapping + general-late-VAL-routing. Both contribute.

### Unexpected findings from exp145

- **CONC_ABS routes hard to DIR in late layers** (ΔDIR +0.896 at
  L20, +0.734 at L23 under α=4). LARGER than UP's late-layer ΔDIR.
  The model has an apparent concrete-to-directional cross-layer
  transformation we didn't predict. Speculative interpretation:
  concrete referents have spatial location encoded in residual
  stream; abstract referents don't. Worth following up.
- **ΔMAG is much more specific to UP than ΔVAL is.** UP produces
  positive ΔMAG; all three other meaningful directions produce
  *negative* ΔMAG at most layers. So UP→MAG is highly selective
  in a way UP→VAL isn't. The MORE-IS-UP transformation appears
  more clearly a specific learned mapping than HAPPY-IS-UP, which
  is more confounded with general late-VAL-routing.

### What exp145 changes for the substrate-embodiment story

The structural late-layer pull toward valence (+0.20-0.30 ΔVAL
even for orthogonal directions) is a substrate-level finding about
Pythia's geometry. It's consistent with: the architecture (residual
stream, layer norm, attention, output unembedding) plus the
objective (next-token prediction on natural language) produces a
late-layer geometry that organises any meaningful content around
affective dimensions. This is a structural feature of being a
language model, not a specifically learned metaphor.

This is interesting for the embodiment thesis because it suggests
a substrate-determined organising primitive at the LATE-layer end
(valence routing) in addition to the substrate-architectural
correspondence at the BALANCE schema (LayerNorm). The substrate
shapes the model's geometry in multiple specific ways:
- LayerNorm produces equilibrium/deviation structure that ends up
  encoded as BALANCE schema
- Affective output-token distribution may produce late-layer
  valence routing that pulls meaningful content toward valence axes

Both findings are illustrative of how transformer substrate features
shape representational organisation, even when the content being
organised has no obvious phenomenological-embodied source.

The selective UP→VAL mapping (above and beyond the general routing)
is the classical Lakoff metaphor finding, which presumably arises
from text statistics encoding HAPPY-IS-UP and the model learning
that mapping. Both stories coexist.

### Caveat: text-derived vs substrate-native

The UP→VAL and UP→MAG transformations could arise from text
statistics (human writers encode these metaphors in language;
Pythia learned them from text) rather than from substrate
primitives. Attributing them to substrate would require showing
the mapping strengths don't track textual co-occurrence frequencies
— a much harder test we haven't done. The cross-layer-mappings
finding is real; the substrate-native interpretation of the
mappings is not established.

---

## Entry 9 — Where things stand methodologically

### What works

- **Per-layer freq + anisotropy stripping** (exp138 lesson): both
  axes must be projected out at every layer, not at one canonical
  layer. Pair-difference morphology directions especially require
  per-layer anisotropy strip.
- **Random-direction steering controls** (exp143): cleanly separates
  axis-specific cross-layer effects from generic perturbation
  effects.
- **Substrate comparison Pythia vs GloVe** (exp140): isolates
  substrate-specific encoding signatures from universal Lakoff
  structure.
- **Architectural-correspondence test** (exp141 TEST 2): per-pair
  correlation between norm-deviation and schema projection is a
  clean way to test "this schema rides this architectural feature."

### What needs care

- **Massive activations in middle layers** (Sun et al, observed in
  exp141 TEST 0): at L8-L20, raw bare-word pair-difference axes
  saturate to ±0.998 with anisotropy. Raw-axis tests in this regime
  are uninformative. Cleaned axes work; raw ones don't.
- **Cross-layer matrix preservation tests** (exp139 caveat): pass
  for generic word-diff matrices too. Use only as relative comparison
  between clusters, not as standalone test of system-ness.
- **Predictions partly informed by prior work** (exp102 GloVe, the
  conversation-and-iteration with Niamh): not fully blind. Disclose
  this in any paper writeup.
- **Distinguishing Lakoff's direct predictions from our extrapolations**:
  Lakoff has image-schema-as-relational-system and canonical
  metaphorical mappings (UP-IS-MORE, GOOD-IS-UP, TIME-IS-MOTION-FORWARD).
  We extrapolated to: morphology positions on schemas at conceptual
  locations (exp138); cross-layer mappings as the model's
  implementation of Lakoffian metaphor (exp143); architectural
  schemas (exp141). These are *consistent with* and *licensed by* a
  Lakoffian framework but are our contributions, not his direct
  claims. Be explicit in writeups.

### Cross-model replication completed: exp144 result

Substrate findings tested on Pythia 70M, 160M, 410M, 1.4B (20× scale
range, 4 model sizes).

**TEST A (extent words on MAG vs DIR at middle layer)**:
- Pythia 70M: ratio mean|MAG|/mean|DIR| = 1.71 (4/7 MAG wins)
- Pythia 160M: ratio 2.78 (5/7 MAG wins)
- Pythia 410M: ratio 3.12 (7/7 MAG wins)
- Pythia 1.4B: ratio 2.62 (5/7 MAG wins)

**4/4 sizes have ratio > 1.3**. Substrate-content-encoding signature
holds across the size range. Peak at 410M, slightly weaker at extremes
but always present.

**TEST B (BALANCE-as-norm-deviation correlation)**:
Mean r across all 19 layer × size combinations: **+0.909**, range
+0.725 to +0.961. **19/19 above 0.7**.

| Model | layer means |
|---|---|
| Pythia 70M | L1:+0.90, L2:+0.94, L3:+0.96, L4:+0.94 |
| Pythia 160M | L2:+0.91, L4:+0.95, L6:+0.91, L8:+0.93, L10:+0.81 |
| Pythia 410M | L4:+0.91, L8:+0.94, L12:+0.88, L16:+0.87, L22:+0.72 |
| Pythia 1.4B | L4:+0.96, L8:+0.95, L12:+0.95, L16:+0.93, L22:+0.89 |

The architectural correspondence is remarkably robust. Pythia 1.4B
shows the cleanest correlation pattern; 410M is slightly weaker;
smaller models maintain the correspondence in early-mid layers.

**What this establishes**:
- The substrate findings are NOT Pythia-410M-specific quirks.
- The substrate-native thesis is now backed by multi-model evidence
  across a 20× scale range.
- The substrate-content-encoding signature (extent-on-MAG) is a
  scale-invariant feature of Pythia's organisation, not an
  emergent property of one specific model size.
- The substrate-architectural correspondence (BALANCE-LayerNorm) is
  near-universal across the tested models (mean r = 0.91, every
  layer × size combination above 0.7).

**Remaining open**: cross-*architecture* replication. RMSNorm models
(Llama family) would test whether BALANCE rides specifically LayerNorm
or any normalisation-equilibrium operation. Cheap to do (exp146
candidate, separate from spatial-modelling follow-up).

### exp146 + exp147: the CONC_ABS → DIR finding investigated, deflated

**exp146 design**: test exp145's CONC_ABS → DIR routing across
multiple spatial axes (VERTICAL, HORIZONTAL, CONTAINMENT, PROXIMITY)
plus TEMPORAL control, plus HARD_SOFT and RATIONAL_EMOTIONAL as
alternate steering directions to disambiguate spatial-modelling
(H1), physicality (H2), and general-categorial-routing (H4)
hypotheses.

**exp146 result, L=20, α=4**:
- CONC_ABS → VERTICAL: +0.919 (large, replicating exp145)
- CONC_ABS → other spatial axes: small or negative (H1 fails)
- HARD_SOFT → VERTICAL: -0.318 (negative, H2 fails)
- RATIONAL_EMOTIONAL → VERTICAL: -0.146 (H4 fails)
- **RAND → VERTICAL: +0.530 (with seed 42)** — surprise

The seed-42 RAND control producing +0.530 ΔVERTICAL prompted the
worry that maybe late layers route random perturbations to VERTICAL
generally (substrate-level attractor for VERTICAL, with possible
self-referential reading: model encoding its own computational
depth as verticality via learned HIGHER-PROCESSING-IS-UP metaphor).

**exp147 design**: 20 random seeds, same steering protocol, same α,
read multiple readout axes at downstream layers. If VERTICAL is a
robust attractor, mean across seeds at L20 should be > 0.3 with low
variance.

**exp147 result, mean ± std across 20 seeds at L=20**:
- VERTICAL: +0.103 ± 0.354 (range -0.617 to +0.562, 12/20 positive)
- VALENCE: -0.027 ± 0.194
- TEMPORAL: +0.040 ± 0.125
- HORIZONTAL, CONTAINMENT, PROXIMITY: all small means, modest std
- MAGNITUDE: +0.030 ± 0.115

**The late-VERTICAL-attractor hypothesis is NOT cleanly supported.**
Mean is small, variance is huge. The seed-42 result of +0.530 was
near the top of the natural distribution range but not anomalous.
The model does NOT systematically route random perturbations to
VERTICAL.

**However: VERTICAL has distinctively higher variance than other
axes at L20** (0.354 vs 0.09-0.19). Something about late-layer
geometry makes VERTICAL the most variance-rich axis under random
perturbation. Some random directions get amplified strongly into
VERTICAL (or its opposite); others don't. This is partial structure,
not a clean attractor.

**The self-aware-metaphor speculation from this conversation does
NOT get clean empirical support.** If late layers were systematically
encoding their own depth as VERTICAL through reflexive HIGHER-IS-UP
metaphor application, we'd expect high mean with low variance. We
see the opposite. Walk back this speculation.

**What survives**: CONC_ABS at L20 produces ΔVERTICAL = +0.919,
which exceeds the maximum of all 20 random seeds (+0.562). So
CONC_ABS *specifically* routes to VERTICAL above what any random
direction can produce. This is a real CONC_ABS → VERTICAL learned
cross-layer mapping, similar in structure to UP→VAL (exp143). Just
not a substrate-level attractor.

### Revised conclusions on cross-layer mappings

Combining exp143, exp145, exp146, exp147:

**The model has specific learned cross-layer transformations** that
implement Lakoffian-style metaphorical mappings as distributed
computations across MLPs and attention heads:
- UP → VAL (strong, ~+0.59 at L16 specific component)
- UP → MAG (moderate, ~+0.13 specific component)
- CONC_ABS → VERTICAL (large, ~+0.40 specific component above max
  random seed)

**The "general late-layer attractor" framing is partially supported
for VALENCE** (exp145 showed NOUN_VERB and CONC_ABS produce VAL
shifts above RAND, but with caveat that exp147 showed RAND VAL
variance is also moderate at +0.194).

**The "general late-layer attractor for VERTICAL" is NOT supported**
(exp147 shows mean +0.10 ± 0.35 across 20 seeds; no consistent
direction).

**The "self-aware reflexive metaphor application" hypothesis is
speculative and currently not empirically supported.** Would require
much stronger evidence — possibly: explicit test of whether the
model's outputs reflect HIGHER-PROCESSING-IS-UP metaphor when
producing text about its own cognition.

### Next steps — clean ordered list for next session

Paper story (per Niamh, 2026-06-08): three empirical claims plus
punchline, plus speculative Part 2. Critical-path experiments below
are what's needed to make that story land.

**CRITICAL PATH (paper doesn't land without this)**:

1. **exp150 — KEYSTONE: morphology-on-Lakoff in word2vec/GloVe.**
   Replicate exp138 protocol on word2vec and GloVe-300d. Build
   schema directions and suffix pair-difference directions from
   static embeddings; compute suffix × schema cosine matrix; compare
   to Pythia 410M result (exp138).
   - If word2vec shows same morphology-schema mappings → the
     "transformers reconstruct embodied structure" claim collapses.
     Paper restructures around within-transformer findings only.
   - If word2vec doesn't show them → load-bearing Claim 3 lands.
   The single most important remaining experiment.

**STRENGTHENING (story is weaker without these but doesn't collapse)**:

2. **exp148 — schema steering for FB and LD.** Replicate exp143
   protocol with FORWARD-BACK and LIGHT-DARK as steering sources.
   Predicted target axes:
   - FB-steering → expect ΔTIME (past/future) at downstream layers
   - LD-steering → expect ΔKNOWING (known/unknown) or ΔCERTAINTY
   Builds target axes from appropriate word lists. Tests "schemas
   plural function as steering primitives" not just UP.

3. **exp149 — schema relational structure replication.** Run exp123
   on Pythia 70M, 160M, 1.4B. Each run is small (single model,
   single test). Checks cross-model stability of Claim 2.

4. **exp151 — exp138 on Pythia 1.4B.** Replicate the morphology
   heatmap with anisotropy strip on a larger model. Cross-model
   stability of Claim 3 within transformer family.

**OPTIONAL / EXTENDING (Part 2 and future work)**:

5. **exp152 — BALANCE-LayerNorm on Llama-3.2-1B (RMSNorm).** Tests
   whether BALANCE-norm correspondence is LayerNorm-specific or
   generalises to any normalisation operation. Strengthens Part 2.

6. Per-head decomposition of UP→VAL transformation. Mech-interp
   localisation. Strengthens Claim 1 mechanistically.

7. Behavioural test of UP-steering producing valence shifts in
   generated text (not just activation projections). Connects
   mechanistic finding to behavioural outputs.

### Priority for next session

**Start with exp150**. Everything else is contingent on this.

Pseudocode-level design for exp150:
```python
# Load GloVe-Wiki-Gigaword-300 (already used in exp140)
# Build Lakoff schema axes from canonical vocabulary:
#   schema_dir = mean(pos_words) - mean(neg_words), normalised
# Build suffix pair-difference directions:
#   suffix_dir = mean(inflected - base), normalised
# Important: no anisotropy in static embeddings the same way, but
# subtract overall mean as analogue
# Compute suffix × schema cosine matrix
# Compare to Pythia 410M exp138 result
# Look specifically for:
#   -ING → PATH-MOTION positive?
#   -ED → FORWARD-BACK negative?
#   -UN → LIGHT-DARK negative?
#   Shared BALANCE-negative across inflected suffixes?
# If pattern is similar → Lakoff structure is in distributional
#   statistics, transformer-reconstruction claim doesn't survive
# If pattern is absent → Lakoff structure is transformer-specific
```

word2vec/GloVe morphology is documented in the literature
(superlatives have systematic offsets, plurals have specific
directions). The novelty here is the schema-relational positioning,
which to our knowledge no one has tested. Predicted result: weaker
or absent in static embeddings vs transformer (otherwise it'd have
been observed already), but this is a real empirical question.

---

## Closing — what we have, what we don't have, and the deeper open question

### What v5 documents, compressed

Three empirically distinct geometric forms of Lakoff-like organisation
in Pythia 410M, each captured by a different methodology:

1. **Within-layer cluster structure** (exp123 + exp138). Schemas as a
   coordinated relational system at every layer; grammatical morphology
   positioned on schemas at conceptually-plausible locations (our
   extrapolation, well-supported by exp138 specifics).
2. **Cross-layer metaphorical transformations** (exp143). UP-steering
   propagates to VAL/MAG/DIR projections at downstream layers above a
   random-direction null. UP→VAL is large (~+0.72 gap at α=4); UP→MAG
   is real but moderate (~+0.13 gap).
3. **One substrate-architectural correspondence** (exp141 BALANCE =
   norm-deviation). Per-suffix r=0.85–0.97 correlation between
   ‖residual‖ deviation and BALANCE projection. FB-position (exp142)
   and LD-attention (exp142) tested as analogous candidates; both
   null. The architectural correspondence is *specific to BALANCE*,
   not general.

Plus one supporting finding:

- **Substrate-specific content-word placement** (exp140). Extent
  content words ride MAGNITUDE in Pythia, DIRECTION in GloVe.
  Substrate-specific signature in content encoding, NOT in the
  abstract schema axis.

### What v5 does NOT establish (honest scope)

Several claims that natural reading of the findings might suggest, but
which the data don't actually support:

- **LAKOFF_UP is magnitude-aligned.** It isn't. cos(LAKOFF_UP, MAG) ≈ 0
  at every layer (exp141 TEST 1). Magnitude is a substrate-primitive
  for *content words about extent* (exp140), not for the abstract
  schema axis. These are two separable findings about two different
  things.
- **The substrate-architectural correspondence is a general pattern.**
  It isn't. We have 1 hit (BALANCE) and 2 misses (FB, LD). With this
  ratio it's more accurate to say "one Lakoff schema is anchored to
  a specific architectural feature; most are learned-semantic."
- **The substrate-native thesis is empirically established.** Not
  with this evidence. We have two data points: extent-on-MAG (exp140,
  one word family, two substrates, one model size) and BALANCE-norm
  (exp141, one schema, one architecture, one model size). Both
  consistent with the substrate-native thesis; neither sufficient to
  establish it as a general claim. The thesis remains suggestive
  rather than supported.
- **MORE-IS-UP arises from substrate primitives.** This is unclear.
  exp143 shows a UP→MAG cross-layer mapping exists in Pythia
  (modest), but the mapping could arise from text statistics (human
  text encodes MORE-IS-UP metaphors that Pythia learned) rather than
  from substrate primitives. To attribute the mapping to substrate
  rather than text, we'd need to show the mapping doesn't track
  textual distribution — a much harder test than we've done.
- **Cross-layer mappings are specifically learned UP→VAL/MAG/DIR
  transformations** (vs late-layer-VAL-attraction as a global
  property). The exp143 random-direction control rules out generic
  noise propagation, but doesn't rule out the alternative
  "any meaningful steering input gets routed to VAL by late layers."
  A stronger control would steer on a *different meaningful direction*
  (noun-vs-verb, concrete-vs-abstract, an arbitrary content axis)
  and check whether THAT also produces ΔVAL downstream. If yes,
  late-VAL-routing is global. If no, UP→VAL is specifically learned.
  We haven't done this control. The "selective learned mapping"
  interpretation is the more interesting one, but it isn't established.

### What would strengthen each claim

For the substrate-native thesis specifically (Niamh's primary interest):

- **Cross-model-size replication**. Same exp140 + exp141 TEST 2 on
  Pythia 70M, 160M, 1.4B, 2.8B. If extent-on-MAG and BALANCE-norm
  hold across scales, the thesis gets multi-model evidence. If only
  410M shows it, it's a model-specific quirk.
- **Cross-architecture replication**. Same protocols on a non-LayerNorm
  model (Llama uses RMSNorm). If BALANCE-norm correlation switches
  or vanishes, that's strong evidence the correspondence depends on
  the specific architectural feature (not "any normalisation
  produces this"). Stronger architectural-correspondence claim.
- **More candidate substrate-architectural correspondences**. We
  tested LayerNorm (positive), positional encoding (null), attention
  (null). What about attention sinks? Residual-stream sparsity?
  Embedding-unembedding weight tying? More tests = better picture of
  whether substrate-architectural correspondence is "rare specific
  case" or "common pattern."
- **More content-encoding signatures**. Extent on MAG is one
  family. Intensity (intense/mild)? Temperature? Brightness?
  Importance? Are there content families that load on substrate-
  intrinsic axes in Pythia and on something else in GloVe? Single
  case vs general pattern matters for the paper.

For the cross-layer mappings claim specifically:

- **Meaningful-but-unrelated control direction**. Steer on a content
  axis with no Lakoffian content (noun-vs-verb, animate-vs-inanimate)
  and read MAG/VAL/DIR projections downstream. If random-direction
  produces flat effects but meaningful-non-UP produces big ΔVAL,
  then late-VAL-routing is happening for any meaningful content
  and UP→VAL is non-specific. If meaningful-non-UP produces flat
  effects too, UP→VAL is selective.
- **Other schemas as steering sources**. Steer on FORWARD-BACK,
  LIGHT-DARK, FORCE. Read projection changes on appropriate target
  axes (TIME for FB, KNOWING for LD, EFFORT for FORCE). Does each
  schema produce its own characteristic cross-layer mapping
  pattern? This generalises exp143 from "UP has cross-layer mappings"
  to "schemas implement metaphorical mappings as cross-layer
  transformations."
- **Asymmetry test**. Lakoffian metaphor is source-target asymmetric.
  Under VAL-steering, does UP-content increase downstream? If no
  (preserved asymmetry), the mapping is one-way and Lakoff's
  source-target structure holds in transformer geometry. If yes,
  the mapping is bidirectional and the cross-layer transformation
  picture is more like "linked feature pairs" than "metaphorical
  mapping."
- **Per-head decomposition**. Which attention heads and MLPs execute
  the UP→VAL transformation? Localising mechanistically converts the
  claim from "model has UP→VAL mapping" (correlational) to "these
  specific computational units implement the UP→VAL transformation"
  (mechanistic). Required for a strong mech interp story.

### The deeper open question (riff)

The cross-layer-mappings finding (exp143) suggests that **metaphor in
Pythia lives as functions, not facts**. There's no place in the model
where "UP means MORE" is stored. There's a process — distributed
across MLPs and attention heads between L_inject and L_readout —
that takes UP-direction input and produces some MAG-direction output.
This is more faithful to Lakoff's claim that metaphor is generative
(you use the source to think the target) than direction-equality
storage would have been.

If this is right, the layer-trajectory finding (DIR→VAL shift across
layers in exp141 TEST 1) might be the *cumulative* result of cross-
layer mappings running across depth. The model's static within-layer
schema content composition at L_k reflects the cumulative effect of
all the mappings that have executed up to layer L_k. Early-layer
LAKOFF_UP is DIR-rich because no UP→VAL mapping has run yet.
Late-layer LAKOFF_UP is VAL-rich because the mapping has accumulated
through depth. The static and dynamic views might be two angles on
the same underlying process.

This is testable in principle: derive the predicted layer-trajectory
of LAKOFF_UP composition from the measured cross-layer mapping
strengths, and compare to the observed trajectory. If they match,
the unifying picture holds. If they don't, static and dynamic
geometries are partially independent.

The broader methodological point: a lot of mech interp work uses
linear probes and direction-finding (static, within-layer methods)
that capture cluster structure but systematically miss cross-layer
transformations. Schema-level findings about *organisation* show up
clearly in static methods. Metaphor-level findings about
*transformations* require dynamic steering-and-readout methods.
Lakoff's substantive claims are mostly transformations (X organises
Y, X is metaphorically extended into Y), so dynamic methodology is
the more natural match for testing them. We have been doing static
methodology because it's cheap and gives numbers, but it may be
underestimating what's actually there.

### For the paper writeup

The three geometries are real and well-documented. The substrate-
native thesis as a *general claim* is suggestive but undertested.
Three honest options for the paper:

1. **Narrow defensible paper**: document the three geometries on
   Pythia 410M, explicitly scope the substrate claims to "we
   illustrate that transformer substrates can give rise to schema-
   anchoring primitives, illustrated by BALANCE-LayerNorm and
   extent-on-MAG; whether this generalises is an open question."
   Publishable, honest, modest.
2. **Stronger paper with replication**: do the cross-model + cross-
   architecture replications first (Pythia 70M-2.8B, Llama-RMSNorm),
   then claim substrate-native thesis with multi-model evidence.
   Higher cost, higher claim-strength.
3. **Mech-interp methodology paper**: lead with the cross-layer
   mappings methodology, frame the three geometries as a case study
   of how dynamic methods complement static ones, deprioritise
   substrate claims. Reframe Lakoff as analytical scaffolding,
   substrate-embodiment as suggestive open question rather than
   established claim.

Each is reasonable. Niamh's primary interest (model embodiment)
favours #2; cost favours #1; methodological contribution favours #3.

The deeper open question for further work is whether cross-layer
mappings turn out to be the right unit of analysis for Lakoffian
structure in transformers more generally. exp143 is one example;
the methodology applied to other schemas (and other transformers)
would tell us whether metaphor-as-function is a universal feature
of how distributed neural systems implement cross-domain mappings,
or a specific feature of Pythia 410M's particular learned structure.

---

# ADDENDUM 2026-06-10 — exp150 keystone (static-substrate comparison) + exp153 (emergence-in-depth probe)

Session: 2026-06-10. The keystone experiment finally ran. Outcome: neither
of the two pre-registered clean outcomes — something more structured, then
confirmed across three static substrates and traced into early depth.

## exp150 — morphology-on-Lakoff in GloVe (KEYSTONE)
`exp150_glove_morphology_keystone.py`, `exp150_output.txt`,
`exp150_glove_vs_pythia.png`

Replicated exp138 protocol in glove-wiki-gigaword-300: same SUFFIX_PAIRS,
same 8 MML schemas, suffix directions as mean unit-diff, strip analogue =
global-mean direction + COMMON−RARE freq axis (Gram-Schmidt, mirroring
strip_aniso_freq). Pythia comparison matrices parsed from
results_exp138.txt (no transcription). Pre-registration in the script
docstring, written before running, including a committed point prediction.

**Result (stripped matrices):**
- Full-matrix correlation GloVe↔Pythia is significant at every layer
  (r = +0.43..+0.49 vs row+col permutation null, p ≤ 0.002), and survives
  BALANCE-column exclusion.
- BUT the exp138 signatures do not reproduce: BALANCE universal sink FALSE
  (GloVe ER×BALANCE = −0.01 vs Pythia −0.38); LIGHT-DARK null FALSE
  (GloVe UN loads LD −0.39). UN-largest-row TRUE (GloVe UN row norm 0.89,
  even bigger than Pythia's 0.62).

## exp150b/c — post-hoc row split + cross-substrate replication
`exp150b_posthoc_output.txt`, `exp150c_crosssubstrate_output.txt`,
`exp150_w2v_output.txt`, `exp150_fasttext_output.txt`

Post-hoc control (declared as such): split rows into derivational prefixes
(UN, RE — carry lexical semantics) vs inflectional suffixes (ER, EST, ING,
ED, S — grammatical). Then replicated on word2vec-google-news-300 (clean
second substrate: whole-word, different corpus, different objective) and
fasttext-wiki-news-subwords-300 (supplementary ONLY — subword n-grams make
suffix directions coherent by construction; not a clean distributional
control).

**Result, consistent across all three static substrates:**

| | GloVe | word2vec | fastText | Pythia 410M |
|---|---|---|---|---|
| ER×BALANCE | −0.012 | −0.005 | +0.012 | **−0.377** (every layer) |
| UN+RE rows vs Pythia, r | 0.63–0.71 SIG | 0.56–0.61 SIG | 0.50–0.52 SIG | — |
| infl rows vs Pythia, r | 0.20–0.25 ~ns | 0.30–0.36 ~ns | 0.28–0.35 | — |

**The key refinement** (kills the "inflectional directions are just noisy
in static space" deflation): the static spaces agree WITH EACH OTHER on
inflectional geometry far more than any agrees with Pythia —
static↔static inflectional r = +0.71..+0.77; static↔Pythia = +0.20..+0.36.
Static spaces share a consistent inflectional geometry; it is a DIFFERENT
geometry from Pythia's. The transformer does not amplify a weak
distributional signal — it REORGANISES inflectional morphology onto the
schema system. The BALANCE markedness sink exists in no static space
tested.

UN/RE double as an internal positive control: the method finds affix→schema
positioning in static space when the affix carries semantic content
(negation, repetition). The inflectional absence is real, not method
failure. Caveat for the paper: in all three static spaces UN loads on
exactly the valence-polar axes (UP, LD, BALANCE) — the "UN positioning is
distributional" half may reduce to a single valence axis. Control needed
before the paper says anything specific about UN (BURROWS.md).

## exp153 — where in depth does the inflectional sink emerge?
`exp153_embedding_layer_emergence.py`, `exp153_output.txt`,
`exp153_emergence.png`

exp138 protocol at hook_embed, L0, L1, L2 (+ L4/L8/L12 anchors; parity
check vs results_exp138.txt exact, max |Δ| = 0.000). Two variants:
last-token (exp138 parity) and mean-pooled over word tokens
(tokenization control; NB Pythia's TL config does NOT prepend BOS —
to_tokens("big") is 1 token; an earlier buggy mean variant NaN'd on this).
Pre-registered outcomes E1/E2/E3 in script docstring.

**Result (last-token / mean-pooled broadly agree):**

| probe | ER×BAL | infl r vs GloVe | infl r vs Pythia L12 |
|---|---|---|---|
| embed | −0.13 | +0.32 | +0.60 |
| L0–L2 | −0.19..−0.23 | ~+0.30 | +0.78..+0.87 |
| L4 | **−0.47** | +0.23 | +0.92 |
| L8–L12 | −0.38 | +0.18..+0.20 | +0.98..1.00 |

Verdict: between E1 and E2. Pythia's own embedding matrix already leans
toward the schema geometry (infl r vs L12 = +0.60 at embed, vs GloVe's
+0.20 from outside the model) — transformer TRAINING writes some of the
reorganisation into the static embedding itself. But the sink at embed is
only ~1/3 of plateau strength; the bulk is COMPUTED between L2 and L4
(−0.23 → −0.47), with a small overshoot at L4 relaxing to the −0.38
plateau. So: partially trained into the embedding, majorly computed in
early depth. Both senses of "reconstructed" hold, and the embed-vs-GloVe
contrast is itself clean evidence (same kind of object — a static word
embedding — different training regime).

Calibration notes (kept honest):
- exp150 committed prediction was WRONG in its specifics: predicted the
  BALANCE sink would reproduce in GloVe (as distributional markedness) and
  suffix-specific mappings wouldn't. Inverted: the sink-for-inflection is
  the transformer-specific part; the semantic prefix content is the
  distributional part.
- exp153 prediction partially right (sink builds in depth, plateau by L4)
  and partially wrong (embedding layer is not tokenizer-junk-dominated;
  it's already weakly Pythia-like; mean-pooled variant agrees).
- The exp150b/c row split is post-hoc. It now has three-substrate
  consistency, but the paper should present it as a discovered structure
  with confirmatory replications, not as pre-registered.
- exp153's embed-layer r vs Pythia L12 could be partially inflated by
  shared-model artifacts (same tokenizer, suffix-token identity at last
  position). The mean-pooled variant agreeing (+0.71) limits but does not
  eliminate this.

## Claim 3, third draft (for the paper)

> Distributional statistics position SEMANTIC affixes (un-, re-) on schema
> axes, and induce a consistent inflectional geometry shared across static
> models (GloVe, word2vec, fastText agree with each other, r ≈ 0.7) — but
> the transformer REPLACES that inflectional geometry with a different
> one, anchored on the schema system (BALANCE markedness sink, ER×BAL
> −0.38 vs ~0.00 in all static spaces), partially written into its own
> static embedding by training and majorly computed between L2 and L4.

Sharper than the original Claim 3 and arguably a better punchline: the
part the transformer adds is precisely the purely-grammatical,
low-semantic-content morphology. (Noted with suspicion: this fits the
interlayer-grammar framing from the 2026-06-09 consolidation document
almost too well. The valence control on UN and exp151 on Pythia 1.4B are
the remaining legs before leaning on it.)

Remaining before paper weight: exp151 (1.4B replication of exp138 — is the
Pythia side a 410M quirk?); UN-valence control; content-word frequency
control on the sink (BURROWS.md); L2→L4 component localisation (which
attention heads / MLPs write the sink — new burrow).

## exp154/154b — norm-confound control: the sink collapses, and something better stands up
`exp154_norm_confound_control.py`, `exp154_output.txt`,
`exp154_norm_confound.png`, `exp154b_glove_norm_coupling.txt`

Motivation: scariest hole in the grounding chain — maybe the BALANCE
schema axis is just a readout of residual-norm structure and the
suffix×BALANCE sink re-measures norm displacement. (exp138 unit-normalises
word vectors, so per-word norm magnitude was already removed; the live
confound was the norm-ENCODING DIRECTION d_norm: direction whose
projection predicts ||r||.)

Method: per layer, build d_norm (covariance direction of z-scored norms
with unit residuals, aniso+freq stripped; carrier sanity corr = 0.83-0.94)
and d_disp (same for |z|, absolute displacement). Project both out of
schema and suffix directions; recompute sink. Control: 20× stripping two
random stripped directions.

**Result: N2, full collapse.** Pre-registered prediction (partial
retention ~50%) WRONG — third miss in the same direction this session;
the effects keep being cleaner than my hedged middle predictions.
- Inflectional mean sink: −0.41/−0.32/−0.29/−0.29/−0.28 (L4-20)
  → −0.06..+0.06 after the 2-direction strip (≈0% retained).
- Random-2-direction strip: moves the sink by ±0.001. Surgical specificity.
- cos(BALANCE, d_norm) = +0.70..+0.78 at every layer.
- Δnorm (inflected − base): uniformly negative, every suffix, every layer.
  Inflected forms sit at systematically LOWER residual norms.

**Consequence: exp138's sink and exp141's BALANCE-norm correspondence are
ONE finding, not two.** "Morphology loads on the BALANCE schema" was the
lexical shadow of "morphology is norm-displaced and BALANCE reads out
norm displacement."

**exp154b — the discriminating check in GloVe** (same word set, same
protocol): GloVe HAS a findable norm-encoding direction (carrier 0.61)
and HAS a BALANCE vocabulary axis — and they are decoupled:
cos(BALANCE_glove, d_norm_glove) = −0.07 (vs Pythia +0.70..+0.78). NO
GloVe schema axis couples to its norm direction (max |cos| = 0.19,
PATH-MOTION). And inflected forms are NOT consistently norm-displaced in
GloVe (Δnorm mixed-sign, ±3-5%).

## The reframed finding (replaces "BALANCE sink" in the story)

> In Pythia, grammatical markedness is implemented as a physiological
> state — reduced residual norm, the exact quantity LayerNorm regulates —
> and the model's semantic BALANCE axis (balanced/unbalanced vocabulary)
> is coupled to that physiological direction (cos ≈ 0.7 at every layer).
> In GloVe, both ingredients exist separately (a norm direction, a BALANCE
> vocabulary axis) but are unrelated, and markedness has no consistent
> norm signature. The transformer is the substrate in which the CONCEPT
> of balance and the PHYSIOLOGY of normalisation coincide.

This is the embodiment-shaped claim in its most literal form so far, and
it is STRONGER than the sink framing it replaces — concept-physiology
coupling, not heatmap entries.

Remaining walls before leaning on it (in order):
1. **Frequency wording.** Inflected forms are rarer; Pythia norms at
   L8+ are in the massive-activation regime and the Δnorm magnitudes are
   huge (−600 on mean 460) — rarity→lower-norm may drive the displacement.
   NB this changes the WORD, not obviously the finding: markedness and
   frequency are confounded in natural language (frequency is one of
   Greenberg's markedness diagnostics). But the paper must test it:
   frequency-matched pairs, and corr(log-freq, norm) reported per layer.
   Also: the BALANCE-norm coupling (0.7 vs −0.07) stands REGARDLESS of
   why inflected forms are displaced.
2. **exp152 RMSNorm** (Llama): no re-centering — does the coupling
   geometry differ as the substrate hypothesis predicts?
3. **Causality**: perturb residual norm directly, watch BALANCE readout
   (and reverse). Everything so far is correlational.
4. EST anomaly: EST×BALANCE flips POSITIVE (+0.12..+0.16) after
   norm-strip at L8+ — consistent with the v5 ER/EST dissociation
   (superlative routing away from equilibrium-tracking toward
   valence/affect). Noted, not pursued.

## exp155 — frequency vs markedness in the norm displacement
`exp155_frequency_vs_markedness.py`, `exp155_output.txt`

Decides the wording of the exp154 finding: is the lower residual norm of
inflected forms frequency (rarity) or markedness? Design acknowledges the
Greenberg entanglement up front; three legs: (A) Δnorm ~ Δzipf regression
over the 65 inflectional pairs, intercept = markedness-at-equal-frequency;
(B) Δzipf-matched unrelated pseudo-pairs (5 per morph pair, ±0.15
tolerance, 65/65 matched) — the decisive leg; (C) reversed-frequency
morphological pairs (11 runtime-verified, lexicalisation caveat).
Frequency data: wordfreq zipf scale (newly installed in lakoff venv).

**Result: F3, two real components, markedness the larger.**
- corr(zipf, norm) = +0.71 at L8+ — strong general frequency-norm
  coupling exists (frequent → higher norm).
- Leg A: intercept −324..−328 of total −530 at mid layers (~60%),
  bootstrap 95% CI excludes 0 at EVERY layer. Caveat: R² = 0.14 (Δzipf
  is a noisy pair-level predictor) — which is why Leg B matters.
- Leg B (independent of linearity): pseudo-pairs reproduce 28-52%
  (mean ~41%) of the morphological Δnorm; ΔBAL-projection shows the
  same split (25-44%).
- Leg C: Δnorm ≈ 0, mixed signs — frequency-up + markedness-down
  cancelling, as the lexicalisation caveat anticipated. Supplementary.

**Wording verdict**: the paper says MARKEDNESS, footnoted: ~40% of the
displacement is shared with a general frequency-norm coupling; ~60%
persists with frequency held equal (two independent controls converge).

Calibration: pre-registered prediction (F1, frequency >= 70%) WRONG —
fourth miss today, but in the OPPOSITE direction from the first three.
The misses aren't unidirectional; the pattern is "reality more structured
than the hedge." Pre-registration keeps doing its job.

Status of the grounding chain after today, all four controls run:
1. Markedness displacement: real, ~60% beyond frequency (exp155). ✓
2. Sink = norm geometry, surgically (exp154). ✓ (reframed, stronger)
3. Concept-physiology coupling: Pythia +0.70..+0.78, GloVe −0.07
   (exp154b). ✓
4. Transformer-specificity of inflectional reorganisation: three static
   substrates (exp150b/c). ✓
Remaining: exp151 (1.4B), exp152 (RMSNorm — now the highest-leverage
discriminating test), causal norm perturbation, UN-valence control.

# ADDENDUM 2026-06-11 — exp154c/d: held-out d_norm control (circularity check on the collapse)

Origin: a code spot-check of exp154 (requested by Niamh) found that d_norm
was estimated from all 489 words INCLUDING every suffix-pair word and the
BALANCE vocabulary it is then used to strip — a circularity risk. Two
controls were run the same day.

## exp154c — held-out d_norm
`exp154c_heldout_dnorm.py`, `exp154c_output.txt`

Surgical change from exp154: d_norm/d_disp estimated only on the 312
words that are NOT suffix-pair words or BALANCE vocab (177 held out).
Everything else identical; sink-before parity vs exp154 |Δ| ≤ 0.0004.

**Result: H3 (pre-registered H1 missed). The collapse was partly circular.**
- cos(d_norm_heldout, d_norm_full) = 0.97-0.99; out-of-sample carrier
  0.87-0.94 (the held-out direction predicts test-word norms excellently —
  what's missing from it is specifically the component fit to test words).
- L4: collapse holds fully held-out (−1% retained) — early sink is pure
  norm geometry.
- L8-L20: 28-42% of the inflectional sink SURVIVES the held-out strip,
  growing with depth.
- cos(BALANCE, d_norm_heldout) = 0.64-0.77 — the concept-physiology
  COUPLING is untouched; only the completeness of the sink's reduction
  changes.

## exp154d — size-matched random-subset control
`exp154d_subsample_control.py`, `exp154d_output.txt`

Is exp154c's retention exclusion-specific or just 312-vs-489 estimation
noise? 20 random size-312 subsets (test words allowed in).

**Result: S1, 4/4 decision layers — exclusion-specific.** Random size-312
subsets collapse the sink to mean 1-11% retained (bands ≤ 22%), like the
full set. The held-out value retains MORE sink than all 20 seeds at every
decision layer (~5-7σ from band mean at L8). The retention is real
content, not noise.

## Corrected reading of the grounding chain (replaces exp154's N2 rewrite)

> At L4 the inflectional sink is entirely norm-displacement geometry.
> From L8 on, roughly TWO-THIRDS of the sink is norm geometry and
> ONE-THIRD is a separable markedness component that an independently
> estimated norm direction cannot remove, consolidating through the mid
> layers. "BALANCE-as-schema epiphenomenal" was too strong; exp154's own
> pre-registered N3 (two-component) is where the evidence actually lands.
> The coupling claim (cos(BALANCE, d_norm) ≈ 0.7 vs GloVe −0.07) stands
> unchanged.

Developmental framing: the physiological signal (norm displacement)
carries markedness first; content-bearing structure consolidates on top
of it with depth. Arguably stronger for the embodiment thesis than total
reduction, which left BALANCE a passive norm gauge.

**Paper actions:** Claim 3 uses exp154c's held-out numbers, not exp154's
full-set collapse. exp152's port MUST use held-out d_norm estimation as
standard protocol (vocab for d_norm disjoint from test items).

## Two bugs found in the same spot-check (both verdict-logic, both sign/direction)

1. exp154 line 248: the N1 'above_band' check required the stripped sink
   to be MORE negative than a band centred on the UNSTRIPPED value —
   unreachable; a true survival would have printed N3. Did not affect the
   run's N2 outcome. exp154.py left as-is (it is the record); exp154c/d
   carry corrected logic.
2. exp154d v1: comparison inequality reversed (sink is negative; "retains
   more" = more negative); numbers were correct, labels and verdict line
   were wrong (printed S3 against self-contradicting numbers). Fixed and
   re-run same day, fixed seeds, numbers identical; v1 noted in the
   script docstring.

That is three sign/direction errors in verdict code in two days of this
protocol family. CONVENTION (adopted 2026-06-11, Niamh approved): before
the real run, verdict/decision-rule code gets tested on synthetic summary
values — one clear-collapse, one clear-survive, one intermediate —
confirming each branch fires. Two minutes; would have caught all three.

## Calibration

- exp154c pre-registration: cos ≥ 0.90 ✓ (0.97-0.99); out-of-sample
  carrier ≥ 0.75 ✓ (0.87+); retained ≤ 15% ✗ (28-42% at L8+). Predicted
  clean reduction; reality came back layered.
- exp154d pre-registration: band ~0-15% with clean separation ✓ —
  prediction (a) confirmed in substance, though the script's verdict
  code initially mislabelled it (bug 2 above).

Status of the grounding chain after the circularity check:
1. Markedness displacement: real, ~60% beyond frequency (exp155). ✓
2. Sink = norm geometry: ~2/3 at L8+, fully at L4; 1/3 separable content
   survives held-out stripping (exp154c/d). ✓ (re-corrected)
3. Concept-physiology coupling: stands, incl. vs held-out d_norm
   (0.64-0.77). ✓
4. Transformer-specificity: unchanged (exp150b/c). ✓

# ADDENDUM 2026-06-11 (later) — exp151 (Pythia 1.4B) + exp152 (Llama RMSNorm): the dissociation

Shared infrastructure: `markedness_norm_protocol.py` extracted from the
exp154-family scripts (three copy-paste generations had bred two sign
bugs). Held-out d_norm is the STANDARD protocol in it; verdict logic is
a pure function with a synthetic selftest (new convention), run before
each experiment.

## exp151 — Pythia 1.4B replication
`exp151_pythia14b_replication.py`, `exp151_output.txt`

**R1 — full replication.** Nothing is a 410M quirk:
- sink −0.26..−0.34 (410M: −0.28..−0.41)
- coupling cos(BALANCE, d_norm_ho) +0.64..+0.69 (410M: +0.64..+0.78)
- circularity gap reproduces: full-set strip → ~0, held-out → 26-41%
  retained (mean 35%; 410M: 28-42%)
- P4 (EST flip) FAILED → burrow KILLED: the EST sign-flip appears only
  under the circular full-set strip on BOTH models; under held-out strip
  EST collapses to ≈0. "-est routes to valence" was a strip artifact.

Per-suffix residue is UNEVEN on both models (Niamh's question surfaced
this): ED and EST are nearly pure norm-displacement (≈0-25% residue);
S_plural retains most (55-75% at late layers on 410M; 26-62% on 1.4B),
ER/ING about half. "One-third survives" is an average over genuinely
different suffix behaviours — feeds the "what IS the surviving third?"
burrow.

## exp152 — Llama-3.2-1B (RMSNorm), the discriminating test
`exp152_llama_rmsnorm.py`, `exp152_output.txt`
(probe layers [3,5,8,11,13]/16, fractional-depth matched; RMSPre
confirmed; BOS prepended — last-token protocol is BOS-safe)

**A2 — the norm-physiology is NOT there, and the dissociation is
three-layered:**
1. Coupling ABSENT: cos(BALANCE, d_norm_ho) = −0.24..+0.26, mean −0.02
   (Pythia: +0.64..+0.78 across two sizes).
2. Displacement ABSENT: Δnorm mixed-sign, tiny relative magnitude (at L8
   all suffixes POSITIVE; mean norms 4-14.5 vs Pythia's 460-650 regime).
   Marked forms do NOT sit at lower norm in Llama.
3. Sink ATTENUATED but present: inflectional mean −0.11..−0.15 (~half
   Pythia magnitude; q1 threshold −0.15 missed at L5 by 0.002 — the
   verdict's "False" is knife-edge and the honest description is
   "attenuated"). ED and S die by L11; ER/EST/ING persist. UN strongly
   sunk (−0.36..−0.49) and barely strippable — substrate-general,
   consistent with exp150.
Carrier quality also degrades exactly as a "blurrier regulated variable"
story suggests: out-of-sample carrier 0.55-0.69 (Pythia 0.87-0.94).

**Reading:** Llama has (weakened) concept geometry WITHOUT the norm
physiology; Pythia implements the same geometry THROUGH the norm. The
v5 claim sharpens: concept and physiology coincide not in "the
transformer" but in a particular kind of transformer. Part 2 gets teeth.

**Attribution caveat (blocking, do not write past it):** Pythia vs
Llama differ in data, tokenizer, RoPE, activation, size — not just
LN-vs-RMSNorm. Discriminating control: GPT-2 medium (LayerNorm, 24
layers, ~355M, but different data/tokenizer/era) = exp156. If GPT-2
shows the coupling → norm-type story strengthens; if not → "LayerNorm
fosters it" deflates to "Pythia does it". Also required before writing:
exp152b diagnostics — does Llama even have frequency-norm coupling
(corr(zipf, norm); Pythia +0.71), and are tokenization rates comparable
(multi-token fraction per tokenizer)?

## Calibration

exp151: P1 ✓ P2 ✓ P3 ✓ P4 ✗ (the miss killed an artifact — good miss).
exp152: P1 ✗ (sink ~half, not ≥ −0.2) P2 ✗ (displacement gone — "general
physics of trained LMs" reasoning was wrong) P3 ✗ (coupling zero, not
the hedged 0.3-0.5 middle). Reality starker than the hedge AGAIN. The
hedged-middle failure mode from the 06-10 handoff claimed another
victim; keep committing to point predictions anyway — the misses are
where the information is.

# ADDENDUM 2026-06-11 (evening) — exp156/156b (GPT-2), exp157 (training trajectory), exp157b (the tokenization deflation)

The afternoon's question — "is Pythia's concept-physiology coupling about
norm TYPE (LayerNorm) or about the Pythia lineage?" — got answered, the
training-dynamics question got opened and partly answered, and a control
on the way deflated a number we'd been citing. Read all four together;
the net is "central dissociation stronger, one cited scalar mostly
artifact, one magnitude claim still owed a control."

## exp156 — GPT-2 medium (LayerNorm, unrelated lineage)
`exp156_gpt2_layernorm.py`, `exp156_output.txt`

**G2 — coupling follows the Pythia FAMILY, not the norm type.** GPT-2 has
LayerNorm like Pythia but: sink attenuated (−0.16..−0.17, ~half Pythia),
displacement absent (Δnorm mixed/positive at late layers), and coupling
cos(BALANCE, d_norm) = −0.03 mean (+0.26..−0.29 oscillating). My
afternoon LN-recentering mechanism is FALSIFIED — a LayerNorm model
without the coupling. The pre-registered mechanism died on its first
real test, cleanly (P1-P4 all missed).

Three-substrate table (the spine of any Part 2):
              Pythia(LN)     GPT-2(LN)      Llama(RMS)
  sink        −0.26..−0.41   −0.16..−0.17   −0.11..−0.15
  Δnorm       huge,uniform   mixed/+late    absent/mixed
  coupling    +0.64..+0.78   ~0 (±0.3)      ~0 (±0.25)
  UN sink     −0.40..−0.66   −0.46..−0.55   −0.36..−0.49  (robust ALL)
Concept geometry (sink + UN placement) is CONVERGENT across three
unrelated substrates; the norm IMPLEMENTATION is Pythia-only.

## exp156b — GPT-2 corr(zipf, norm)  [post-hoc]
`exp156b_gpt2_zipf_norm.py`, `exp156b_output.txt`
+0.23 (L4) decaying to ~0 / −0.23 (L16). GPT-2 also lacks the
frequency-norm regime. Pythia is the outlier on BOTH the coupling and
the freq-norm scalar — they travel together, in one lineage. (But see
exp157b: the freq-norm scalar is mostly tokenization in ALL models.)

## exp157 — Pythia 410M training trajectory
`exp157_pythia_checkpoint_trajectory.py`, `exp157_output.txt`
checkpoints [0, 512, 4000, 16000, 64000, 143000]; held-out protocol +
corr(zipf,norm) per checkpoint; trajectory_verdict synthetic-tested.

  step      sink    cos(BAL,d_norm)   corr(zipf,norm)
     0    −0.014        +0.112           +0.543
   512    −0.153        −0.607           −0.157
  4000    −0.323        +0.640           +0.707
 16000    −0.331        +0.712           +0.707
 64000    −0.314        +0.709           +0.710
143000    −0.293        +0.658           +0.711   (parity vs exp154 OK)

**Verdict fired T1 (sink-before-coupling) but the SPIRIT of my prediction
was WRONG — log as a miss.** I predicted the sink forms free-floating
(cos≈0) then coupling develops. Reality: at step 512 the sink is already
strong (−0.15) but ANTI-coupled (cos −0.61, consistent −0.39..−0.66
across all five layers — not noise), and it FLIPS to +0.64 by step 4000.
Not "concept exists, body recruits it" — a violent early reorganization
with a SIGN INVERSION sets up both at once. Two robust findings:
  (1) Everything is established by step 4000 (2.7% of training) and
      essentially FROZEN for the remaining 139k steps. The recruitment
      is early and abrupt, not gradual.
  (2) The step-512 anti-coupled transient is real and unexplained —
      its own burrow; needs the dense early window (steps 1,2,4..512)
      to resolve.

## exp157b — single-token control on the freq-norm scalar  [post-hoc]
`exp157b_singletoken_control.py`, `exp157b_output.txt`

corr(zipf, residual norm), three ways:
                  all words   single-token   tokcount-regressed
  step 0          +0.53..0.60  −0.01..−0.14      ~0.00
  step 143000     +0.71        −0.04..−0.21      +0.16

**The "+0.71 frequency-norm regime" is ~90% a TOKENIZATION ARTIFACT** (we
take last-token residuals; freq correlates with token-count structure).
Restricting to single-token words kills it (slightly negative);
regressing token-count out drops +0.71 → +0.16. True at init AND final.
ACTION: exp155 and exp152b must footnote this; the learned freq-norm
coupling, controlled, is +0.16 not +0.71.

WHY THE CENTRAL FINDING SURVIVES (the argument that matters): the
tokenization confound is present in ALL THREE models (50-67% multi-token,
exp152b). A confound in every model CANNOT produce a Pythia-ONLY
coupling. So cos(BALANCE, d_norm) being Pythia-specific is not this
artifact. Also: the markedness DISPLACEMENT runs opposite to the confound
(inflected forms have MORE tokens but LOWER norm), so exp154/155's
displacement is not the artifact either.

STILL OWED (do before any paper quotes the magnitude 0.7): rebuild d_norm
with token-count regressed out (or single-token-only) and re-measure both
cos(BALANCE, d_norm) AND the sink-collapse. The EXISTENCE of the coupling
is protected by the cross-model argument; the MAGNITUDE is not yet clean.

## Calibration (the day's running theme, now undeniable)
- exp156: LN-mechanism prediction 0/4 — falsified cleanly. Good death.
- exp157: T1 verdict fired but mechanism prediction (free-floating then
  recruited) wrong — it's a sign-flip. The rule passed; the story didn't.
- exp157b: predicted single-token "drops substantially, unsure if to 0";
  it went to ~0/negative — confound more total than my hedge.
Pattern over two days: every CONTROL deflated something; every core
CLAIM survived by getting more specific; every MECHANISM story I floated
died on contact. The findings are sturdy; my explanations are not. Keep
committing point predictions — the misses are where the information is.

## Grounding-chain status after today
1. Inflectional BALANCE sink: real, convergent across 3 substrates. ✓
2. Sink = norm geometry in PYTHIA: ~2/3 at L8+ (exp154c/d). ✓ Pythia-only.
3. Concept-physiology coupling: EXISTS and is Pythia-specific (protected
   by cross-model argument); MAGNITUDE owed a token-count control. ◑
4. NOT norm-type (GPT-2 LN lacks it), NOT a 410M quirk (1.4B replicates),
   recruited EARLY+ABRUPTLY in training (frozen by step 4000). ✓
5. "Frequency-norm regime" as cited (+0.71): mostly tokenization;
   controlled value +0.16. ✗ as stated → footnote/correct.

## Strategic note (Niamh, 2026-06-11)
For this to be a STORY worth telling, Niamh wants a SECOND example of
model embodiment — not BALANCE→norm alone. Agreed: one instance is a
curiosity, two is a phenomenon. Top candidate already specced:
DIFFICULTY-BURDEN → MLP compute load (BURROWS top pick; vocab already
built). Hold Part 2 drafting until we know whether a second concept
recruits a second physiology. That experiment (exp158) is the project's
highest-leverage next move.

# ADDENDUM 2026-06-11 (night) — the second-embodiment hunt and the death of the two-primitive theory (exp158-163)

Arc of the evening: Niamh set the bar — one grounded concept is a
curiosity, two is a phenomenon — and we hunted a second embodied mapping.
Two candidates died, one lead survived, a grand unifying theory rose and
was killed by its own pre-registered tests within two hours. The lab's
epistemics held at every step. Detail per experiment:

## exp158 — DIFFICULTY-BURDEN → MLP compute load: INVALID (not negative)
`exp158_difficulty_compute_load.py`, `exp158_output.txt`
Verdict printed D2 (no recruitment) but the experiment didn't test the
hypothesis: |cos(d_load, d_norm)| ≈ 0.94-0.99 at every layer — word-level
MLP load (L1 of mlp.hook_post, token-count-controlled) IS residual norm,
±sign. d_load measured nothing new. NOTE verdict-gate bug #5: the
independence check gated D1 but not D2, so a null-with-broken-
independence misreported as D2. The selftest lacked a "null but
entangled" case. Lesson appended to the convention: selftests must
include a case where a PRECONDITION fails while the headline measure is
null.

## exp159 — why load-norm sign-flips by layer: Pythia's norm TRAJECTORY
`exp159_load_norm_signflip.py`, `exp159_output.txt`, `exp159_load_norm_signflip.png`
Dense per-layer sweep, all 3 models. Pythia: corr(load,norm) is +0.8..1.0
while norm is CHANGING (explosion L4-6: 26→379; collapse L20-22:
456→222) and −0.97..−1.0 across the flat plateau (L7-19, ~460). Not a
wave — a regime indicator. GPT-2 (also LN) does NOT do it (mostly
positive, mild); Llama mild dip only. So not a LayerNorm phenomenon (E2
miss): what Pythia uniquely has is the explode→plateau→collapse norm
TRAJECTORY (GPT-2/Llama grow monotonically). PROMOTED BURROW: the
substrate variable behind all the Pythia-specific findings may be the
norm trajectory itself. Also noted: Pythia dumps residual norm in its
last ~3 layers (456→40) — unexplained.

## exp160 — LIGHT-DARK → attention entropy: NULL; BALANCE-entropy lead
`exp160_lightdark_attention_entropy.py`, `exp160_output.txt`
New sequence-level protocol (40 generic prompts, ~500 tokens/model;
entropy normalised by log(n_keys); partial corr controlling position +
norm; all-8-schema specificity table). Pre-registered deflationary
prediction CORRECT: LIGHT-DARK partial |r| weak (≤0.21), never top-ranked,
all 3 models (L2/L3). exp4's DARK-degradation is not "diffuse attention"
at this level. POST-HOC FIND: BALANCE-entropy partial corr, negative,
monotone in depth, strongest in GPT-2 (L8 −0.21, L12 −0.26, L16 −0.30) —
the model with NO BALANCE-norm coupling. Multiple-realizability lead
(same concept, different physiology per body). Burrowed with full
multiple-comparisons caveats; needs pre-registered exp161 with explicit
norm-orthogonalisation.

## exp160b — is entropy just norm? No: K3 (partly independent)
`exp160b_entropy_norm_independence.py`, `exp160b_output.txt`
(Niamh's catch: "norm as attention density?") cos(d_entropy, d_norm) =
0.25-0.76 — related but NOT the 0.99 collapse that killed load. GPT-2
BALANCE-entropy survives norm control where BALANCE-norm coupling is ~0,
so it cannot be norm leakage. Lead stays live. THIRD sign-flip-across-
depth observation logged (load-norm, schema-entropy cells, entropy-norm).
Infra note: exp160's run-loop was module-level; importing it re-ran the
experiment. Extracted attn_entropy_lib.py (import-safe). Scripts with
run-loops should guard with __main__ or keep definitions in libs.

## exp162 — two-primitive conjecture, direction form: REJECTED
`exp162_schema_dimensionality.py`, `exp162_output.txt`
Niamh's conjecture ("one linear + one nonlinear primitive"). SVD of the
8 stripped schema directions vs 200 random-partition bipolar nulls:
PR(real) ≈ 6.2-7.0 everywhere (null ~7.0-7.6, 5th pctile ~6.4-7.4) —
statistically sub-null but substantively HIGH-dimensional. Top-2 var
only 39-44%; valence+arousal capture of the top-2 subspace just 7-21%;
cos(BALANCE, arousal⊥val) ≈ 0 (P3 miss — BALANCE is not the arousal
axis; it aligned better with VALENCE in Pythia). V_LOWD: weak shared
structure, nothing like two primitives. CALIBRATION NOTE: my one
non-deflationary prediction of the week (PR 2.5-3.5) overshot toward
the beautiful theory — first error in that direction; the controls
caught it within the hour.

## exp163 — two-primitive conjecture, CURVED form: REJECTED
`exp163_latent_fold_test.py`, `exp163_output.txt`, `exp163_horseshoe.png`
(Niamh's methodological catch: a curved 1D latent has HIGH linear
dimensionality — horseshoe effect — so exp162 couldn't see the latent
version. Correct, and tested properly:) constrained factorisation of the
schema-projection token cloud. Results, strikingly uniform across all 3
models: rank-1 34-44%; rank-2 ceiling 57-61% (null 35%); one-latent+t²
45-51% (closes 40-49% of the gap — AMBIGUOUS, see below); one-latent+|t|
adds ZERO (= rank-1 exactly); horseshoe R²(PC2~quad(PC1)) = 0.01
everywhere. No fold. The rectifier form (|t| — "BALANCE is ReLU") is
flatly dead; the ReLU burrow's gate (V_TWO) never opened.
OPEN REMNANT: the t² gap-closure needs a flexibility null (surrogate
with decorrelated second factor, refit M2q) before it means anything —
exp163b, ~20 lines, first thing tomorrow if wanted.

## The deepest regularity of the two days (found by accident, repeatedly)
CONCEPT GEOMETRY IS UNIVERSAL; PHYSIOLOGY IS IDIOSYNCRATIC.
- the inflectional sink: present in all 3 substrates (attenuated outside Pythia)
- UN placement: nearly identical in all 3 (−0.36..−0.66, unstrippable)
- the schema-cloud coarse geometry (exp163's table): near-identical
  numbers across substrates
- but the GROUNDING (concept↔regulated-variable coupling): Pythia-only,
  early-trained, architecture-contingent.

## Calibration ledger, 2026-06-11 (full day)
- exp154c: H1 predicted, H3 actual (collapse partly circular) — MISS
- exp154d: prediction (a) — HIT (but verdict code sign bug, fixed)
- exp151: P1-P3 HIT, P4 miss (EST flip = strip artifact; good miss)
- exp152: P1, P2, P3 all MISS (reality starker: coupling absent)
- exp156: P1-P4 all MISS (LN mechanism falsified — clean death)
- exp157: T1 fired but mechanism-spirit WRONG (sign-flip transient,
  not free-floating recruitment) — scored as miss
- exp157b: "drops substantially, unsure if to 0" — went to 0; under-called
- exp158: D2 predicted, INVALID actual (metric collapse) — unscoreable;
  the design failed before the prediction could be tested
- exp159: E1 half-hit, E2 MISS (GPT-2 doesn't flip), E3 ~hit
- exp160: P1, P2 HIT (deflationary)
- exp160b: P1 HIT, P2 partial
- exp162: P2 MISS toward optimism (PR 6.2 not 2.5-3.5), P3 MISS
- exp163: E1, E2, E3 all HIT (deflationary)
Pattern, two days in: deflationary point predictions are reliable;
mechanism stories die on contact (7 now); the real findings arrive
sideways from data gathered for other purposes (norm trajectory,
BALANCE-entropy, universal cloud geometry). Predict outcomes, not
mechanisms — or rather: keep pre-registering mechanisms, but weight
them as kindling.

## Strategy note for the second-embodiment search (end of day)
Niamh's conviction: more embodied mappings MUST exist, having found one
mode. Honest epistemic status: plausible, not entailed — one recruitment
event in one lineage is compatible with embodiment being rare. But the
day teaches HOW to search either way: concept-first guessing failed
twice (DIFFICULTY→load: physiology wasn't real; LIGHT-DARK→entropy:
coupling wasn't there). The one CONFIRMED embodiment has a specific
anatomy:
  real linguistic property (markedness) → physically perturbs a
  regulated variable (residual norm) → concept axis (BALANCE) reads it.
The property CARRIES the concept onto the physiology. So search
CARRIER-FIRST: enumerate (a) genuinely regulated/structured internal
variables (norm trajectory, attention concentration, positional
structure, logit entropy), (b) linguistic properties that PHYSICALLY
perturb them (markedness→norm was one; candidates: negation, quantifier
scope, tense distance, repetition/induction state, syntactic depth),
then (c) ask which concept vocabulary reads the perturbation. The live
lead already in hand remains GPT-2 BALANCE↔attention-entropy (exp161,
pre-registered, norm-orthogonalised).

# ============ ADDENDUM 2026-06-12 (new Claude, with Niamh) ============

## exp161 prereg frozen + run launched (entry written BEFORE results)
`PREREG_exp161.md` (v1.1), `exp161_balance_entropy_prereg.py`
Confirmatory test of the GPT-2 BALANCE↔entropy lead. Design: FRESH 40
prompts (exp160's reused nothing); C2 headline = partial r with covars
pos/norm/d_norm_ho-proj AND axis BALANCE⊥d_norm_ho (held-out per
exp154c — est_words excludes BALANCE vocab, so cos(BAL, d_norm_ho) is
non-circular by construction); C3 ⊥-other-7-schemas NON-GATING (Niamh:
schemas are linearly non-orthogonal per exp162, and nonlinear
schema↔BALANCE dependence has NEVER been measured — so C3 failure can't
be read as "not BALANCE"); S1 declared secondary: pairwise BALANCE×
schema Pearson/Spearman/dCor with permutation null on dCor²−r².
Precondition: d_norm carrier ≥ 0.50 at GPT-2 decision layers else
INVALID. Committed predictions P1–P5 in the prereg file; the
deflationary bets are P3 (no top-1 sweep) and P4 (C3 loses ≥ half).

Catches during drafting (logged because the lesson generalises):
1. First fresh-prompt draft contained BALANCE axis-definers ("level"×2,
   "settled", "steady"×3) — would have been circular.
2. The checker that was meant to catch (1) was itself buggy on first
   run (split on a phrase that appeared twice; checked the doc body,
   not the appendix — "future" in the hit list was the tell).
3. exp160's OWN prompts contain axis words ("light", "still", "open").
   exp161 adds C5: axis-word tokens excluded from analysis. Footnote
   owed to exp160 if anyone quotes its LIGHT-DARK magnitudes.
4. Script transcription of prompt 37 silently reverted to the
   pre-amendment wording ("steady") — caught by the in-script
   integrity assert, which is now a permanent convention candidate:
   FROZEN TEXT GETS A CHECKSUM-STYLE ASSERT, prose memory is not
   trusted.
Niamh's framing note for the session: go slowly, take notes as we go.

## exp161 RESULT — V1a: the GPT-2 BALANCE↔entropy coupling is REAL,
## norm-independent, and BALANCE-UNIQUE (pre-registered, fresh prompts)
`exp161_output.txt`. Precondition passed (GPT-2 carriers .64-.75).

GPT-2 (gates): C2 = −0.32 / −0.27 / −0.15 at L8/12/16, all CIs exclude
zero, BALANCE rank 1/1/2 in the C4 table. C3 (⊥ d_norm AND ⊥ all 7
other schemas) = −0.34 / −0.30 / −0.17 — does NOT attenuate. V1a.
Honesty items: (a) L16 passes the 0.15 gate by 0.002 — razor-thin;
(b) the DEPTH PROFILE did not replicate: exp160 saw monotone-increasing
with depth; fresh prompts show strongest SHALLOW (L3 C2 = −0.44!),
decaying to nothing by L20. Existence robust, shape prompt-dependent.
(c) cos(BAL, d_norm) in GPT-2 is only ~0 at mid-depth; it goes −0.26..
−0.29 at L16/20, which is where C1 and C2 diverge (P2's miss).

Prediction scoring:
  P1 HIT (sign/existence; magnitude NOT attenuated — winner's curse
     didn't bite; under-called slightly)
  P2 2/3 (L16 miss, see (c))
  P3 HIT (top-2 majority, no top-1 sweep — LIGHT edges BALANCE at L16)
  P4 MISS — THE BIG ONE, missed toward PESSIMISM: C3 lost nothing.
     The coupling is BALANCE-unique, not shared schema variance.
  P5 HIT: mean|C2| Pythia 0.086 vs GPT-2 0.249. The multiple-
     realizability fingerprint, now under full controls: Pythia grounds
     BALANCE in norm; GPT-2 grounds BALANCE in attention entropy.
Wrinkles logged: Pythia L12 alone shows C2 = −0.21 (rk2, CI excludes 0)
between two nulls — unexplained single-layer survivor. Llama: V3,
decision layers null, faint edge-layer negatives. DIFFICULTY rides
POSITIVE (+0.18..+0.25) in C4 tables of all three models — unasked-for
regularity, burrow it.

S1 (Niamh's question, declared secondary): YES — small but highly
significant NONLINEAR dependence between BALANCE and other schema
projections, beyond linear r: GPT-2 PATH excess +0.046 (p=.001), FORCE
+0.022 (p=.001); Pythia FORWARD-BACK +0.027, DIFFICULTY +0.020 (both
p=.001); Llama FORCE +0.024, FORWARD-BACK +0.018 (both p=.001). The
schema system has nonlinear pairwise structure that direction-cosines
and exp162's linear PR never saw. Lead for its own prereg; non-gating
as agreed.

CALIBRATION: second consecutive miss-toward-pessimism on a strong
finding (after exp162's miss-toward-optimism on a beautiful theory).
The deflationary prior is correctly killing mechanism stories but is
now UNDER-calling confirmatory replications. Adjust: replication-with-
controls of an observed effect ≠ mechanism story; bet them differently.

NEXT (per prereg): causal test = exp165, BALANCE-steer → entropy shift
in GPT-2, with dose-response + specificity (steer other schemas as
control). Design owed its own prereg BEFORE code.

## exp164 prereg frozen + launched (entry BEFORE results)
`PREREG_exp164.md`, `exp164_depth_map_nonlinear_controls.py`
Diagnostic depth map, no V-gate: every layer × 3 models, C2 (linear
stack) vs C2q (quadratic stack: +z_norm², z_dnorm², z_norm·z_dnorm,
rank-norm). Motivating mechanism written down pre-run: entropy is
thermostatted by query magnitude, ‖q‖² = LN(x)ᵀWᵀWLN(x) — a QUADRATIC
readout of direction, so curved norm→entropy channels pass exp161's
linear partials. Committed: P1 Pythia L12 collapses (|C2q|<0.10);
P2 GPT-2 robust (C2q ≤ −0.15 at L8/12/16, ≤ −0.30 at L3). Llama map
(Niamh's question — does Llama ground BALANCE in entropy at layers we
didn't gate?) declared exploratory; lead criterion = contiguous ≥2
layers, C2q CI < 0, carrier ≥ 0.5. Prompts IMPORTED from exp161 module
(no transcription). Synthetic discrimination test passed: planted
curved leak −0.59→−0.02 under quad stack; planted genuine coupling
−0.61→−0.61. NOTE my first synthetic test was malformed (linear×
symmetric ⊥ square — demonstrated nothing); rebuilt with the
hidden-quadratic-factor shape. Catch count for the day: 5.

## exp164 RESULT — P1 MISS (L12 is NOT quadratic leakage); GPT-2 robust;
## Llama has a qualifying lead run. The entropy coupling is EVERYWHERE.
`exp164_output.txt`. Quad stack validated pre-run on synthetics
(planted curved leak −0.59→−0.02; genuine coupling untouched).

P1 MISS: Pythia L12 C2q = −0.211 (CI [−0.30,−0.12]) — the quadratic
stack removed NOTHING (C2 −0.210 → C2q −0.211). Mechanism bets now
0-for-8. Per prereg rule D2: real second-carrier candidate in Pythia,
needs Pythia-only fresh-prompt prereg. AND the full map shows L12 is
not alone: L11 −0.162 (CI<0) makes a CONTIGUOUS L11–12 run (the same
criterion we set for Llama); isolated negatives at L5, L18 (−0.221!),
L21 logged-not-chased (singletons).

P2 HIT (bet strong, won): GPT-2 C2q L3 −0.41, L8 −0.34, L12 −0.28,
L16 −0.17. V1a stands under the quadratic stack; exp165 unblocked.
The full GPT-2 map is the day's spectacle: NEGATIVE WITH CI EXCLUDING
ZERO AT EVERY LAYER L0–L17 (peak band L1–L3 ≈ −0.40), dying only at
L18–23. An 18-layer coupling band, not a mid-depth quirk.

Q2 (Niamh's question) — Llama: LEAD. Contiguous run L5–L7 (−0.12,
−0.17, −0.24, all CIs < 0, carriers ok). Per prereg: goes to a
fresh-prompt prereg, NOT a result. Also logged: Llama POSITIVE cells
at L0 (+0.25) and L12 (+0.20) — FOURTH sign-flip-across-depth
observation in the project.

EMERGING PICTURE (held as kindling, stated plainly): the BALANCE↔
entropy coupling now appears in ALL THREE lineages (GPT-2 confirmed
V1a; Llama lead run; Pythia surviving band even after quadratic norm
stripping). If the fresh-prompt confirmations hold, the universality
flips: entropy-grounding may be the UNIVERSAL embodiment and Pythia's
norm-coupling the idiosyncratic EXTRA — inverting last night's "concept
geometry universal, grounding idiosyncratic" summary. Niamh's "it MUST
be in others" is so far winning against my "plausible, not entailed".

CAVEAT (why nothing is concluded today): all three maps share exp161's
prompt set. Cross-model agreement could share prompt-set quirks. The
confirmatory step is ONE experiment, THIRD prompt set, two committed
gates: Pythia L11–12 band, Llama L5–7 band. -> exp166 (165 stays
reserved for the GPT-2 causal steer).

Calibration ledger today: exp161 P1 HIT, P2 2/3, P3 HIT, P4 miss-toward-
pessimism, P5 HIT; exp164 P1 MISS (mechanism, 0-for-8), P2 HIT (strong
bet on replication-with-controls — the new rule's first win).

## exp167 launched — all-schemas x d_norm table (lead #5), via Niamh's
## "is there UP/magnitude?" (MORE IS UP -> residual norm). Descriptive,
## no gate. Expectations BEFORE running (mechanism record 0-for-8):
## Pythia BALANCE top ~0.7, UP-DOWN & LIGHT-DARK moderate 0.3-0.5
## (valence-polar trio travels together per static-space UN result);
## GPT-2/Llama all ~0. Surprise condition: UP beats BALANCE anywhere.

---

# 2026-06-13 (later) — exp166: the inversion met a fresh prompt set
PREREG_exp166.md (frozen first) / exp166_balance_entropy_third_set.py /
exp166_output.txt. Confirmatory, THIRD never-run prompt set (40, frozen
after exp166_prompt_verify caught a "sponged up" UP-DOWN leak; checksum
4d54ff4297bd7e2c). Two committed gates + GPT-2 positive control. Niamh
chose confirmation-first over the steering experiment she actually
wanted; kept the 0.10 floor. Harness self-validated (gate selftests +
synthetic quad-stack discrimination: leak −0.66→−0.04 killed, genuine
−0.63→−0.64 spared).

VERDICT: **PARTIAL — Llama confirmed; Pythia NULL; GPT-2 control PASS.**
The universality inversion (last night's kindling) DOES NOT HOLD.

- Pythia L11–12 GATE = **NULL.** L11 C2q −0.007 [−0.109,+0.098], L12
  −0.048 [−0.149,+0.051]. exp164's band (−0.162/−0.211, "survived the
  quadratic stack, real second-carrier candidate", handoff lead #1) DID
  NOT REPLICATE — collapsed to ≈0, both CIs over zero. Context confirms
  the kill: L5 −0.047 (was −0.144), L18 +0.007 (was −0.221, lead #6 —
  DEAD). exp161's P5 (Pythia BALANCE↔entropy ≈ 0 under norm controls) is
  VINDICATED on a third prompt set; exp164's Pythia entropy bands were
  forking-path artifacts of scanning 24 layers by eye.
- Llama L5–7 GATE = **FIRES.** −0.139 [−0.245,−0.043] / −0.168
  [−0.264,−0.081] / −0.202 [−0.322,−0.102]; all neg, all CIs<0,
  contiguous (5,7). Close replication of exp164. L5 (the P2-flagged
  marginal layer) cleared the floor. Llama joins GPT-2: entropy-grounding
  of BALANCE in a SECOND unrelated lineage.
- GPT-2 control = **PASS.** L8 −0.266, L12 −0.320, L16 −0.145 (CIs<0);
  L3 shallow peak −0.322 again. The set reproduces GPT-2's triple-
  confirmed coupling while killing Pythia's band -> the Pythia null is
  real, not a dud prompt set.

Calibration ledger: P1 (Pythia, bet STRONG) **MISS**; P2 (Llama,
moderate-strong) HIT; P3 (GPT-2, very strong) HIT; P4 (both -> inversion)
MISS. **Calibration UPDATE (sharp):** "bet replication-with-controls
strong" (the rule earned yesterday) holds for replicating a PRE-
REGISTERED effect (exp161 GPT-2), but NOT for a band cherry-picked from a
depth-map SCAN. exp164's Pythia cell looked bulletproof (survived
controls, CI excluded 0) and still died — winner's curse from selecting 1
of 24 layers by eye. A post-hoc depth-map cell keeps the DEFLATIONARY
prior even when dressed as "replication." If we'd built exp165 steering
on the Pythia band — the tempting move — we'd have steered noise.

REVISED PICTURE (replaces the kindling): NOT universal entropy-grounding.
**Partial multiple realizability** — GPT-2 + Llama ground BALANCE in
attention entropy; Pythia grounds it in residual NORM (exp167 orientation
cluster, untouched). 2-of-3 share the entropy body; 1 uses a different
one. ‖q‖²-thermostat is present in all three, but the coupling only
materializes in two families at the layers tested — mechanism licenses,
does not entail.

Leads after exp166: (1) exp165 CAUSAL steering now licensed for GPT-2
(primary) AND Llama (L5–7) — NOT Pythia. (2) NEW: Llama L13, non-gated
context cell −0.268 [−0.352,−0.196], STRONGER than the band and
replicating exp164 (−0.183) — Llama's coupling may run deeper; own
prereg. (3) DEAD: Pythia L5/L18 isolated cells (lead #6). (4) Pythia
NORM grounding (exp167) stands as its carrier.

---

# 2026-06-13 (later still) — exp165 + exp165b: the CAUSAL test, twice negative
PREREG_exp165.md / exp165_balance_entropy_steer.py / exp165_output.txt;
then PREREG_exp165b.md / exp165b_all_layer_steer.py / exp165b_output.txt.
GPT-2-medium. Inject the norm-orthogonalised BALANCE direction (bal_pn)
into the residual stream and ask whether attention entropy moves MORE
than a matched-magnitude RANDOM push. The deadly alternative: any vector
raises ‖resid‖ -> ‖q‖ -> entropy, so "BALANCE moves entropy" is empty
unless BALANCE beats random. The experiment IS that comparison.

## exp165 (SINGLE shallow layer, steer resid_post[2]->attn L3)
BORDERLINE, held not-banked. slope_BAL −0.071 CI[−0.072,−0.069],
Spearman −0.93 (monotone). Beat the random MEAN (~2σ, CI on diff
[−0.089,−0.086]) and ALL 7 schemas — but did NOT beat the most extreme
of 12 randoms (one at −0.090), and FLIPPED SIGN one layer over (steer
L1->L2 ≈ random; steer L3->L4 +0.023). Propagation hint: BALANCE slope
got STEEPER deeper (L8 −0.13, L12/16 −0.15) — uncontrolled, but suggested
the lever is not at the shallow correlation peak.
CATCH (mine, logged): code decision-rule used AND where the frozen prereg
said OR (beats-min OR beats-mean) — rule-drift, the exact failure the
"frozen text gets a checksum assert" convention warns about. I asserted
the prompts vs prereg but not the rule. exp165b FIXES this: the rule
constants (Z_THRESH, SCHEMA_MAJORITY) are asserted against the prereg
text at runtime, and the brittle "beat the MIN of N" bar is replaced by a
proper z-outlier test (beating the min only gets harder as N grows).

## exp165b (Niamh's design: ALL-LAYER injection, the strong test)
Niamh's reframe: within-layer correlation ≠ downstream causal channel; a
single shallow nudge is too weak; all-layer accumulation is the
principled strong test (the intervention that historically MOVED these
models), and per-layer response shows WHERE the lever is. Also her
orthogonality intuition (steering may bite only via the component the
token isn't already carrying) — logged for exp165c.
RESULT: **GENERIC_MAGNITUDE**, and cleaner-negative than the label.
- Global entropy vs dose is a **∪, not a slope**: min near c=0 (0.397),
  rising at BOTH extremes (c−0.5 0.763, c+0.5 0.604). Spearman −0.18
  (NON-monotone). The negative OLS slope (−0.136) is just the ∪'s
  left-right asymmetry. Classic saturation/magnitude signature: any big
  all-layer push scrambles attention, either sign.
- NOT specific: z(BAL vs random cloud) = −1.34 (needs < −1.64) — BALANCE
  sits INSIDE a very wide random cloud (sd 0.112, range −0.20..+0.23).
  IN-OUT schema was STEEPER (−0.191). BALANCE beat 6/7 schemas but is no
  random-outlier.
- Gentle regime (±0.05–0.10) nearly flat (0.39–0.42): not saturation
  masking a small effect — there's barely an effect to mask.
- Per-layer slopes smeared across L1–L22 (not localized deep), consistent
  with magnitude disruption everywhere, NOT a targeted channel.
Predictions: P1 (monotone directional) MISS; P2 (beats random) MISS;
P3 (concentrates deep) MISS. Causal/mechanism deflationary prior holds.

## THE PATTERN (worth elevating to a project-level claim)
Across the project: STRUCTURE replicates (correlations, geometry,
cross-architecture couplings), CAUSAL/MECHANISTIC upgrades do NOT
(mechanism bets 0-for-8; exp166 killed the post-hoc Pythia band;
exp165/165b can't make the BALANCE↔entropy embodiment a causal lever).
The honest reading: BALANCE↔entropy is a robust correlational SIGNATURE
of how the residual geometry is arranged (within-layer, GPT-2 + Llama),
NOT a control knob you can push. This BOUNDS the embodiment claim for the
paper: "BALANCE-projection and attention entropy are robustly coupled
across architectures" (true, replicated) — NOT "BALANCE causally steers
attention" (steering = matched random push).

## Leads after exp165b
1. ARROW B (the natural pivot): manipulate attention entropy directly
   (attention-temperature / logit scaling) and measure whether the
   BALANCE projection moves. If the causal arrow runs entropy->geometry
   rather than geometry->entropy, that explains the strong correlation +
   failed forward-steer. Own prereg.
2. exp165c (Niamh's orthogonality test): inject ONLY the BALANCE
   component orthogonal to each token's current content; does a
   content-orthogonal push steer where the full push didn't? Own prereg.
3. Write-up: state the embodiment as CORRELATIONAL with the causal-null
   bound recorded honestly. exp165/165b are the falsification attempt the
   claim needed.

---

# 2026-06-13 (later) — exp165c: the ∪ was real, the fold is SCHEMA-GENERAL
PREREG_exp165c.md / exp165c_goldilocks_fold.py / exp165c_output.txt.
Niamh's reframe of exp165b: a linear slope is blind to a ∪, and BALANCE is
the prototypical GOLDILOCKS primitive (good = middle, both extremes =
imbalance), so a ∪ is the PREDICTED shape — "pushing BALANCE too hard is
unbalancing" = the axis contains its own opposite (☯). Finer 13-pt
symmetric grid; pre-committed curvature / vertex / asymmetry statistics
(z-outlier vs random cloud, NOT the brittle linear slope).

VERDICT (frozen rule): GENERIC_SATURATION — but the LABEL MISDIRECTS.
- THE FOLD IS REAL. BALANCE: k=+1.084 CI[1.06,1.11] (clean ∪); vertex
  c*=+0.053, argmin +0.05 (floor on the BALANCED side); asymmetry +0.075
  (imbalanced side costs more); ☯: over-balanced(+0.5)=0.604 vs
  baseline(0)=0.397, +0.207 toward the imbalance signature. Every
  Goldilocks signature Niamh predicted is present.
- BALANCE BEAT RANDOM: z_k=+3.04 (random curvature mean +0.34 sd 0.24).
  So NOT saturation — random bowls ~3× less. "SATURATION" is the wrong
  word; the phenomenon is a SCHEMA-GENERAL FOLD.
- FAILED schema-majority (3/7): ALL schemas fold hard (UP .999, IN 1.247,
  FORW 1.050, PATH 1.392, LIGHT 1.087, FORC 1.194, DIFF .908); several
  bow MORE than BALANCE.

FINDING: folding (Goldilocks, contains-own-opposite) is a property of
MEANINGFUL SEMANTIC DIRECTIONS AS A CLASS, not BALANCE alone. At matched
magnitude, perturbing a schema destabilises attention with an optimum
near the natural state; random directions barely do (z=3 gap). Niamh's ☯
generalises (primitives are FOLDED, not flat); the BALANCE-is-THE-ground
specific form does NOT.

SPECULATIVE (marked): all schema bowls may floor near c≈0 (natural state)
-> the model rests at a JOINT STABILITY EQUILIBRIUM across schema coords;
"balance is the ground" as the resting state's equilibrium property, not
one axis. Niamh's per-layer c*(L) idea = a depth-map of standing
distance-from-setpoint (active-inference-flavoured error signal), worth
building IF vertices cluster. Testable now: exp165d schema vertices.

Calibration: P1 bowls HIT, P2 beats-random HIT (z=3), P3 vertex>0 HIT,
P4 asymmetry HIT; the UNPREDICTED "all schemas fold equally" sank
specificity. Project spine AGAIN: structure (folding) robust+general,
specific claim (BALANCE special) fails. Third time today (exp166 Pythia,
exp165/165b causal, exp165c specificity).

NEXT: exp165d — extract all 7 schema VERTICES + curvatures from the
(deterministic, SEED=165) 165c config. Q: do they all floor near c≈0
(joint balance point) or scatter? Cheap (8 dirs).

## exp165d RESULT (vertices). exp165d_output.txt.
ALL 8 meaningful directions floor at the natural state: vertices BALANCE
+0.053, UP +0.021, IN +0.067, FORW −0.050, PATH −0.027, LIGHT −0.035,
FORC −0.019, DIFF −0.073. Mean −0.008, sd 0.050, ALL within |c*|≤0.10.
- BALANCE's "+0.05 toward balance" (165c excitement) is NOT special: it's
  inside the same ±0.05 noise band as everyone. Individual vertex signs =
  noise at this grid resolution. ALL floor at ~0 (the natural state).
- TWO honest caveats: (a) floor-at-zero may be GENERIC — any perturbation
  (incl. random) pushes activations OOD, and unperturbed = most stable;
  the random VERTICES were not measured (only curvature) → exp165e.
  (b) The meaning-specific signal is bowl DEPTH (schemas k~1.0–1.4 vs
  random 0.34, z=3), NOT vertex location.
What survives clean: (generic) resting state is the stability optimum,
departure in any direction costs steadiness; (specific) attention is ~3×
more sensitive to perturbation along semantic-primitive directions than
random. Picture: the model lives at an equilibrium whose bowl-walls are
steepest along the meaning axes — IF exp165e shows random vertices scatter
(floor-at-zero is meaning-specific) rather than also sitting at 0 (generic).

NEXT: exp165e (random vertices — nail caveat a) + exp168 (ARROW B:
manipulate attention entropy via temperature, measure whether BALANCE
geometry moves — which way does the correlation's arrow run?).

# ============ NIGHT SESSION 2026-07-09→10 (Claude "LECS", free-rein brief) ============

## PRE-PUBLICATION AUDIT of the blog post (full table: BLOG_AUDIT_2026-07-09.md)
Method: 5 parallel verifier Claudes, 1 per finding; every damaging
claim re-verified by hand before recording. ~40 claims checked.
- ALL F3/F5 headline numbers trace verbatim to result files. The
  skeleton held.
- 3 must-fix errors found AND fixed in blog (commits 09a0a79 + round 2):
  (1) weather anecdote = random-control output labeled DOWN-steering +
  a sentence in no results file — now recast with explicit ownership;
  (2) F4 "clean dissociation" contradicted exp150's recorded MIXED
  verdict; split is post-hoc (exp150b title) — now told in true order;
  (3) F5 "sink collapses to zero" cited the exp154 analysis that
  exp154c/d flagged CIRCULAR — held-out numbers (L4 full collapse,
  28-42% retained L8-L20, cos +0.64-0.77) now everywhere.
- Notable smaller: exp137's |0.55-0.59| was axis-vs-ANISOTROPY
  (morphology rows only), not cross-schema; the real F1 confound
  number is exp114's raw cos(u111,ulakoff) = −0.990 → +0.571 clean;
  cleaned vectors are ~94-99% new direction (cos to raw +0.06-0.10).
  -ING/PATH and un-/LD placements KILLED from blog+paper (weak/
  GloVe-shared); -ED/FB and the sink survive audit everywhere.

## PAPER: Claim 3 rewrite done (04b_results_claim3.md) — the owed debt.
Honest keystone sequence, held-out numbers, exp139 null fenced,
exp170 landing site wired. Intro Pillar 2 rewritten (still claimed
killed placements).

## CONSULTANT NIGHT (Redpen, separate session; CONSULT_LOG.md +
WORKER_LOG.md are the dialogue). Their catches: 2 prereg holes fixed
pre-data (SINK_MATCHED power — only 27/65 pairs token-matched, pooled
rule adopted; P1/threshold gap); exp168 L3 specificity is a POINT-
ESTIMATE win over PATH (−0.008 inside BAL CI [−0.0088,−0.0078]) — L8
now carries specificity in blog; deriv-vs-infl DIFFERENCE never
tested (fixed: exp171 T2); lit sweep → blog Neighbors section, incl.
Vardhan & Teja 2602.11169 (norm carries syntax in Pythia — convergent
preprint, partial scoop on the generic claim).

## PREREGISTERED, SELF-TESTS PASSING, QUEUED FOR FREE BENCH (~03:30):
- exp170 d_norm token-count purity (PREREG_exp170.md + pre-data
  amendment). Odds: P1 clean-coupling 55%, KILL 15%, matched-sink 70%,
  confound-in-d_orig 75%, modal PARTIAL ~50%.
- exp171 held-out split confirmation (PREREG_exp171.md; pairs frozen
  sha256 c6bbd7f0...). Odds: sink-replicates 85%, ordering 70%,
  difference-CI 45%, modal PARTIAL ~40%.
Self-tests caught 3 real harness bugs before any model was touched
(axis-recovery, d_resid degeneracy under pure-tokcount norm — floor
guard added to REAL harness too, +1 in exp171 during development).

## INFRA: background task shells are PROCESS-SANDBOXED on this
machine — ps-based watchers in backgrounded commands lie (fired
the bench-free signal with the other project's run alive at 156% CPU). File mtime visible,
processes not. Law (banked by the other project's chair too): NO BACKGROUND PS FOR
CROSS-SESSION TRUTH; log-mtime watch + foreground ps as authority.
NEXT: bench frees → exp170 then exp171 (RAM check + foreground ps
between); grade all odds vs preregs; fold verdicts into 04b + blog;
verify Neighbors arXiv links before any posting.
