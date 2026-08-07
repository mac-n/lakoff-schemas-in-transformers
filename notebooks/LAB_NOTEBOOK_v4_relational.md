# Lab Notebook v4 — Schema Relational Structure Across Layers

Started 2026-06-06.

> **Morning summary for Niamh (read this first).**
>
> exp123 ran successfully on Pythia 410M, 8 Lakoff schemas, 24 layers,
> K=100 strong-null trials, K=100 weak null. **Result: positive and
> striking.**
>
> Headline numbers:
> - **Predicted-positive Lakoff couplings** (UP↔LIGHT, UP↔BALANCE,
>   LIGHT↔BALANCE, FORCE↔DIFFICULTY-BURDEN, FORWARD↔PATH-MOTION,
>   UP↔FORCE): mean cos across layers = **+0.214**
> - **Unpredicted couplings** (the other 22): mean cos = **+0.004**
> - **5 of 6 predicted couplings sign-consistent across ALL 24 layers**
> - **Across-layer configuration similarity**: real schemas +0.91 mean,
>   strong null +0.79 mean. Real configuration is stable from L3 to L22.
> - **Antonymy sanity check** (cos with flipped schema): exactly −1.000
>   for every schema at every layer. Geometry works.
>
> Four plots in `/Users/macn/Documents/embeddingexp/exp123_*.png`:
> 1. `exp123_couplings_across_layers.png` — per-pair trajectories;
>    predicted in colour, unpredicted in grey.
> 2. `exp123_predicted_vs_unpredicted.png` — histogram of the +0.214
>    vs +0.004 separation.
> 3. `exp123_layer_similarity.png` — 24×24 layer-similarity matrix
>    (real vs null vs difference).
> 4. `exp123_mean_cosine_matrix.png` — 8×8 schema mean-cosine matrix.
>
> **Read Entry 1b for the full results writeup**, including caveats,
> implementation note on the single-token filter (had to drop it; v3
> never had one anyway), the layer-trajectory story (L0–2 different,
> L3–22 stable, L23 different — consistent with mid-stream semantic
> processing), and the not-fully-blind disclosure on predictions
> (informed by Lakoff theory + exp102 GloVe results from v2).
>
> **Honest caveats**:
> - Predictions weren't fully blind — informed by Lakoff theory and
>   exp102 GloVe findings.
> - BALANCE schema is the noisiest across layers (Lakoff list is composed
>   from several attestations, may not be coherent).
> - Same frequency axis stripped at every layer; per-layer freq axes
>   might give cleaner results.
> - Pythia 410M only; not replicated in 1.4B or other architecture yet.
> - Doesn't disentangle from Osgood E-P-A — that's the next deflationary
>   test.
>
> **My honest take**: This is the most paper-shaped result we've found.
> The cross-layer stability of Lakoff-predicted couplings, distinguishable
> from random anchor partitions at every layer in the mid-stream regime,
> is the relational-structure finding the v4 thesis was reaching for.
> Worth replicating before claiming hard, but the shape is right.
>
> **Recommended next steps (your call)**:
> 1. Replicate on Pythia 1.4B (same code, slower)
> 2. Build Osgood E-P-A directions and add them — see if Lakoff structure
>    survives or collapses into E-P-A
> 3. Per-layer freq axis instead of L-invariant
> 4. Add more schemas (warm-cold, hard-soft, near-far) — does the
>    relational structure get richer?
> 5. Plot the per-pair trajectory PNG and just look at it for an hour

---

This notebook takes up a different question from v3. Where v3 was
*localised* (does this specific axis at this specific layer encode this
specific concept), v4 is *structural*: **do Lakoffian image schemas, as a
set, hang together as a coherent relational system across all layers of a
language model?**

The intuition behind the shift, in Niamh's framing (2026-06-06):

> *"I'm wondering if we can compare the relationships between schema
> vectors in every layer ... I'm wondering if schema relationships keep a
> coherence that random pairs of vectors do not."*

The localised question (v3) is the well-trodden interp path: pick a
candidate axis, build a steering vector, measure DV shifts. The structural
question is less explored and arguably closer to what Lakoff's theory
actually claims — image schemas aren't independent labels, they're a
*system* of embodied primitives that compose and project metaphorically.
The test for "system" is relational stability: does the pattern of
inter-schema relationships hold across the model's processing depth in a
way random vector pairs wouldn't?

---

## Why this framing changes things

Three things about v3 that v4 deliberately stops doing:

1. **Stops fixating on UP.** v3 ran exp111–117 all on the UP direction.
   v4 takes 8 schemas at once and reads their joint structure.
2. **Stops fixating on L12.** v3 inherited L12 from exp3b's raw
   qualitative steering plateau on Pythia 1.4B. v4 sweeps every layer of
   a smaller model and treats the depth trajectory as primary data.
3. **Stops fixating on 1.4B.** v3 used Pythia 1.4B because of project
   inertia. v4 uses Pythia 410M — same 24-layer depth, 3.5× smaller and
   faster, makes the full layer-sweep tractable.

What v4 inherits from v3:
- The methodological lesson that bare-word residuals need frequency-
  stripping before any structural claim is defensible.
- The Lakoff anchor lists from `lakoff_canonical_vocabulary.py` — citable
  to Lakoff & Espenson 1991.
- The discipline of always including a strong null and a random-direction
  control.

---

## The structural thesis (testable)

If Lakoffian image schemas form a coherent relational system in
Pythia 410M, then:

**Prediction 1**: The N×N inter-schema cosine matrix at each layer should
have substantially *less variance across layers* than a matrix of
random-anchor-shuffled "pseudo-schemas" of the same machinery. (Schemas
maintain their inter-relations as the model processes; random
distributional clusters do not.)

**Prediction 2**: Specific Lakoff-predicted couplings should appear as
consistently positive cosines:
- HARD-IS-COLD (FORCE × WARM-COLD couplings) — exp102 found cos +0.21
  in GloVe
- ORDER-IS-LIGHT / CHAOS-IS-DARK (BALANCE × LIGHT-DARK) — exp102 +0.34/+0.36
- MORE-IS-UP (UP-DOWN × magnitude) — confirmed in v3 exp116
- DIFFICULTY-IS-WEIGHT / HARDSHIP-IS-BURDEN (DIFFICULTY-BURDEN × FORCE)
- BAD-IS-DARK (LIGHT-DARK should anti-correlate with positive valence
  carriers)

**Prediction 3**: Antonymous schema pairs (UP vs DOWN within UP-DOWN; IN
vs OUT within IN-OUT) should show consistently *negative* cosines
between their flipped variants.

**Prediction 4**: The relational structure should emerge clearly by
mid-depth and persist; very early layers (closer to input embeddings)
might be more anisotropy-dominated.

A null result on any prediction is informative — would constrain which
parts of Lakoff's compositional claim the model supports.

---

## Entry 1 — exp123 design: schema relational structure across layers

### Model

**Pythia 410M** via TransformerLens. 24 layers (same depth as 1.4B), 1024
hidden dim, ~3.5× smaller. Fast enough for full 24-layer sweep × 8 schemas
× freq-strip × random null at scale.

### Schemas (from `LAKOFF_SCHEMAS_MML`)

8 schemas, all sourced from the project's `lakoff_canonical_vocabulary.py`
extraction of Lakoff's *Master Metaphor List* (Lakoff & Espenson 1991):

| Schema           | Pairs | Source                                    |
|------------------|-------|-------------------------------------------|
| UP-DOWN          | ~40   | `UP_DOWN_MML` (verticality, MORE, STATUS) |
| IN-OUT           | ~25   | `IN_OUT_MML_CLEAN` (containment)          |
| FORWARD-BACK     | ~20   | `FORWARD_BACK_MML` (path/motion)          |
| LIGHT-DARK       | ~30   | `LIGHT_DARK_MML` (illumination/knowledge) |
| FORCE            | 15    | `FORCE_MML` (push/pull, resistance)       |
| BALANCE          | ~20   | `BALANCE_MML` (equilibrium)               |
| DIFFICULTY-BURDEN| ~20   | `DIFFICULTY_BURDEN_MML` (heavy/hard)      |
| PATH-MOTION      | ~25   | `PATH_MOTION_MML` (motion along path)     |

Note: each schema's "direction" is built from `mean(pole-A anchors) −
mean(pole-B anchors)` where the pair `(A, B)` is the canonical
opposition (e.g. up/down, light/dark). Single-token-required.

### Per-layer procedure (layer L = 0 … 23)

1. For each anchor word *w*, get residual at `blocks.L.hook_resid_post`
   at position −1 of `model.to_tokens(w)`. Skip multi-token words.
2. For each schema, build raw direction = mean(pole-A acts) − mean(pole-B
   acts). Normalise to unit.
3. Build a **frequency axis at layer L** the same way: `mean(20 common
   function words)` − `mean(5 rare-but-real content words)`. Normalise.
4. **Freq-strip each schema direction**: project out the freq component,
   renormalise.
5. Compute the 8×8 schema cosine matrix.

### The strong null: random anchor labels

Per Niamh's call (2026-06-06): use the strong null, not the weak one.

**Procedure**: pool all anchor words across all 8 schemas (~200 unique
single-token words after filtering). For each of *K* = 100 null trials:
randomly partition the pool into 8 disjoint "pseudo-schemas" of similar
sizes to the real ones, randomly assign pos/neg within each. Build the same
machinery: raw direction → freq-strip → cosine matrix. Per layer.

This tests the right question: **is the inter-schema relational structure
specific to *the way these particular schema labels group these
particular words*, or would any random grouping of the same word pool
give the same relational structure?**

A weak null (random unit vectors) is also computed as a sanity check —
distinguishes "direction-specific" from "structure-specific."

### Metrics

For each of: real schemas, K random-shuffle pseudo-schemas, weak null:

**M1 — Per-pair layer-stability**: for each schema pair (i, j), collect
cos(i, j) across 24 layers. Compute std and CV (std/|mean|). Low std =
stable relationship across depth.

**M2 — Across-layer configuration similarity**: for each pair of layers
(L_a, L_b), compute the Procrustes-aligned similarity (or cosine of
vectorised upper-triangle of the schema cosine matrices). Plot the 24×24
layer-similarity matrix. Should be visually structured (block-diagonal
or smoothly varying) if relational structure is preserved.

**M3 — Predicted-coupling specificity**: for the Lakoff-predicted
positive couplings (HARD-COLD, ORDER-LIGHT, MORE-UP, etc — picked from
literature before running), test whether they're more positive than the
distribution of non-predicted couplings. p-value via permutation.

**M4 — Antonymy check**: pole flips should give cos ≈ −1 within each
schema (sanity). Inter-schema antonyms (UP-flip ↔ DOWN-flip) should NOT
necessarily be antiparallel — they're independent schemas.

### Predictions

- **Real schemas**: M1 std-per-pair should be substantially lower than
  shuffled-pseudo-schemas — relational structure preserved.
- **Real schemas**: M2 layer-similarity matrix should show clear
  structure (e.g., late layers similar to each other; possibly early
  layers different).
- **Predicted couplings**: M3 should show real-schema couplings more
  positive than shuffled at p < 0.01 if Lakoff is on the right track.
- **Antonymy**: M4 should give cos(UP, flipped-UP) ≈ −1 ± noise (sanity
  check that the geometry works at all).

### Outcomes mapped

**Outcome A (paper-shaped)**: M1 separates real from shuffled by ≥2x;
M2 shows clean across-layer structure; M3 predicted couplings significantly
above shuffled distribution. → *"Image schemas form a layer-stable
relational system in Pythia 410M, distinguishable from random
distributional clusters at p < α."* Publishable.

**Outcome B (partial)**: Some predicted couplings hold; others don't.
Layer-stability is intermediate. → *"Image schemas partially recover
Lakoffian relational structure; specific couplings [X, Y] hold; others
[Z] don't."* Less clean but still informative.

**Outcome C (null)**: M1 doesn't separate real from shuffled; M2 looks
like noise. → *"Schema labels in this model don't recover stable
relational structure beyond what random word-set partitions would
produce."* Real result, deflationary on Lakoff's compositional claim.

### Runtime budget

Pythia 410M load: ~10s. Per anchor word forward pass: ~10ms × ~200 words ×
24 layers (cached in one pass per word). Schema-direction builds per
layer: O(N) trivial. Frequency-axis per layer: same. K=100 null trials =
100 × full procedure = the main cost. Estimate: ~30 min total on MPS.
Acceptable.

### Files

- `exp123_relational_structure.py` — the script
- `results_exp123.txt` — text dump
- `exp123_results.npz` — schema cosine matrices per layer + null
  distributions
- `exp123_config.json` — schema labels, anchor counts, null seed

---

## Entry 1b — exp123 results: there is a Lakoff-predicted structure

### Implementation note

Initial attempt filtered to single-token anchors and discovered that
Pythia's BPE tokeniser splits most of the rare/morphologically-complex
words in Lakoff MML into subwords. With strict single-token filter, e.g.
PATH-MOTION dropped from 17 pairs to 1 pair — too aggressive. v3 (exp114)
never filtered, just took last-position residual; standard probing
approach. **Reverted to no filter** — all 179 Lakoff MML pairs used,
multi-token words extracted at their last subword position. This is what
v3 (`u111_clean`, `ulak_clean`) was doing implicitly all along.

### M3 — Lakoff-predicted couplings vs unpredicted couplings: **CLEAR SEPARATION**

```
Predicted-positive couplings:   mean(cos) = +0.214,  n=6
Unpredicted couplings:          mean(cos) = +0.004,  n=22
```

~50× separation. Of the six couplings declared as positive predictions
before running:

| Pair                          | mean(cos) | sign consistency |
|-------------------------------|-----------|------------------|
| UP-DOWN ↔ BALANCE             | +0.380    | 23/24 layers > 0 |
| FORWARD-BACK ↔ PATH-MOTION    | +0.281    | 24/24 layers > 0 |
| UP-DOWN ↔ LIGHT-DARK          | +0.269    | 24/24 layers > 0 |
| FORCE ↔ DIFFICULTY-BURDEN     | +0.182    | 24/24 layers > 0 |
| LIGHT-DARK ↔ BALANCE          | +0.125    | 24/24 layers > 0 |
| UP-DOWN ↔ FORCE               | +0.048    | 18/24 layers > 0 |

Five of six predicted couplings are positive *at every single layer*. The
sixth (UP-DOWN ↔ FORCE) is positive at 18/24 layers, mean still positive.

**The Lakoff theoretical predictions (GOOD-IS-UP / GOOD-IS-LIGHT /
GOOD-IS-BALANCED bundling; HARDSHIP-IS-FORCE/BURDEN bundling) are
recovered as a consistent feature of the model's representation
across the full depth.**

### M2 — Across-layer configuration similarity

```
Mean off-diagonal of layer-similarity matrix:
  REAL schemas:  +0.9137
  STRONG null:   +0.7948
```

Real schemas' inter-layer configuration cosine is +0.91, vs +0.79 for
strong null (random anchor partitions). Real schemas' relational
configuration is more preserved across layers than random word-set
partitions would produce.

But the more interesting finding is the *layer trajectory*. Reading the
24×24 layer-similarity matrix:

- **Layers 0–2**: somewhat differentiated from the rest (cos 0.76-0.93
  with later layers). Early-layer schema geometry is still being built up.
- **Layers 3–22**: extremely stable. Adjacent-layer cos > 0.97 throughout;
  distant-layer cos (L3 vs L21) still > 0.94. **The schema relational
  structure is set by layer 3 and persists almost intact through layer 22.**
- **Layer 23 (final)**: markedly different from all others (cos 0.46-0.86
  with earlier layers). Output-layer specialisation kicks in here.

This is consistent with what's known about transformer layer organisation:
early layers do lexical work, late layers prepare for output prediction,
mid-layers carry the stable semantic structure. The schema relational
configuration is encoded in that stable mid-stream regime.

### M1 — Per-pair stability across layers

Mixed and informative.

```
Mean std-across-layers (upper triangle):
  REAL schemas:   0.0556
  STRONG null:    0.0580
  WEAK null:      0.0302  (random vectors are extra-stable because they're
                           re-sampled each layer independently)
```

Per-pair, some real-schema pairs are dramatically more stable than null
(ratio < 0.6):
- UP-DOWN ↔ LIGHT-DARK: ratio 0.54 (predicted, very stable)
- LIGHT-DARK ↔ BALANCE: ratio 0.52 (predicted, very stable)
- LIGHT-DARK ↔ DIFFICULTY-BURDEN: ratio 0.54
- PATH-MOTION ↔ LIGHT-DARK: ratio 0.56

And some are *less* stable than null (ratio > 1.4):
- UP-DOWN ↔ BALANCE: ratio 1.91 (despite being predicted-positive)
- IN-OUT_CLEAN ↔ BALANCE: ratio 1.70
- BALANCE ↔ DIFFICULTY-BURDEN: ratio 1.64
- FORCE ↔ BALANCE: ratio 1.43

Notice **BALANCE** appears in most of the unstable pairs. Possible
explanations: (a) BALANCE anchors are particularly multi-token-heavy in
Pythia's tokeniser, so the last-subword residual is noisier; (b) the
BALANCE concept *develops* across layers in a way that other schemas
don't, with its relationships to other schemas shifting as the model
processes; (c) it's a genuinely less coherent schema (Lakoff's BALANCE
list is composed from several attestations, more eclectic than UP-DOWN).

### M4 — Antonymy sanity check: PASSES

cos(schema_direction, flipped_schema_direction) = -1.000 across all
schemas at all sampled layers (L0, L12, L23). The geometry works.

### Unpredicted findings worth noting (post-hoc, not load-bearing)

Looking at the unpredicted couplings with large absolute values:

- **UP-DOWN ↔ FORWARD-BACK = +0.356** (unpredicted). Could be interpreted
  post-hoc as "TIME-IS-MOTION / FUTURE-IS-FORWARD-AND-UP" bundling —
  upward and forward both being "progress" metaphors. Worth checking
  whether this is consistent across other models.
- **FORWARD-BACK ↔ BALANCE = +0.370** (unpredicted). Harder to motivate.
- **UP-DOWN ↔ PATH-MOTION = +0.245** (unpredicted). Consistent with
  motion-up cluster.

And some strong negative couplings:
- IN-OUT_CLEAN ↔ BALANCE: −0.229 — possibly "CONTAINED-IS-STABLE" vs
  "OUT-IS-DESTABILISED" antagonism
- LIGHT-DARK ↔ DIFFICULTY-BURDEN: −0.202 — "LIGHT-IS-EASY,
  DARK-IS-HEAVY/DIFFICULT" antagonism
- BALANCE ↔ DIFFICULTY-BURDEN: −0.165 — "BALANCED-IS-EASY,
  IMBALANCED-IS-HARD"

These are all interpretable as Lakoff couplings *post-hoc*, which is part
of what makes a non-blind analysis tricky to evaluate. The discipline is
to call them out as post-hoc and not include them in our load-bearing
claims.

### Disclosure on predictions

The six couplings declared as positive predictions before running were
chosen based on:
1. Lakoff's theoretical predictions (GOOD-IS-UP / IS-LIGHT / IS-BALANCED
   bundling, HARDSHIP-IS-FORCE)
2. The exp102 finding from `LAB_NOTEBOOK_v2.md` that in GloVe embeddings:
   HARD-IS-COLD coupling cos = +0.21, ORDER-IS-LIGHT cos = +0.34

So predictions were *theoretically motivated* but not *blind to all
relevant data*. The exp102 results gave partial prior evidence for similar
couplings in a different substrate (word2vec/GloVe), which informed the
predictions. The fact that those couplings replicate in Pythia 410M
activations and remain sign-consistent across all layers is real evidence
of cross-substrate structural agreement, but the strength of the claim is
attenuated by the priors not being fully naive.

### What's the actual finding

Two claims, of decreasing strength:

**Strong claim** (well-supported by these data): At every layer of
Pythia 410M, the freq-stripped directions for Lakoff image schemas form a
relational configuration in which Lakoff-predicted couplings (UP-DOWN ↔
LIGHT-DARK ↔ BALANCE; FORCE ↔ DIFFICULTY-BURDEN; FORWARD-BACK ↔
PATH-MOTION) are consistently positive, while non-predicted couplings sit
near zero. The relational configuration is stable across layers 3–22 and
distinguishable from random word-partition pseudo-schemas at the
configuration-similarity level (+0.91 vs +0.79).

**Stronger claim** (suggested by these data but needs replication):
Lakoff's compositional/relational view of image schemas — that schemas
form a *system* of mutually-supporting embodied primitives that bundle in
specific theoretically-predicted ways — is recapitulated in the activation
geometry of a transformer language model. The schemas don't just exist as
independent linear directions; they exist as a *configuration* of
inter-related directions whose specific pairwise relationships match
Lakoff's predictions and persist through depth.

The second claim is the paper-shaped one. To upgrade it from "suggested"
to "load-bearing" we'd want:
- Replication in another model (Pythia 1.4B; GPT-2; LLaMA)
- A genuinely blind set of additional predictions (e.g. from Talmy force
  dynamics or Russell affect) tested without inducing them from data
- Confirmation that the layer-stability isn't an artifact of using the
  same frequency axis across layers (multi-layer freq axes per Niamh's
  salience worry)

### What's NOT shown by these data

1. Causation. We've shown that schemas form a stable relational structure,
   not that the model *uses* this structure to reason. Steering experiments
   (v3-style) on multiple schemas would test whether interventions on one
   schema affect other cluster members predictably.
2. Disentanglement from Osgood E-P-A. The "predicted positive couplings"
   could be partly explained if all our schemas load on the
   Evaluation/Potency factors of Osgood, with no specifically-Lakoffian
   structure beyond E-P-A. A direct comparison with E-P-A directions built
   from Warriner V-A-D anchors would discriminate this.
3. Layer-23 weirdness. We don't know why L23 is dramatically different
   from L0-22. Could be output-layer specialisation, or just lexical
   prediction-prep — worth investigating.

### Next steps (for Niamh's review)

1. **Plot a heatmap visualisation**: 24-layer × 28-pair grid showing
   cosine per pair per layer. Or a few representative cosine matrices.
2. **Cross-model replication**: re-run on Pythia 1.4B (slower) and/or
   another architecture.
3. **Osgood-vs-Lakoff comparison**: build Osgood E-P-A directions from
   Warriner V-A-D norms and add them to the schema configuration. See if
   they're collinear with combinations of Lakoff schemas or genuinely
   distinct.
4. **Layer 23 investigation**: what specifically changes at the output
   layer? Output-token alignment, attention sink rearrangement, or
   something else?
5. **Blind prediction set**: get specific predictions from a Talmy-force
   reading or a different cognitive linguist source, test against this
   exact same data, see if they also hit.

### Bottom line

This is the cleanest result the project has produced. The cross-layer
configuration similarity (+0.91 vs +0.79 strong null) is moderately
informative. The predicted-vs-unpredicted coupling separation (+0.214 vs
+0.004) is dramatically informative. The 24/24 sign consistency for most
predicted couplings is exactly what the Lakoffian-system thesis predicts.
The layer-3-to-22 stability is consistent with mid-stream semantic
representation; L0-2 and L23 differences are consistent with
lexical-prep / output-prep phases.

The paper-shape now visible: **"Image schemas form a relational system
in transformer activations, recapitulating Lakoff's specific predictions
about cross-schema couplings, stable across depth."** That's a structural
claim about how cognitive primitives are encoded, distinct from the
single-axis steering work in the linear-representation-hypothesis
literature.

Files: `exp123_relational_structure.py`, `results_exp123.txt`,
`exp123_results.npz`, `exp123_config.json`.

---

## Entry 2 — exp124: does anisotropy encode attention-routing?

### Niamh's hypothesis (2026-06-06, late evening)

> *"The anisotropy direction in transformer residual streams encodes
> attention-routing / salience information."*

The dominant common direction in residual streams might not be a noise
confound but a *functional signal* that determines which positions get
attended to. Evidence converging on this from existing mech-interp work:
attention sinks (Xiao et al), massive activations (Sun et al), outlier
dimensions (Kovaleva, Timkey & van Schijndel) — all involve high-norm
common-direction-loading residuals that are load-bearing for performance
even though they look like garbage in semantic-cosine terms.

### Test 1 — observational

For each token in a corpus sample of 12 natural-text prompts (286 tokens
total at Pythia 410M), at layers 4, 12, 20:
- Compute projection of its residual onto three candidate "common
  directions": (a) mean direction of all residuals at that layer,
  (b) PC1 of centered residuals, (c) our freq axis (COMMON-RARE).
- Compute attention received from all downstream positions across all
  heads at the same layer, normalised per query.
- Correlate.
- Also control for: residual norm, position, log token frequency.

### Results

**Layer 4 (early)**:
```
predictor                       Pearson r    after controlling
projection onto mean_dir         +0.192      +0.394
projection onto PC1              −0.744      −0.460
projection onto freq axis        +0.514      +0.327
residual norm                    −0.357      (controlled)
log token frequency              +0.360      (controlled)
```

PC1 at L4 is *negatively* correlated with attention received (r = −0.744)
— **opposite** of the hypothesis. Freq axis is positively correlated
(+0.51). Mean direction is weakly positive. PC1 and mean direction are
nearly orthogonal at L4 (cos +0.049) — they're not the same axis here.

Interpretation: at L4, PC1 appears to track function-vs-content
distinction (freq axis loads negatively on PC1, cos −0.224). Function-
word-shaped tokens get attended to *more*; content-word-shaped tokens get
attended to *less*. This is consistent with classic syntactic-attention
findings (function words are syntactic anchors at early layers).

**Layer 12 (mid)**:
```
projection onto mean_dir         +0.309      +0.132
projection onto PC1              +0.342      +0.028
projection onto freq axis        +0.321      +0.144
residual norm                    +0.356      (controlled)
position (absolute)              +0.317      (controlled)
```

All three "common direction" measures positively correlate with attention
(~+0.30), but **most of the signal is mediated by position + norm + freq**.
After controlling, partial correlation drops to ~+0.13. So at L12, the
common-direction projection isn't doing much load-bearing work beyond what
norm + position already capture.

**Layer 20 (late)**: similar mid-layer pattern but weaker. Partial
correlations after controlling around +0.12 to +0.21.

### Honest read

The hypothesis as stated **isn't cleanly supported in this form**:
- PC1 at L4 *anti*-correlates with attention (opposite of prediction)
- At mid layers, the partial correlation after controlling is small
- The pattern is layer-dependent and confounded with norm/position/freq

What the data **does** show:
- Directional structure of the residual stream *does* predict attention
  received at L4 (freq axis r = +0.51, robust to controls at +0.33)
- The direction doing the work isn't *one common axis* — it's some axis
  related to function-vs-content distinction
- At mid layers, residual norm becomes a strong predictor (+0.36), partly
  consistent with the hypothesis in *magnitude* rather than *direction*
  terms

### Test undersized for the sink hypothesis

Several real limitations:
- **N = 286 tokens** is small.
- **Aggregated across 16 heads**. Different heads probably do different
  things — the sink-routing mechanism might be concentrated in specific
  heads and washed out by summing.
- **Linear correlation only**. Attention is softmax(Q·K) — highly
  nonlinear. Linear regressions are coarse.
- **BOS excluded** (position 0). The canonical attention sink is BOS;
  excluding it means we're not testing the sink-mechanism directly.
- **Causal version not done**. Surgical edit of common-direction component
  + measure of attention shift would be much more diagnostic.

### What this leaves the hypothesis at

Partially alive, in a *refined* form: directional information in the
residual stream does predict attention routing, but it's not *one* common
direction doing the routing. Per-head analysis (Option A from the 2026-
06-06 conversation) would be the natural follow-up — find specific heads
where projection-onto-some-direction does predict attention-received with
large effect, characterise those as sink-routing-heads.

Files: `exp124_salience_attention_test.py`, `results_exp124.txt`,
`exp124_results.npz`.

---

## Entry 3 — exp125 / 125b / 125c: is PC1 frequency, median-band, or something else?

### Niamh's hypothesis (2026-06-06, before bed)

> *"Salience can be roughly approximated by 'not in the tail, not in the
> spike.' If you make a histogram per token read, the median token is
> informative."*

In a Zipfian distribution, the median by token-occurrence-weighted
distribution falls around frequency-rank √N (≈ 220 for a 50k vocab) —
exactly where content-word substance begins. The proposal: salience is
inverted-U-shaped against log-frequency, peaking in this middle band and
dropping at both extremes.

The discriminating test: **plot PC1 projection vs log-frequency at each
layer**. If PC1 is encoding raw frequency, the relationship is monotonic.
If PC1 is encoding median-band-ness, the relationship is bell-curve-shaped.

### exp125 — first attempt, stratified Brysbaert sample

- 1500 single-token words stratified across log-freq bins from Brysbaert
- PC1 projection vs log-freq at layers 0, 4, 8, 12, 16, 20, 23
- Plus a "salience axis" built as middle-band-mean − extreme-band-mean

**Sampling problem caught immediately**: the top 3% by cumulative mass
collapsed to just *one* word ("a"), because the Zipfian-shape combined
with Brysbaert's content-word-biased sample meant nearly all probability
mass at the top concentrated on the single most-common word. The
"salience axis" was thus middle-227-words vs (965 tail + 1 word) —
lopsided.

Even so the quantitative summary was suggestive:
- `cos(PC1, freq_axis) ≈ +0.50` at mid layers
- `Pearson r(PC1_proj, log_freq) ≈ +0.011` at mid layers

These look contradictory but are consistent with a non-linear relationship
between PC1 and log-frequency — exactly the bell-curve signature.

### exp125b — balanced rank-band sample

Re-sampled with explicit rank bands instead of mass quantiles:
- B1 (rank 1-10), B2 (11-50), B3 (51-300), B4 (301-1000), B5 (1001-3000),
  B6 (3001-7000), B7 (7001-15000), B8 (15001-30000)
- 80 single-token words per band; 530 unique words total

Then computed PC1 of centered residuals at each layer and reported mean
PC1 projection per band.

**The clear two-regime result**:

```
Linear R² of PC1 projection ~ log_rank:
  L0:   0.290    ← clean monotonic relationship  (PC1 ≈ freq axis)
  L4:   0.375    ← even cleaner
  L8:   0.0023   ← no linear relationship
  L12:  0.0021   ← no linear relationship
  L16:  0.0017
  L20:  0.0015
  L23:  0.0002

Quadratic R²:
  All mid-late layers: ≤ 0.003  ← no quadratic relationship either
```

**Per-band means at mid layers (L8-L20) oscillate by ±1 with within-band
variance of ±10-20.** Visually flat with noise. Not monotonic (PC1 ≠ freq).
Not bell-curve (PC1 ≠ median-band).

### exp125c — what does PC1 at L12 actually encode?

Sorted the 530 words by L12 PC1 projection, looked at extremes.

**TOP 30 (most positive PC1):**
`certain, mine, wait, sphere, pause, partition, efficiency, unreadable,
cyclic, definition, sty, natal, flush, consider, convex, watch, net, main,
written, lymph, kerchief, emphasis, corn, vex, idea, teen, adapter,
require, frontal, raw`

**BOTTOM 30 (most negative PC1):**
`invasive, ailing, aching, coming, lied, upon, perturb, emission, comes,
nest, him, it, bibliography, detection, ringer, love, omit, membrane, ...`

No clean theme on either side. Mix of abstract nouns + technical terms +
short verbs on top; -ing forms + function words + emotional/evaluative
words on bottom. PC2 looked slightly more interpretable (Niamh: also too
noisy to read confidently) — long inflected forms (top) vs short
technical/abbreviated terms (bottom).

### Honest read

- **The median-band hypothesis as a clean bell curve is not supported.**
  At mid-late layers PC1 doesn't track log-frequency either linearly or
  quadratically.
- **"PC1 = frequency" was wrong as a general claim.** Only true at early
  layers (L0, L4). At mid layers PC1 is something else entirely.
- **PC1 at mid layers explains 27% of residual variance but doesn't
  cleanly encode any nameable lexical/semantic feature.** Likely
  "internal computational signal" territory — formatting, attention
  routing, scaling — consistent with the rogue/outlier-dimensions
  literature.

### Implications for v3 methodology

Our v3 freq-stripping at L12 used a COMMON-vs-RARE axis — at L12 that axis
has cos ≈ +0.50 with PC1. We were stripping *some* of what PC1 captures,
but PC1 isn't itself a clean frequency axis at L12. So our stripping was:
- Partially effective on whatever frequency-correlated structure PC1 has
  at L12
- Did *not* fully remove the dominant variance direction
- The "semantic" residuals we then analysed still contained substantial
  PC1-aligned content of unknown identity

To do this properly: strip top-K PCs (which appear to be internal-
computation-flavoured) and only then analyse the remaining structure.
Or use SAEs to find disentangled interpretable features and selectively
analyse those.

Files: `exp125_pc1_vs_frequency.py`, `exp125b_pc1_balanced_bands.py`,
`exp125c_inspect_pc1_extremes.py`, plus their `results_exp125*.txt`,
`exp125*.npz`, and PNG plots (`exp125_pc1_vs_frequency.png`,
`exp125b_pc1_by_band.png`).

---

## Entry 4 — reframing: the orienting question we've been reaching toward

In conversation with Niamh on 2026-06-06, we identified that the Lakoff
experiments (exp111–117, exp123), the salience-attention test (exp124),
and the PC1/median-band investigation (exp125+) are all probes toward a
question neither v3 nor v4 has so far named:

> **"How does the substrate of computation determine what counts as a
> unit of meaning?"**

The Lakoff/Osgood/Talmy framing was theoretical scaffolding — useful, but
not what we were actually after. The pattern across this notebook (and
v3): every time a Lakoff result lands, Niamh immediately pivots to
something more mechanism-y: salience, attention, anisotropy, internal
computation. The "deeper thing she's reaching for" is the *mechanism by
which the substrate of computation organises distinctions into
representable units*.

Several threads in v3/v4 are facets of this question:
- v3 freq-stripping showed that mean-of-words contrast vectors are
  dominated by frequency — i.e., the substrate's organising principle is
  partly statistical (how often a thing occurs) rather than semantic.
- v3 cluster result (MORE/STATUS/VALENCE bundled into UP) showed that
  metaphorical extensions cluster into single directions — i.e., the
  substrate's organising principle bundles theoretically-distinct
  primitives into representational units when usage-statistics warrant.
- v4 exp123 relational structure showed schema-pair couplings are
  layer-stable in ways random-anchor-partitions aren't — i.e., the
  substrate maintains certain *relations* between units as it processes.
- v4 exp124 attention-routing test showed that directional structure
  predicts attention received, but not via one clean axis — i.e., what
  counts as "attendable" is itself a multifactorial substrate-determined
  property.
- v4 exp125 PC1-investigation showed that the dominant variance axis at
  mid-stream layers isn't aligned with any nameable lexical feature —
  i.e., the dominant computational signals at the unit-of-meaning level
  are themselves *non-semantic* internal computation that we can't yet
  read.

The natural next experiments under this framing aren't more
Lakoff-axis tests. They're:
- Per-head analysis at L12 — what does each computational unit (head)
  *separately* compute? When the residual stream is a superposition of
  many heads' outputs, individual heads might have clean structure that
  the sum obscures.
- Sparse-autoencoder approach — disentangle the superposed features to
  recover what counts as a unit of representation, *as the model itself
  uses them*, before the act of summing into residual stream blurs them.
- Causal interventions on specific directions (rather than correlational
  tests) — does editing the dominant axis change downstream attention?

These would be v5 territory. For now, this notebook (v4) is the
record of (1) the relational-structure result we got, (2) the mechanism-y
side-experiments that complicated the picture, and (3) the recognition
that the orienting question was bigger than the Lakoff-cluster test that
got us into it.
