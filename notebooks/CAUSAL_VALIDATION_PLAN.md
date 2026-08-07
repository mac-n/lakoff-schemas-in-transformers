# Causal Validation & Cross-Domain Transfer — Experimental Design (v1)

Created 2026-05-28. This is the design for **Thread 6** (the load-bearing validation phase the handoffs keep pointing at). It synthesizes a web-Claude research-advisor conversation (2026-05-28) into a runnable plan, organized around one template and a strict minimal-paper scope.

**Provenance:** the per-experiment designs, the four-step template, the Potency/Goldilocks reframes, and the lit-positioning targets below come from that advisor chat. Niamh's framing throughout the project (epistemic discipline, "if they were primitives they'd separate," distrust of own carving) is the reason the scope stays tight.

---

## 0. The thesis this phase tests

The claim that makes the project matter — and the alternative reading that would kill it:

- **Interesting hypothesis:** the metaphor is a *manipulable causal direction in the residual stream*. Steering a **source-domain** vector (physical) moves **target-domain** language (abstract). That is cross-domain schema reuse — metaphor *reuse*, not mere co-occurrence — and it's the result no one has shown mechanistically.
- **Deadly alternative:** "we found a sentiment/evaluative direction and Lakoff handed us the word to label it." Under this reading the vector moves mood but does *nothing* to literal source-domain output.

The whole phase is built to force those two apart. The discriminating move is always: **build from source-only language, then show it moves a target-domain DV.**

The metaphor being *in the language* is old (Lakoff 1980). The metaphor being a *steering vector* is the new mechanistic claim.

---

## 1. Literature to position against (READ before writing a word of intro)

> Position against these from knowledge → unassailable. Position in ignorance of them → desk-reject.

**Steering / mechanistic:**
- Turner et al. — **ActAdd** (activation addition)
- Rimsky et al. — **Contrastive Activation Addition (CAA)**
- Tigges et al. — **linear sentiment directions** in LLMs

**Psych / cognitive-linguistics:**
- **Meier & Robinson** — metaphor and vertical position (good-is-up, affect-space)
- **Casasanto** — mental metaphor / body-specificity
- **Osgood (1957)** — semantic differential, the **E-P-A** factors (Evaluation, Potency, Activity), replicated across ~two dozen cultures
- **Talmy** — force dynamics (the tradition Lakoff's image schemas under-weighted; hardness lives here)
- **Williams & Bargh** — physical warmth ↔ interpersonal warmth shared representation

**Human norms (for external validation):**
- **Warriner et al. (2013)** — Valence/Arousal/**Dominance** norms (~14k words)
- **Brysbaert et al. (2014)** — **concreteness** norms (~40k words)

**Physiology (temperature framing only):**
- Thermoreceptors are receptor-distinct: **TRPM8** (cold), **TRPV3/4** (warm), **TRPV1** (noxious heat). There is no single bipolar "thermometer" channel — warm and cold are detected in parallel.

---

## 2. The four-step template (reuse for every primitive)

1. **Build from source-domain-only contrasts** — zero target-domain words in the construction.
2. **Confirm literal-domain control** — does steering move source-domain output?
3. **Show abstract-domain transfer** — does the *same* vector move target-domain language?
4. **Dose-response** — graded DV vs steering coefficient (a numeric curve, not a vibe).

**Order of proof: source in, target out.** If a vector built from purely spatial language nonetheless steers happiness, no reviewer can say "you just made a sentiment vector and renamed it."

This template turns "a result" into "a method." UP/height/happy is the first clean instance; heavy/weight/difficulty the second.

---

## 3. Minimal-paper core — 3–4 vectors, then STOP

### 3.1 UP/DOWN — positive control + mania kicker

The obvious case (HAPPY IS UP, Lakoff 1980) is the **control**, and its obviousness is the feature: it's what licenses belief when we then show the non-obvious heavy→difficult.

- **Construction:** build UP from spatial-only contrasts — `high/low, top/bottom, rise/fall, ceiling/floor, above/below, peak/valley, ascend/descend`. **Strip the affect-laden words currently in exp17** (`uplifting, soaring, elevating`) — they leak target-domain content into the source vector and ruin the proof order.
- **Literal control (the linchpin):** prompt `"His height in centimetres is ___"`, read the number, **plot mean height vs steering coefficient**. A metaphor vector producing a *graded numeric shift in a literal-domain output* is the figure that makes the paper.
  - *Fallback:* if small-Pythia free-gen produces garbage numbers, use **logit / probability mass on tall-vs-short tokens**. Measure only in the steering regime where output is still coherent (the mania regime is too degraded to read).
- **Abstract transfer:** same vector → more positive affect.
- **Mania kicker:** over-amplify → the *pathological-excess syndrome* the vertical-elevation cluster predicts: elevated mood + grandiosity (HIGH status) + pressured output (MORE IS UP) + increased activity (ACTIVE IS UP). The test that upgrades this from "suggestive" to "load-bearing": does one intervention **simultaneously** shift valence + verticality + quantity + dominance? If yes → vertical-elevation *superordinate cluster*, not a sentiment scalar.
- **Status:** steering infra exists (`exp3`, `exp17`; 1.4B responded). Need: de-affected UP anchors, height-completion DV, dose-response sweep.

### 3.2 HARDNESS / heavy → difficulty — the single load-bearing novel test

This is **the** experiment that decides whether the thesis is true or trivial.

- Steer the **physical** vector (`heavy`/`hard`) and measure whether the model rates **exams / tasks / problems** as *harder* (difficulty domain). Source = physical weight/hardness; target = difficulty.
- Cross-domain transfer here = metaphor reuse, the thing no one has shown. (The Goldilocks/Potency/wire-mother theory is **not** this experiment — see §5. This is the validity check; keep it, drop the theory.)

### 3.3 WARM/COLD → affect

- Build from physical-temperature-only contrasts; measure transfer in how the model **describes people and relationships** (warm person, warm welcome), **not thermostats**. DV must be target-domain (interpersonal), or it isn't a transfer test.
- For the minimal paper, warm-vs-cold with an interpersonal-warmth DV is enough. The non-bipolarity of temperature is parked in §5.3.

---

## 4. External validation — embedding-space, cheap, **do this first**

These kill the "artifact of my carving" and "artifact of word2vec" objections without any steering. The advisor's explicit ordering: **Warriner-Dominance correlation first, today — it's an afternoon, and it either dissolves the worry or tells you something's off.**

1. **soft–hard vs Warriner DOMINANCE** — Dominance ≈ Osgood Potency ≈ hardness. Strong correlation = decisive external validation, kills "artifact of my filtered word set" in one shot.
2. **soft–hard vs Brysbaert CONCRETENESS** — rule out that we've just recovered a concrete/abstract axis wearing a hardness mask.
3. **Cross-architecture replication** — reproduce the decomposition in GloVe / fastText / an LLM embedding space. Survival kills "artifact of word2vec."
4. **Four-axis bake-off** — `soft–hard` vs `strong–weak` vs `heavy–light` vs `dominant–submissive`. If interchangeable (near-rotations of one subspace) → write it up as **Osgood Potency**, not "hardness." If soft–hard is genuinely *sharper* → that's the real, surprising claim; defend it explicitly.
5. **Residual test** — regress `strong–weak` + `heavy–light` out of `soft–hard`. Is there residual variance? If yes (the *hard facts / hard science / hard data* sense — epistemic, unyielding-to-wish), **that residual is the actual discovery** and is worth isolating. (Genuinely uncertain whether hardness retains independent variance after removing Potency-proper.)

> **RESULT — exp99 (2026-05-29).** The advisor's "hardness IS Osgood Potency" prediction is **not supported**; the cheap check did its job.
> - **Hardness is a weak Dominance handle.** HARDNESS(exp90 anchors) × Warriner Dominance: **r = 0.21**, vs STRENGTH × Dominance **r = 0.35** and the DOMINANCE axis itself × Dominance **r = 0.19**. If Dominance ≈ Potency, *strength* is the better handle; hardness is weaker.
> - **Concrete/abstract-mask objection KILLED.** HARDNESS × Brysbaert concreteness: **r ≈ 0.03**. Not concreteness in disguise.
> - **Not one subspace.** Bake-off cosines: HARDNESS–STRENGTH **0.44** (moderate), HARDNESS–DOMINANCE **0.33**, HARDNESS–WEIGHT **0.12**, STRENGTH–WEIGHT **−0.06**. Not the >0.6 "near-rotation" pattern Potency would predict.
> - **78% of hardness is independent** of span(strength, weight). So hardness has a *partial* Potency component but is mostly its own axis — "overlapping but distinct," not reducible.
> - **Two methodological cautions surfaced.** (a) **"firm" polysemy** — the hard-pole residual is dominated by *management / firm / company / government / plans*, i.e. business-*firm* leaked in via the exp90 anchor `("firm","mushy")`/`("flimsy","solid")`. (b) **Anchor sensitivity** — exp90 (r=0.21) vs a broader soft-anchor set (r=0.10) differ ~2× and cos only 0.74; broad soft anchors pull in a food-texture register (creamy/savory/silky).
> - **Implications:** the "Potency / E-P-A bridge" big paper (§5.1) is **off the table**. The residual is NOT the predicted epistemic "hard facts" sense. **Next:** rebuild HARDNESS without the polysemous `firm`/`solid` anchors and re-run. Results in `results_exp99.txt`.
>
> **RESULT — exp99 rebuild (firm/solid removed).** The firm artifact is confirmed and removable, and the picture sharpens *against* Potency:
> - **firm→company artifact confirmed.** Removing `firm`/`solid`, the management/company/government cluster **vanishes** from the residual; the hard-pole residual resolves into a clean, coherent **CONSTRAINT/RIGIDITY** cluster: *rigid, strict, imposed, rules, regulations, restrictions, laws, discipline, stringent, constraints, imposing, break/escape*. So hardness metaphorically extends into **imposed rules / rigidity / constraint** — a force-dynamic (Talmy) sense, NOT the epistemic "hard facts" the advisor predicted, NOT Potency.
> - **Hardness is even *more* distinct once cleaned.** Clean axis: Dominance r drops 0.21→**0.125**; cos with STRENGTH drops 0.44→**0.22**; residual rises to **91.6% independent** of strength+weight. The 0.44 hardness–strength cosine was itself partly the firm/solid contamination.
> - **Concreteness mask dead across all 3 variants** (r ≈ 0.00–0.03).
> - **Remaining caveat:** the *soft* pole is robustly food/sensory in GloVe (savory/creamy/juicy) regardless of anchors — soft-foods dominate "soft." So the axis is somewhat asymmetric (abstract-constraint hard pole vs concrete-sensory soft pole); worth a check with abstract soft anchors (yielding/lenient/flexible).
> - **LOAD-BEARING follow-up:** re-run the exp90 greedy coverage with the *clean* hardness axis. If clean-hardness still hits ~21% / is still picked first, the headline finding survives the polysemy. If exp90's coverage was inflated by firm/solid institutional vocabulary, the +21pp shrinks. This is the real robustness test for the original headline.
>
> **RESULT — exp100 (PC1 / anisotropy check; Niamh's catch).** exp90 mean-centered but **did NOT remove PC1**, and PC1 turns out to be load-bearing for the hardness number:
> - **Reproduced exp90:** HARDNESS_exp90 coverage (mean|cos| on the 86-word cognitive sample, PC1 present) = **0.212** ✓ (matches the "+21pp").
> - **Hardness is heavily aligned with the GLOBAL anisotropy direction:** `cos(HARDNESS_exp90, PC1_global) = −0.40` (clean version −0.30). The difference-vector construction did NOT cancel PC1 here, because hard-pole vs soft-pole words differ systematically along the dominant anisotropy/frequency axis.
> - **Remove global PC1 (ABTT k=1) and coverage HALVES:** 0.212 → **0.099** (variance fraction 0.058 → 0.015, ~26% of original). So **~half of the "+21pp" was hardness riding on the global anisotropy direction**, not semantic content. The global PC1 direction *by itself* covers the cognitive sample at mean|cos| = 0.40 — better than hardness does.
> - **NOT the coherence/sense direction.** `cos(HARDNESS, PC1_local) = −0.05` (orthogonal); removing the cognitive-sample PC1 (the exp97 coherence/sense direction) does NOT reduce coverage (0.212 → 0.232). So this is the *global frequency/register anisotropy* leak, distinct from the coherence/sense gradient.
> - **Implication for the headline:** "HARDNESS is the dominant cognitive axis (+21pp)" is **inflated by a frequency confound** (see exp101 below for the precise diagnosis — NOT a wholesale artifact). Deflated of frequency, hardness coverage ≈ 0.10. Combined with exp99 (near-zero human-norm correlations once cleaned), the HARDNESS headline (WRITEUP_v4 candidate-novelty #7) needs softening.
> - **THE test now:** re-run exp90's full greedy selection in an **ABTT'd space** (remove global PC1, maybe top-k). If hardness is no longer picked first once frequency is gone, the headline finding does not survive. Results in `results_exp100.txt`.
>
> **RESULT — exp101 (what IS global PC1? — Niamh's challenge to the "artifact" framing).** Don't assume PC1 is nuisance; check it.
> - **PC1 is the frequency axis.** PC1(+) pole = rare noise tokens (`afp03, hahb, eupithecia, bulbophyllum`); PC1(−) pole = commonest function words (`the, but, this, and`). `spearman(PC1, frequency rank) = +0.87`. It is the classic Mu-et-al anisotropy common-component.
> - **Directionality (which end ↔ which end):** every hardness anchor sits on the frequent (−) side, but **hard words are far more frequent than soft words** — `hard −0.72, firm −0.57, solid −0.54` vs `mushy −0.22, squishy −0.18, pliable −0.16`. The vector `hard−soft` thus points rare→common and loads −0.40 on PC1.
> - **Corrected interpretation:** NOT "hardness is anisotropy" (cos 0.40, not 0.9). The real issue is an **anchor-frequency imbalance** — exp90's soft anchors (`mushy/squishy/pliable`) are rarer than its hard anchors (`hard/firm/solid`), injecting a frequency component that pads coverage. The +21pp is *inflated by* a frequency confound, not *explained away* by it; the semantic contrast survives PC1 removal at ~0.10.
> - **Fix / real test:** rebuild hardness with **frequency-matched anchors**, AND re-run greedy coverage in **ABTT space** (frequency removed for ALL candidate axes, so the comparison is fair — every Roget/cognitive axis may carry its own frequency loading). If hardness still leads with frequency controlled, the finding survives in honest, weakened form. Results in `results_exp101.txt`.
>
> **RESULT — exp102 (Niamh's clean redesign: single-word contrasts, GloVe + fastText, frequency stripped).** Seven two-word contrasts (soft→hard, warm→cold, up→down, forward→back, past→future, chaos→order, darkness→light), ABTT top-3.
> - **The redesign works.** Single-word `soft→hard` frequency loading is only **−0.19** (GloVe) vs the multi-anchor axis's −0.40. The confound was largely the rare soft-anchors (mushy/squishy/pliable); plain `soft`/`hard` halves it. ✓ Niamh's instinct.
> - **fastText carries no frequency in its top PCs** (max |spearman(PC,freq-rank)| = 0.25, vs GloVe PC1 = 0.93). So the frequency confound is substantially a *GloVe* property; fastText is natively cleaner. (Caveat: ABTT-3 therefore strips frequency in GloVe but generic top-variance in fastText — not identical operations.)
> - **HARDNESS is NOT special once frequency is gone.** Cognitive-sample coverage (freq-stripped, mean|cos|): GloVe `chaos→order 0.113 > darkness→light 0.094 > soft→hard 0.089 > …`; fastText same ordering (`chaos→order 0.103` tops). soft→hard is mid-pack in BOTH substrates. The "+21pp dominant HARDNESS" headline does not survive frequency control — it drops to ~0.09 and is beaten by chaos→order and darkness→light.
> - **Robust cross-axis couplings (replicate in both substrates):** `chaos→order ↔ darkness→light` = **0.35 / 0.36** (ORDER-IS-LIGHT, CHAOS-IS-DARK); `soft→hard ↔ warm→cold` = **0.21 / 0.23** (HARD-IS-COLD). These are credible structural metaphors, not noise.
> - **Metaphors recovered in pole words:** hard→*difficult/struggle/perseverance*; warm→*friendly/gracious/welcoming*; chaos→*anarchy/mayhem*; darkness→*despair/madness/abyss*. The physical→abstract transfer is visible. BUT polysemy contaminates several poles (order=*command/request*, light=*lightweight/boxing weight-class*, up=*set up*).
> - **Only moderate cross-substrate agreement** (spearman of word-projections, n=42k): chaos→order 0.54, warm→cold 0.46, darkness→light 0.46, forward→back 0.44, soft→hard 0.41, past→future 0.37, **up→down 0.11**. So "substrate-invariant" is an overclaim; "moderately substrate-consistent" is honest. **up→down is the weakest** (noisy poles, lowest agreement) — ironic for the most canonical Lakoff schema; the culprit is `up` polysemy (set-up / up-to). Flag for the planned UP/DOWN steering work. Results in `results_exp102.txt`.
>
> **RESULT — exp103 (the right metric: WHOLE-vocabulary directional variance, PC1 stripped — Niamh's correction to the circular curated-sample coverage).** Replaces the hand-picked 85-word "coverage" with: fraction of the entire space's directional variance along each axis (`u^T M u`, M = mean of outer products of all PC1-stripped unit vectors), calibrated vs a random axis and the PC spectrum.
> - **In GloVe (PC1 = frequency, rho 0.89, stripped): every contrast is a MINOR direction.** All seven explain **0.0041–0.0044** varfrac = **1.2–1.3× random** (random baseline 0.0033). They behave like the **~PC76–PC112 of 299** — deep in the flat tail. The differences among them are within that flat tail (not meaningful). The curated-sample ranking ("chaos→order covers most") **vanishes** — over the whole space all seven are essentially equal and minor.
> - **GloVe is nearly directionally isotropic after frequency removal:** even its top PC explains only 1.6% of directional variance. There is no small set of dominant semantic axes — cognitive structure is spread thin across many tiny directions. This supports the project's own "primitives are non-unique" recalibration, more strongly than before.
> - **HARDNESS-as-dominant is now fully retired.** Whole-space, frequency-free, it's a minor direction (~1.3× random), indistinguishable from up→down or warm→cold.
> - **fastText caveat:** there PC1 is NOT frequency (rho 0.06), so stripping PC1 removed top *semantic* variance, not the confound. Its rankings differ (forward→back 2.5×, up→down 2.1× biggest; warm→cold 1.0× = random) — i.e. "which axis is biggest" is **not robust across substrates**.
> - **What survives as robust structure:** the cross-axis couplings replicate in both — `chaos→order ↔ darkness→light` = 0.34/0.29, `soft→hard ↔ warm→cold` = 0.21/0.15. Those metaphor couplings, plus the pole-word metaphor recovery (exp102), are the durable positive findings. Results in `results_exp103.txt`.

---

## 5. Parked — second paper (future-work sentence only; do NOT scope-creep)

> Reviewers attack surface area. A tight result with a small claim survives; a sprawling one with a big claim gets picked apart. Everything below waits.

### 5.1 The Potency / E-P-A bridge
HARDNESS-at-~20%-and-second-largest is the **Osgood Potency signature** (Evaluation > Potency > Activity). The bigger paper: **Lakoff image schemas and Osgood's E-P-A are partly the same object**, and embeddings let you measure the overlap — GOOD/HAPPY-IS-UP lands on Evaluation; hardness lands on Potency. Lakoff "missed" hardness because his tradition under-weighted Talmy force dynamics, and hardness is force-dynamic (resistance to deformation under force).
- **Caution — primitive ≠ generative.** Variance-explained tracks how *productively a contrast metaphorizes* across domains, not developmental/somatic primacy (word2vec is adult text about adult concerns). Hardness may dominate because it's the most *generative* source domain, not the most *primitive*. These can coincide but must be argued, not assumed.
- **Fork inside hardness:** tactile hardness (modality-specific) vs resistance-to-force (amodal, force-dynamic = Potency). Pick, or show they're the same.

### 5.2 Monotone vs Goldilocks primitives (the candidate general law)
Some axes are roughly **monotone** ("more is more": size, brightness). Others are **Goldilocks/homeostatic** — an optimum, both extremes aversive (temperature, arousal, speed/proxemics, fullness) — and they metaphorize **non-bipolarly**. The **mania result is the first instance**: over-amplified UP went bad ⇒ even UP has a ceiling.
- **Probe:** is `soft–hard` actually Goldilocks too? Good pole may be **firm** (in the middle): too soft = weak/spineless/pushover; too hard = rigid/brittle/cruel. Test whether `firm/resilient` sits as a **positive-valence midpoint** rather than on the line between poles — same geometry test as warm-between-cold-and-hot. If both break bipolarity the same way → **general law**, and mania becomes its third instance rather than a curiosity.

### 5.3 Temperature split (physiology-grounded, stronger than metaphor alone)
Not bipolar. **warm = good anchor**; two qualitatively distinct departures: **cold** = withdrawal/absence/death (clusters with *hard*: cold-hearted / hard-hearted) and **hot** = excess/anger/danger/sex (heated, hot-headed). Receptor-distinct (§1). Williams & Bargh give the social-cognition end (warm cup → rate strangers warmer).

### 5.4 wire-mother / attachment-tactile material — explicitly parked.

---

## 6. Scope discipline (the meta-rule)

The clean paper writes itself:
1. **UP/happy** as the validated anchor (+ **mania** as the kicker that says "real dose-responsive mechanism, not a parlor trick"),
2. then **heavy → difficulty** and **warm/cold → affect** as the same method revealing things people *don't* already know are in there.

3–4 vectors, cross-domain DVs, dose-response, shown crisply. **Stop there.** Point at the gold. Bank the result — the seam isn't going anywhere, but unpublished gold doesn't count. The rabbit holes (§5) are all still there once the minimal paper is banked.

---

## 7. What we have vs what we need

**Have:**
- Pythia steering infra — TransformerLens `HookedTransformer`, residual-stream injection at all layers, 70m–2.8B. **1.4B responds to UP-steering generatively** (`exp3`, `exp17`).
- HARDNESS axis construction (built inline in the greedy work, `exp90`).
- GloVe-300 embedding bases + full anchor vocabularies (`project_axis_vocabulary.py`).

**Need:**
- De-affected spatial-only UP anchors (drop `uplifting/soaring/elevating`).
- Height-completion + dose-response harness (numeric DV with logit-mass fallback).
- **Warriner et al. VAD norms** + **Brysbaert concreteness norms** CSVs (free download) — not present locally; exp28 used hand-curated proxies, not the real norms.
- Difficulty-domain DV probes (exam/task/problem framing) for the heavy→difficulty test.
- fastText for cross-architecture replication of the soft–hard decomposition.

---

## 8. Suggested run order

1. **soft–hard × Warriner-Dominance correlation** (embedding-space, an afternoon) — dissolves-or-flags the Potency worry before any steering. *Do first.*
2. **Four-axis Potency bake-off + concreteness check + residual test** (embedding-space) — decides whether the framing is "I found hardness" or "Potency, measured."
3. **UP/DOWN clean rebuild → height linchpin → affect transfer → dose-response** (steering) — the positive control; converts result into method.
4. **heavy → difficulty** (steering) — the load-bearing novel transfer.
5. **warm/cold → interpersonal affect** (steering) — third vector, same method.

Then write, positioned against §1. Park §5.

---

# Part B — Primitives as OPERATIONS (a transformation algebra), 2026-05-29

A separate, more ambitious program that crystallized after the exp99–103 deflation. Recording it here so it doesn't evaporate; it is **more complex than the steering work** and should be gated behind the same discipline that just paid off.

## B.0 The reframe

The whole exp99–103 arc deflated because we were describing **points** (directions of variance, contrast vectors) when Niamh's actual intuition is about **dynamics**: a primitive is an **operation** `T: embedding → embedding` — a discrete, composable, *invertible* transformation that carries one embedding to another. "PCA felt wrong" because PCA decomposes where the cloud sits; it structurally cannot see the moves. The right maths is **group / representation / Lie theory**, not dimensionality reduction (web-Claude framing, 2026-05-29).

## B.1 The ladder (which the first experiment must locate empirically)

- **translation** — `b ≈ a + t`. The `king − man + woman ≈ queen` result and our steering vectors are both this. (TransE.)
- **linear map** — `b ≈ a·W`, W a matrix. (TransR / RESCAL; needed because translation alone fails for many relations.)
- **bilinear / higher order** — if linear is insufficient.
- **discrete generators + continuous parameter** = a **Lie group**: the discrete primitives are the Lie-algebra basis (generators); "intensify by 30%" is the continuous one-parameter subgroup. Resolves the "discrete ops, continuous space" tension.
- **Higgins et al. (2018)** already formalize Niamh's exact claim: a disentangled representation = a group action factorizing into independent subgroup actions; the primitives *are* the generators.

## B.2 The insight that helps (invertibility selects the clean cases)

TransE was beaten on knowledge-graph **relations** (`capital-of`, `born-in`) because those are **many-to-one** → no inverse → never group elements. Niamh's **operations** (pluralize, negate, intensify, past-tense) are **bijective** → invertible → exactly what *can* form a group. So the KG-literature pessimism doesn't transfer; invertibility **predicts the dividing line** between transforms that will be clean (operations) and messy (many-to-one relations). Sharper hypothesis than "operations are linear."

## B.3 The traps it inherits (this framing is MORE confound-prone, not less)

1. **A 300×300 matrix has 90k params** — it fits any few-dozen pairs perfectly and means nothing. The fit is free; **held-out generalization + composition are the only real tests.** (Need n_train ≫ d, or regularize, or work in a reduced subspace.)
2. **The Linzen (2016) "exclude the inputs" artifact** — `king−man+woman≈queen` is partly an artifact of excluding the input words from the answer search; don't exclude and the top hit is often `king`. Same species as "what are you secretly leaving out of the space." Report retrieval **including vs excluding** the source, always.
3. **Frequency leak** — plural forms are rarer than singulars; a "plural operation" can be faked by a frequency shift. **Fit in the PC1-stripped space** and compare to raw (the exp100/103 control).

## B.4 Minimal first experiment — `exp104_operations_plural.py` (BUILT, ready to run)

One clean **invertible** operation: `singular → plural`. Train a linear map on a large `(w, w+'s')` set harvested from vocab (so n_train ≫ 300); test on the clean Google-analogy `gram8-plural` noun pairs (held out). Four predictors — **identity** (strong baseline: cat ≈ cats), **translation** (TransE), **linear map** (lstsq), **orthogonal Procrustes** (rotation) — scored by mean-cosine-to-target and top-1 retrieval (incl vs excl source, plus how often NN = source). Run in **raw and PC1-stripped** space.

**Falsifiers:** (a) if nothing beats identity on *exclusive* retrieval, plural isn't linearly encoded; (b) if it collapses when PC1 is stripped, it was a frequency shift; (c) if linear map ≯ translation, plural is a pure offset (TransE suffices, no matrix needed).

> **RESULT — exp104 (n_train = 18,400 ≫ d; 37 clean test pairs).** The program clears its first bar.
> - **Plural is a LINEAR operation, and it needs a MATRIX, not a translation.** Mean-cos to target: identity 0.729, **translation 0.726 (≈ identity — TransE adds nothing)**, **linear map 0.793**, procrustes 0.778. The linear map is the only predictor that relocates off the source: inclusive top-1 retrieval **30%** (vs 0% for identity/translation) and NN=source drops to 70% (vs 100%). So plural sits on the **linear-map rung, not the translation rung** — consistent with the KG literature (TransE insufficient, TransR-style matrices needed). Answers falsifier (c): matrix > translation.
> - **NOT a frequency artifact.** PC1-stripped, the same pattern holds: linear map 0.737 vs identity 0.660, 30% incl, survives. Answers falsifier (b): survives.
> - **The Linzen artifact, demonstrated live:** identity and translation have **NN=source = 100%** → inclusive top-1 = 0% (the prediction just sits on the input word). Exclude the source and identity jumps to **84%** (a singular's nearest non-self word is usually its own plural). So *both* retrieval variants are confounded in opposite directions; the honest signal is the *difference between predictors*, where the linear map cleanly separates.
> - **Honest caveats:** only 37 clean test pairs (expand it); the `+s` train set conflates noun-plural with verb-3sg (it's the "+s morphology" operation broadly); 30% off-source relocation is modest — plural is a *small* operation (singular≈plural already), so "is the small displacement linearly predictable?" → yes, via a matrix, partially. Results in `results_exp104.txt`.
> - **Verdict:** enough to proceed to the real payoff (B.5): composition and commutativity, where group structure either appears or doesn't.

## B.5 Roadmap (gated behind B.4 surviving)

- **Generalize the harvest** to `+ed` (past), `+er` (comparative) — large vocab-derived sets so the maps are well-posed; clean analogy sections as held-out tests. Data already downloaded: `norms/questions-words.txt` (gram2-opposite, gram3-comparative, gram7-past-tense, gram8-plural, …).
- **Composition** — does `A∘B` (matrix product) land on the embedding of the doubly-transformed concept? Composition matching semantics = proof of a genuine operation *algebra*, not seven regressions.
- **Commutativity** — does `AB = BA` on held-out pairs? If operations commute they share an eigenbasis → disentanglement *for free* (a natural coordinate system where each primitive scales its own axes). If not → genuinely non-abelian group structure, itself a finding about the shape of meaning.
- **Unsupervised generator discovery** — sparse dictionary of operator matrices reconstructing observed transitions; cluster fitted transition-matrices; how few generators suffice?
- **Lit anchors:** Mikolov analogies; Linzen 2016 (analogy-eval critique); TransE/TransR/RESCAL; orthogonal Procrustes / cross-lingual alignment (Smith, Artetxe, Conneau/MUSE); Higgins et al. 2018 (group-theoretic disentanglement).

**Discipline reminder:** start with plurals beating the identity map in the frequency-stripped space. If even that doesn't survive, stop — don't excavate. (The lesson of exp99–103.)

> **RESULT — exp105 (composition & commutativity; `exp105_composition_commutativity.py`).** Mixed, and it caught a real bug.
> - **Conditioning bug (caught + fixed).** v1 fit a full 300×300 map; it was wildly ill-conditioned (inverse error `‖W_f W_b − I‖ ≈ 1e4`), which made the commutator a meaningless `0.000`. Not reported as a result. Refit in the top-100 semantic subspace (frequency PC dropped) → well-conditioned. **Lesson: unregularized full-space operation matrices are garbage for algebra; the prediction in exp104 survives, the algebra needs regularization/subspace.**
> - **Composition (Part 2, ground-truth gender×number, leave-one-out, 26 quads): partial success.** Single ops retrieve the real word well — number (sg→pl) **77%**, gender (masc→fem) **69%**. Stacked, `masc_sg + gender + number → fem_pl` lands at **cos 0.48, 35% exact retrieval** of the real doubly-inflected word (queens, duchesses, lionesses…). So the algebra **approximately corresponds to semantics** — A∘B lands in the right region — but errors **compound under stacking** (35% vs ~70%), and rare composed targets are hard to retrieve exactly.
> - **Commutativity (Part 1): inconclusive.** In the top-100 subspace the `+s/+ed/+ing` maps are **weak** (held-out cos barely beats identity, e.g. +s 0.426→0.466) and **not cleanly invertible** (inverse err ~0.8). Their pairwise commutators (0.015–0.019) sit ~8× below the random-matrix null (0.141) — *suggesting* shared structure — BUT this is confounded: `+s/+ed/+ing` are all near-variants of one "suffixation" operation, so low commutators may just mean "they're the same operation," not "independent operations that commute."
> - **Methodological finding:** morphological operations act mostly in the **low-variance** directions — restricting to the top-100 PCs throws away most of the effect (exp104's full-space 0.66→0.74 vs subspace 0.43→0.47). So the conditioning/subspace choice is load-bearing.
> - **To make commutativity conclusive (next):** (a) **ridge-regularized full-space maps** (keep the fine-grained action, condition the inverse) instead of hard subspace truncation; (b) genuinely **distinct operation types** — gender, negation/opposite (test as an involution, `T²≈I`), comparative — not three suffixations. Only then does "do they commute / share an eigenbasis" mean something. Results in `results_exp105.txt`.

> **RESULT — exp106 (decompose the operator; `exp106_decompose_plural_operation.py`).** Niamh: "can I see the matrix? what if we decompose it?" Matrix saved (`results_exp106_W_plural_raw.{npy,txt}`, `results_exp106_W_plural_ridge.npy`).
> - **Second conditioning subtlety, caught:** ridge shrinks `W`→0, but it should shrink the *action* `E=W−I`→0 (the op should do *nothing* in unrelated dims). Refit `(B−A) ≈ A·E`, `W = I+E`. (Raw lstsq `W` had ‖W‖_F ≈ 5e5 — pure noise; ridge → ‖W‖_F ≈ 9 at no cost to held-out cos.)
> - **Is the operation low-rank? NO — it's distributed.** SVD of `E`: 50% of action energy needs **79** of 300 components, 90% needs 198. There is **one mildly-dominant component** (σ₁=2.0, ~1.4× the next) then a long slow tail. So pluralization is "one main direction + broad word-class-dependent adjustments," not a handful of clean directions. (Plausibly real: the plural shift isn't identical across word classes.)
> - **The dominant component IS interpretable** — it reads the singular↔plural axis (+ side: `a, is, an, was`; − side: `these, are, those, many`) and writes toward the plural side. A coherent grammatical-NUMBER direction. (Component 2 captures the verb-3sg sense — an artifact of the `+s` set conflating noun-plural with verb agreement.)
> - **Mostly stretch, not rotation:** `E` is **94.6% symmetric / 5.4% antisymmetric**. So the operation is a scaling/projection along directions, not a Lie rotation; its generator is ~symmetric. (The small antisymmetric part still yields ~127 tiny complex eigenvalue pairs.)
> - **Caveats:** held-out gain is modest (0.385→0.435 on the noisy harvested `+s` set; cleaner on exp104's curated nouns); "high-rank" may partly reflect the noisy training set. A clean **noun-only** plural set would sharpen the decomposition. Results in `results_exp106.txt`.

> **RESULT — exp107 (control for frequency explicitly; `exp107_frequency_control.py`).** Niamh: "is there a way to control for the frequency axis in the decomposition?" Built an explicit frequency axis `f = unit(regression of log-rank on embeddings)` (R²=0.83 on log-rank) and projected it out; compared the plural operator under raw / PC1-stripped / frequency-regressed controls.
> - **Plural is NOT a frequency artifact (clean contrast with hardness).** The raw operator's action couples to frequency by only ~5% (‖Ef‖/‖E‖ = 0.052 input, ‖fᵀE‖/‖E‖ = 0.048 output), and deflating `f` from both sides removes just **0.2%** of ‖E‖. The dominant component is nearly **orthogonal** to frequency: cos(v₁,f)=0.09, cos(u₁,f)=0.07.
> - **Controlling for frequency changes essentially nothing:** raw vs freq-controlled decomposition is identical (cos op 0.452→0.453, ‖E‖ 7.80→7.98, σ₁ 1.42→1.44, rank@50% 72→72). So the grammatical-number interpretation of the dominant component stands — it is real beyond frequency. (Unlike HARDNESS, which was ~half frequency.)
> - **Loose end (flag for earlier conclusions):** the regression frequency axis `f` is nearly **orthogonal to PC1** (cos = 0.22), even though both correlate with frequency rank (PC1 spearman ~0.9 monotonic; `f` R²=0.83 linear-on-log). So "PC1 = frequency" (exp100–103 framing) was too glib — PC1 captures frequency *monotonically* but isn't the *linear* frequency gradient; there are multiple frequency-associated directions. Doesn't overturn exp101 (PC1 poles are unambiguously frequency-sorted: noise/rare vs function/frequent) or the plural result (robust to both controls), but the cleanest frequency control would remove a small frequency **subspace** (PC1 + regression axis), not a single direction. Results in `results_exp107.txt`.

> **RESULT — exp108 (the nonsense tail; `exp108_frequentiser_nonsense.py`).** Niamh: the rare end of frequency is mostly nonsense, not rare real words — does that matter? **It matters a lot, and it reframes the PC1 story.**
> - **PART A — "PC1 = frequency" was actually "PC1 ≈ nonsense/coherence."** Frequency axis fit on a wide set incl. the junk tail (`f_all`) vs on meaningful words only (`f_clean`): `cos(f_all, f_clean) = 0.48` (substantially different directions). `cos(f_all, PC1) = 0.62` but **`cos(f_clean, PC1) = 0.017`** — the *real-word* frequency axis is **orthogonal to PC1**. f_all's extreme = function words/punctuation; f_clean's rare extreme = rare-but-REAL words (`telerate, baronetcies, oxidant, cofactors`). **So stripping PC1 in exp100–106 was controlling the meaning↔nonsense axis, not real-word frequency.** Reframes "hardness was ~half frequency" → "~half aligned with the dominant meaning/coherence(nonsense) axis." The nonsense tail was hijacking what we called "frequency."
> - **PART B — a register frequentiser works, but "frequency" is ≥3 tangled directions.** 36 formal→casual synonym pairs (`purchase→buy`): real frequency shift (casual ~8000 ranks more common), and as an operation it generalizes leave-one-out (69% retrieves the casual synonym, **100% lands on a more-frequent word**, meaning preserved cos 0.82). BUT the register direction aligns with PC1/f_all (−0.60/−0.50), **not** with f_clean (−0.11). So there are at least three frequency-ish directions, not one: **nonsense** (PC1; meaning↔junk), **register** (`t_freq`; formal↔casual, ≈PC1-ish), **genuine rarity** (`f_clean`; common-real↔rare-technical-real, ⊥PC1). They overlap partially.
> - **Implication:** "the frequency confound" is not a single axis — it's a small tangle. Niamh's two-factor idea (a **nonsensifier** × a **frequentiser** as separate factors) is the right shape; exp108 shows they're genuinely distinct directions. Next: fit both from data (WordNet synonyms → register frequentiser at scale; rule-identified junk → nonsensifier) and test whether they're orthogonal and jointly account for the confound. Results in `results_exp108.txt`.

> **RESULT — exp109 (do the confound factors separate? `exp109_confound_factors.py`).** Niamh: should there be a FORMALISER too? Fit three candidate directions and let the data decide the number of factors (20,157 WordNet register pairs; 28,813 junk tokens vs 43,015 real words).
> - **The formaliser does NOT separate — it collapses into the nonsense axis.** Pairwise cosines: nonsensifier↔PC1 **0.78**, formaliser↔PC1 **0.83**, **nonsensifier↔formaliser 0.74**. So formality, nonsense, and PC1 are essentially **one "markedness" axis** (meaning ↔ formal/rare/junk). You can't split formality from nonsense in this space — formal words and junk both sit "away from the common core" in the same direction. So: **no separate formaliser warranted.**
> - **The frequentiser (f_clean) IS distinct** — orthogonal to everything (cos 0.02–0.23). So "frequency" is ~**2D**: a markedness/coherence axis (≈PC1, absorbs nonsense+formality) **plus** an orthogonal frequency-residual (`f_clean`). But it is NOT the 3-way nonsense×formality×rarity split that was hoped — formality folds into nonsense, and f_clean's interpretation is unclear (it predicts log-rank linearly but is ⊥ the monotonic markedness axis; possibly two coordinates jointly encode frequency).
> - **Hardness rode the markedness axis:** cos(hardness, formaliser) = **−0.55** (strongest), PC1 −0.40, nonsensifier −0.35, frequentiser −0.15; **31% of hardness's variance sits in the 3-factor confound span.** Confirms the hardness contrast was substantially a common-vs-marked(formal/rare/junk) effect, ~31% confound / ~69% other.
> - **Verdict:** the clean multiplicative factorization (nonsensifier × formaliser × frequentiser) does NOT hold — tested, and formality collapsed into nonsense. The confound is ~2 entangled directions (markedness + an orthogonal frequency-residual), best removed as a small subspace, not assumed-separable factors. Results in `results_exp109.txt`.
