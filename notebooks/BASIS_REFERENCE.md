# BASIS_REFERENCE.md

Structured reference for the basis-construction phase of the embeddingexp project (exp45–exp70). Substrate throughout: `glove-wiki-gigaword-300` (400 000 token vocabulary). All axes built as the mean of (word_a − word_c) differences across an anchor-pair set, then unit-normalized.

Sources: `LAB_NOTEBOOK.md` entries 23–25 (lines 2445–2882) and `LAB_NOTEBOOK_v2.md` entries 1–23 (exp61–exp98 + clustering work); `exp45_diverse_triples.py` through `exp98_all_but_the_top.py`; result text dumps for each. Numeric tables below are taken from the result text dumps; consolidated anchor lists for all current and deprecated axes are in `project_axis_vocabulary.py`.

**Recalibration note (2026-05-28).** The earlier framing "current basis (post-exp80): C, W, ATTENTION_CLEAN, INTENTION_CLEAN, R, D, IO_CLEAN, DV, MB, EPISTEMIC_VALUE, ABSTRACT_CONCRETE, REAL_IMAGINARY" treated the 12-axis list as *the* basis. The May-27/28 session (exp61–exp98 + clustering) undid that framing on three independent grounds:

- **Roget-13 baseline (exp86)** — a 13-axis basis built from purely physical/sensory antonyms (TEMPERATURE, MOISTURE, HARDNESS, etc.) matches the 13-axis cognitive basis on cognitive-content coverage (~39% vs ~41%), with only 12.8% subspace overlap. Antonym structure itself does substantial work; cognitive-vs-physical may be two organizational schemes capturing the same subspace from different angles.
- **Greedy selection (exp90)** — HARDNESS picked FIRST at +21pp single-axis coverage on cognitive content, beating any cognitive axis. Greedy hybrid bases (cognitive + Lakoff + Roget pool) outperform any single-framework basis by ~9pp.
- **Greedy-PCA parity (exp94)** — greedy from an interpretable-axis pool achieves the same coverage as PCA at the same axis count. Multiple decompositions of similar quality exist; specific axis sets are non-unique.

What this document now documents: **a pool of constructable candidate axes, plus what we know about the space and what we don't.** Specific 12-axis or 13-axis subsets are *historical states used in specific experiments*, not "the basis."

§1 has been rewritten with this framing (§1.0–§1.4 below). The detailed per-axis entries (C, W, ATT_CLEAN, …) are preserved as axis-pool reference material; what changed is the surrounding framing. §2–§7 still reflect the 7-axis-era analyses (inter-axis cosines, Lakoff PC validations, etc.) — those are accurate historical documentation of specific experiments; they need light updating to acknowledge resolved items (A-G entanglement, IO_CLEAN-misses-self/other gap) which are flagged inline.

---

## Section 1 — What we know about the space, the axis pool, and what we don't know yet

Updated 2026-05-28 to reflect the recalibration following exp86–exp98 + clustering (Entry 23). Previous version (2026-05-27) framed this section as "The 12 axes (post-exp80)" and presented the 12-axis basis as the current state of the project. The Roget result (exp86), greedy selection (exp90), and greedy-PCA parity (exp94) together made that framing untenable: specific axis sets are non-unique, multiple decompositions of similar coverage quality exist, and the durable contribution is methodology + structural findings about the space, not a fixed basis.

This section now documents: the substrate, what we know structurally about the embedding space, what "basis" means (and doesn't mean) here, which subsets have been used in which experiments, what we don't yet know, and the axis pool itself (detailed per-axis entries preserved from the previous version, reframed as documented constructions rather than basis members).

### The substrate

`glove-wiki-gigaword-300` (Pennington et al., GloVe embeddings trained on Wikipedia + Gigaword, 300 dimensions, 400 000 token vocabulary). Heavy anisotropy at the raw-vector level — a few high-variance directions (corpus frequency, register, tokenization) dominate the geometry and obscure semantic content. Cross-substrate work (Pythia 70m residual-stream SAE, exp45–51) showed substrate organization differs in PC distribution (SAE bundles where word2vec spreads). fastText work is partial; see §6.4 of WRITEUP_v4.

All anchor-based axes throughout this document are constructed as `A = unit(mean(w[pole_a_i] − w[pole_b_i]))` over matched pairs. The subtraction cancels shared anisotropy components by construction (see WRITEUP_v4 §1.5).

### Structural facts about the cognitive-content subspace

Non-circular operationalizations (independent of our axis-curation choices):

- **The 4-super-region macro structure of word2vec (Entry 23, full dirty 50K vocab, K=30 k-means → K=4 hierarchical superclustering):**
  - Super-cluster 1 (~7 sub-clusters): English discourse + Western proper nouns
  - Super-cluster 2 (~10 sub-clusters): foreign / noise / under-represented tokens
  - Super-cluster 3 (~5 sub-clusters): scientific / technical / compound vocabulary
  - Super-cluster 4 (~8 sub-clusters): numbers + Eastern/foreign geographic

  Cognitive content lives in super-clusters 1 + 3. Super-clusters 2 + 4 are largely non-cognitive structural variance. The dominant organization of word2vec is by token-type / source-language / format, not by semantic content. Only ~6 of 30 sub-clusters are clearly semantic.

- **Centroid-arithmetic noise direction (Entry 23):** noise centroid built from rule-identified noise tokens (decimal numbers, email-format strings, etc.); meaning centroid from non-noise tokens; the centroid-difference direction has `cos = +0.65` with COG-PC1. Validates PC1's coherence/sense reading via methodology independent of curated cognitive vocab.

- **PC1 of any curated cognitive vocab ≈ COHERENCE/SENSE direction.** `cos(clean-PC1, SENSE_vs_NONSENSE_anchor) = +0.48`; `cos +0.87` across vocab filterings (exp97). Robust structural finding.

- **Greedy ≈ PCA at same axis count on cognitive test split (exp94).** Interpretable greedy from the axis pool matches PCA coverage at the same dimensionality. Interpretability "tax" is approximately zero on this objective.

- **HARDNESS picked first by greedy at +21pp single-axis on cognitive content (exp90).** Multi-sense embodied-metaphor cluster (physical hardness → difficulty → severity → reality-status → personality → factuality). Not in Lakoff's canonical schema list.

- **Imagining as coordinated multi-axis rotation (exp81).** Mean delta vector `v(imagined) − v(real)` across 14 probe pairs decomposes as approximately `unit(0.60·REAL_IMAGINARY + 0.42·ABS − 0.31·W − 0.26·INT)`.

- **ABTT cross-substrate consistent at K=1–2 (exp98).** Same procedure surfaces phenomenological-vs-institutional (K=1) and virtue-vs-confusion (K=2) in both GloVe and fastText; substrates diverge at K=3.

### What "basis" means in this project (and what it doesn't)

A pool of constructable candidate axes (~50+, see below), plus a methodology for selecting subsets given an objective. **There is no privileged N-axis basis.** Multiple decompositions of similar coverage quality exist; specific axis sets that get selected depend on candidate pool, objective, and curation choices.

The Roget result is load-bearing for this reframe: a 13-axis basis built from purely physical/sensory antonym categories (TEMPERATURE, MOISTURE, SPEED, SIZE, STRENGTH, TEXTURE, TASTE, AGE, SOUND_VOLUME, CLEANLINESS, HARDNESS, SHARPNESS, DENSITY) matches our 13-axis cognitive basis on cognitive-content coverage at ~39% vs ~41%, with only 12.8% subspace overlap. Either antonym structure itself does most of the work (with cognitive-vs-physical being two organizational schemes capturing the same subspace from different angles), or we don't yet understand what the space is organized by. We are not making a strong choice between those readings.

The detailed per-axis entries in the next subsection are **documented constructions in the axis pool**. They are not "the basis." Specific experiments have used specific subsets of these axes; see "Subsets used in specific experiments" below.

### Subsets used in specific experiments

For reproducibility — anyone wanting to replicate exp60 needs Cog-7; anyone wanting to replicate exp90 needs the greedy-19 set.

| Subset name | Axes | First used | Purpose |
|---|---|---|---|
| Cog-7 | C, W, A, G, R, D, IO_CLEAN | exp60 | First active-inference-shaped basis; Part 3-4 of WRITEUP_v4 |
| Cog-9 | C, W, ATT_CLEAN, INT_CLEAN, R, D, IO_CLEAN, DV, MB | exp69 + exp63/64 | A/G replaced by cross-bleed-screened versions; gaps in self/other and verdict addressed |
| Cog-10 | Cog-9 + EV | exp73 | Full first-order EFE = C − W − EV lexically realized |
| Cog-12 | Cog-10 + ABS + REAL_IMAGINARY | exp76, exp80 | Cognitive operators Lakoff-bootstrapping missed |
| Cog-13a | Cog-12 + UD | exp84 | UD added as foundational ordering primitive |
| Cog-13b | UD, C_resid, W, ATT_CLEAN, INT_resid, R, D, IO_CLEAN, DV_resid, MB, EV, ABS, REAL_IMAG_resid | exp84 | Trouble axes residualized against UD; max off-diagonal 0.348 |
| Roget-13 | TEMPERATURE, MOISTURE, SPEED, SIZE, STRENGTH, TEXTURE, TASTE, AGE, SOUND_VOLUME, CLEANLINESS, HARDNESS, SHARPNESS, DENSITY | exp86 | Physical-antonym control baseline |
| Greedy-19 | HARDNESS + NATURAL_ARTIFICIAL + 6 Roget + 8 cognitive + 4 Lakoff + 1 cluster | exp90 | Hybrid basis from greedy selection on cognitive test objective; ~48% coverage |

The Roget-13 axis constructions and the Greedy-19 axis constructions are documented in the corresponding exp scripts (`exp86_roget_baseline.py`, `exp90_greedy_cognitive_roget.py`) and in `project_axis_vocabulary.py` once consolidated. Detailed reference entries for those axes are not yet in this document; HARDNESS specifically is treated in WRITEUP_v4 Appendix D.

### What we don't know yet

Open structural questions about the space:

- **Whether the cognitive-content subspace organizes around antonym structure, embodied-metaphor clusters, active-inference primitives, or some hybrid.** The Roget result is consistent with all three readings; it doesn't resolve which.
- **Whether the basis structure is substrate-general or GloVe-specific.** ABTT cross-substrate-consistent at K=1–2 only. fastText full-basis replication pending.
- **Whether HARDNESS dominates in other substrates / objectives / curations.** Single-substrate (GloVe-300), single objective (cognitive test categories), one candidate pool. Cross-substrate replication + external-vocabulary replication needed.
- **Whether the rotation-as-imagination geometry (REAL_IMAG +0.60, ABS +0.42, W −0.31, INT −0.26) is substrate-general or GloVe-specific.**
- **Whether causal steering effects in Pythia reproduce the geometric findings.** Load-bearing for the project's overall claim; HARDNESS in particular needs causal validation before it can be claimed as a substantive extension of Lakoff.
- **What the relationship is between super-cluster structure (Entry 23) and the axis-construction work.** Two non-circular operationalizations of "cognitive content vs noise" but they haven't been formally connected.
- **Whether the methodology produces the same axes when run by someone else with different vocabulary intuitions.** Curation-independence is open; external-vocabulary (WordNet psychological subcategories, BNC, LIWC) replication would test it.

See WRITEUP_v4 §0 (Scope and Status) and §7 (What's Pending) for the load-bearing validation tracks.

### The axis pool — detailed reference entries

The per-axis entries below are the constructable axes documented so far in the project. They are **not** "the basis"; they are the pool from which specific bases have been selected in specific experiments (see "Subsets used in specific experiments" above). Each entry includes anchor vocabulary, validation history, top pole words, and active-inference reading where applicable.

Newly added in this version: the UD entry (exp84). Roget-13 and Greedy-19 axes are not yet documented here in full detail; their constructions live in the respective exp scripts. The deprecated A_RUSSELL_DIAGONAL and G_GOAL_DIRECTED entries are preserved in §1.1 (immediately following this section) for reproducibility of exp52–exp68.

### C — Integrated reward / expected value of state

- **Constructed:** exp54 (`exp54_pc1_comparator.py`, candidate `C_REWARD_COMPOSITE`).
- **Anchor vocabulary (12 pairs, 12/12 in vocab):**
  ```
  (flourishing, suffering)  (thriving, struggling)  (prospering, declining)
  (blessed, cursed)         (fortunate, unfortunate)  (fulfilled, ruined)
  (privileged, oppressed)   (graced, plagued)       (charmed, hexed)
  (favored, disfavored)     (lucky, unlucky)        (wholesome, broken)
  ```
- **Semantic description:** integrated wellbeing / outcome-and-resource-state, bundling multiple dimensions into a single composite-reward word per anchor.
- **Active-inference reading:** the value-pole of expected free energy — prior over outcomes / pragmatic value. The "what's wanted" content of PC1.
- **Validation:**
  - `cos(C, PC1) = +0.544` — winner of the exp54 PC1 comparator (1.7× the next candidate B_VALUE_PURE at 0.321).
  - Top input-axis loadings: VALENCE +0.530, LOSS +0.479, UD +0.478, SUCCESS +0.400, EXIST +0.370.
  - Top positive-pole words: wholesome, vibrant, cosmopolitan, thriving, personable, flourishing.
  - Top negative-pole words: caused, suffered, plagued, suffering, injuries, causing, injury, blamed.

### W — Weight / cost / felt-burden

- **Constructed:** exp59 (`exp59_weight.py`, `TARGET_WEIGHT_PAIRS`).
- **Anchor vocabulary (12 pairs):**
  ```
  (heavy, weightless)       (weighty, airy)            (ponderous, buoyant)
  (burdensome, effortless)  (laden, unburdened)        (cumbersome, nimble)
  (leaden, feathery)        (dense, wispy)             (encumbered, unencumbered)
  (heavyweight, featherweight)  (massive, delicate)    (oppressive, lighthearted)
  ```
- **Semantic description:** somatic weight / effort-cost / resistance-to-action; later reframed by Niamh as **computational cost** with somatic vocabulary as one expressive modality.
- **Active-inference reading:** cost-pole of expected free energy (anti-reward). Together with C, the two contributions to expected-free-energy minimization on PC1. (Niamh's end-of-session "W → COST" reframe.)
- **Validation:**
  - `cos(W, PC1) = −0.381` (strongest single-axis hit on PC1 among all 7 axes — primarily a PC1-negative direction, not PC2 as initially predicted).
  - `cos(W, PC2) = −0.164`, `cos(W, PC3) = −0.021`.
  - Top input-axis loadings: VALENCE −0.334, AROUSAL +0.283, SUCCESS −0.249, LD −0.156, UD −0.147, LOSS −0.120.
  - `cos(W, DIFF) = +0.502` (strongest schema cosine for W; the next is SUC at −0.249).
  - Top positive (heavy) pole words: heavy, burden, costly, massive, government, authorities, blamed, heavily.
  - Top negative (light) pole words: ingenuous, effortless, wonderment, exudes, feathery, weightless, vivacious, effortlessly, playfulness.

### ATTENTION_CLEAN — Perceptual precision (γ)

- **Constructed:** exp69 (`exp69_clean_AG_and_intent_struct.py`, `ATTENTION_CLEAN_PAIRS`). Built from anchor pairs **screened to exclude intentional content** (no focused/directed/attentive — all of which carry agentive engagement, not pure perception).
- **Anchor vocabulary (12 pairs, 12/12 in vocab):**
  ```
  (noticing, missing)         (perceiving, overlooking)   (sensing, missing)
  (detecting, missing)        (spotting, missing)         (recognizing, overlooking)
  (seeing, missing)           (hearing, missing)          (registering, ignoring)
  (witnessing, missing)       (observing, missing)        (aware, unaware)
  ```
- **Semantic description:** perceptual registration of content — whether sensory input is being noticed or missed. Replaces A_RUSSELL_DIAGONAL (exp52) which conflated perceptual precision with intentional engagement via attentive/focused/directed.
- **Active-inference reading:** γ (perceptual precision) — the gain on prediction errors. The "what's being noticed" content of perception, separable from the policy-precision (π) that ATTENTION_CLEAN was previously entangled with.
- **Validation:**
  - **cos(ATTENTION_CLEAN, INTENTION_CLEAN) = −0.13.** Baseline `cos(A_orig, G_orig) = +0.561`. The −0.69 swing (exp69) established that the +0.56 entanglement was anchor-vocabulary bias, not substrate-real coupling. **This is the headline result of the basis-restructuring.**
  - `cos(ATTENTION_CLEAN, A_orig) = −0.13` — note negative. ATT_CLEAN is structurally a *different direction* from the original A, not a refined version of it.
  - `cos(ATTENTION_CLEAN, G_orig) = −0.05` — clean orthogonality against the old G.
  - Max |cos| with other basis axes: +0.18 with C (see §2 for the 9×9 matrix).
  - Top positive-pole words: perceiving, manifesting, legitimizing, perceive, manifested, semi-automated, multi-vendor, tolerantly, turbo-prop, 40min.
  - The "perceiving / perceive / manifested" cluster is the clean signal; "manifesting / legitimizing" reflect the noticing-as-publicly-acknowledging register; the tail is the same anisotropy-artifact pattern that appears on most axes.
- **History:** A_RUSSELL_DIAGONAL was used as the A axis in exp52–exp68. Its anchor vocabulary and pre-screening validation against Lakoff PCs are preserved in §1.1.

### INTENTION_CLEAN — Policy precision (π)

- **Constructed:** exp69 (`exp69_clean_AG_and_intent_struct.py`, `INTENTION_CLEAN_PAIRS`). Built from anchor pairs **screened to exclude attentional content** (no oriented/targeted — both of which carry perceptual engagement).
- **Anchor vocabulary (12 pairs, 12/12 in vocab):**
  ```
  (intending, drifting)       (planning, improvising)     (deciding, deferring)
  (committing, hedging)       (choosing, defaulting)      (designing, improvising)
  (resolving, postponing)     (scheduling, winging)       (plotting, freelancing)
  (aiming, drifting)          (intending, stumbling)      (plan, improvise)
  ```
  Note: `(aiming, drifting)` is flagged in `project_axis_vocabulary.py` as leaning slightly attentional but retained because aim-as-plan dominates the distributional sense.
- **Semantic description:** commitment-to-a-plan-or-course-of-action — whether a policy is being chosen or deferred. Replaces G_GOAL_DIRECTED (exp53) which conflated policy commitment with perceptual orientation via oriented/targeted.
- **Active-inference reading:** π (policy precision) — commitment to action policy. The "what's being committed to" content of agency, separable from the perceptual precision (γ) that INTENTION_CLEAN was previously entangled with.
- **Validation:**
  - cos(INTENTION_CLEAN, ATTENTION_CLEAN) = −0.13 (see ATTENTION_CLEAN entry).
  - `cos(INTENTION_CLEAN, G_orig) = +0.59` — INT_CLEAN substantially overlaps original G. Most of G's content was genuine policy-precision; the part that wasn't was the attentional cross-bleed.
  - `cos(INTENTION_CLEAN, A_orig) = +0.43` — INT_CLEAN also overlaps original A *more than ATT_CLEAN does*. This is the asymmetry that resolved the entanglement: most of "A's content that was getting credited to affect" was actually intentional content (focused/directed/attentive) landing on the policy-precision axis.
  - Max |cos| with other basis axes: +0.21 with IO_CLEAN, +0.20 with W (see §2).
  - Top positive-pole words (clean signal): planning, planned, wanted, plan, intended, plans, responsible, establish, aims, to. The infinitive marker "to" appearing in the top 10 is structurally interesting — the to-marker carries intentional content distributionally.
- **History:** G_GOAL_DIRECTED was used as G_pol in exp53–exp68. Its anchor vocabulary and pre-screening validation against PC2 are preserved in §1.1.

### R — Perceptual precision / regulability

- **Constructed:** exp52 as `TARGET_EQUILIBRIUM_RUNAWAY_PAIRS`; in exp55+ used as **R = EQ_RUN residualized against VALENCE and AROUSAL** (V+A-stripped to remove affect contamination).
- **Anchor vocabulary (14 pairs, 14/14 in vocab):**
  ```
  (correcting, escalating)   (adjusting, cascading)    (recalibrating, snowballing)
  (righting, spiraling)      (stabilizing, mushrooming)  (regulating, ballooning)
  (moderating, surging)      (tempering, propagating)  (dampening, amplifying)
  (restraining, intensifying)  (restoring, exploding)  (atoning, raging)
  (mending, festering)       (reconciling, ravaging)
  ```
- **Semantic description:** control-theoretic damped-feedback-regulation vs positive-feedback runaway cascade.
- **Active-inference reading:** perceptual precision over predictions — the third active-inference quantity. Semantic shape matches Adams/Friston precision-collapse models of psychosis.
- **Validation:**
  - Raw `cos(EQ_RUN, PC3) = +0.205`.
  - After PC1-residualization (95.7% magnitude retained): `cos = +0.214` (essentially unchanged).
  - After V+A-residualization (94.2% retained): `cos = +0.103` (partially dented — regulation is heavily affect-correlated in natural language).
  - Top input-axis loadings (pre-residualization): AROUSAL −0.286, LD +0.218, VALENCE +0.217, LOSS +0.164.
  - Positive pole: restoring, interpreting, modifying, tempering, adjusting, atoning, effecting, erred, accomplishing, nyts.
  - Negative pole: mushrooming, spiralling, spiraling, escalating, raging, engulfed, erupted, raged, escalated, sparked.
  - **Status:** "regulability hypothesis PARTIAL: PC3 alignment present but not strong after residualization" (exp53 verdict).

### D — Compression / surprisal / predictability

- **Constructed:** exp54 (`TARGET_SURPRISAL_PAIRS`).
- **Anchor vocabulary (12 pairs, 12/12 in vocab):**
  ```
  (familiar, unfamiliar)     (routine, novel)          (recognized, unrecognized)
  (anticipated, unanticipated)  (foreseen, unforeseen)  (commonplace, extraordinary)
  (mundane, astonishing)     (typical, atypical)       (customary, unprecedented)
  (rote, startling)          (habitual, jarring)       (everyday, shocking)
  ```
- **Semantic description:** predictability / compressibility — model performance signal. (Distinct from coherence: D is about per-token surprisal; COHERENCE is about within-sequence orderliness.)
- **Active-inference reading:** the prediction-error / surprisal / compression signal — one of three independent loss-function candidates (with A and C) in Niamh's multi-loss reading.
- **Validation:**
  - `cos(D, PC1) = +0.204`, `cos(D, PC2) = −0.245`, `cos(D, PC3) = +0.223`. Not a single-PC primitive.
  - Top input-axis loadings: COHERENCE +0.444, BAL +0.257, LOSS +0.197, VALENCE +0.175, AROUSAL −0.109.
  - Positive (predictable) pole: regular, practiced, usual, routines, routine, preferred, chores, practice.
  - Negative (surprising) pole: astonishing, startling, unforeseen, astounding, unexpected, unanticipated, shocking, stunning.

### IO_CLEAN — Container-topology (spatial in-out)

- **Constructed:** From `lakoff_canonical_vocabulary.py` as `IN_OUT_MML_CLEAN`. Added to the basis in exp60.
- **Anchor vocabulary (10 pairs):**
  ```
  (inside, outside)         (interior, exterior)       (within, without)
  (enter, exit)             (entered, exited)          (contained, released)
  (enclosed, exposed)       (sealed, opened)           (inhaled, exhaled)
  (during, after)
  ```
- **Semantic description:** spatial-containment / inside-vs-outside topology. Captures the **spatial-container expression** of self/non-self distinctions; the **abstract-agent expression** lives on MB (see below).
- **Active-inference reading:** spatial-container Markov-blanket expression. With MB now in the basis, the original "IO as Markov-blanket candidate" reading is refined: IO is the spatial side, MB is the abstract-agent side, both are real and complementary (see MB entry).
- **Validation:**
  - "Gorgeously orthogonal" to other 7 basis axes: max `|cos|` with any other axis = 0.11 in the 7-axis matrix — see Section 2 for the 9×9 values.
  - Tier-2 status: only ~22% basis-explained in the 6D AI-plus basis from exp55.
  - Adding IO as a basis axis improves other schemas' explained-variance by 0–4% (max +3.9% for PATH); IO is "its own thing" — other schemas don't tap into it.
  - On concept-word coverage: only `belonging` (+0.291) loads strongly on IO among self/other-relevant probes. Other phenomenal-self vocabulary (self, ego, soul, identity, autonomy) loads on MB instead (see MB entry).

### DV — Decision-verdict

- **Constructed:** exp63 (`exp63_thread1_selection.py`) as `target_SELECTION`; renamed across exp64 Part A (SELECTION → GATING → DECISION-VERDICT in exp64 A3) once the pole vocabulary made the verdict-outcome reading clear. The hyphenated form `("singled-out", "ignored")` was dropped; otherwise the anchor list matches the original SELECTION construction.
- **Anchor vocabulary (11 pairs, 11/11 in vocab):**
  ```
  (selected, rejected)        (chose, refused)            (picked, discarded)
  (admitted, denied)          (accepted, declined)        (kept, removed)
  (chosen, eliminated)        (preferred, overlooked)     (favored, excluded)
  (designated, omitted)       (highlighted, neglected)
  ```
- **Semantic description:** the lexical axis of evaluation **outcomes** — what counts as in vs out after a judgment process. Distinct from the evaluation-process itself (which is distributed across ATTENTION_CLEAN, INTENTION_CLEAN, and C), and distinct from spatial containment (cos with IO_CLEAN = +0.017).
- **Active-inference reading:** not a precision quantity. The discrete-verdict output of an evaluative process — what active-inference would call a sampled-action from a precision-weighted policy distribution, expressed lexically in past-tense judgment verbs.
- **Validation:**
  - From exp63 construction: cos(DV, IO_CLEAN) = +0.017 (refuted DV-reframes-IO hypothesis), cos(DV, G_pol) = +0.16 (refuted DV-subsumes-G hypothesis). DV is its own thing.
  - 91.6% of DV's variance is NOT C: `|DV − ⟨DV,C⟩·C| = 0.957`. Distinct from reward.
  - Residual-after-C decomposition onto remaining 7-axis basis (exp64 A2): cos(., A_orig) = +0.22 (highest), cos(., G_orig) = +0.16, cos(., W) = −0.10. Small couplings throughout — DV is an independent axis.
  - **Positive pole** (after C-removal): chosen, selected, select, choosing, selecting, chose, pick, preferred, choose, designated, favorite, selection, picked — a clean selection-verb cluster.
  - **Negative pole**: flatly, categorically, refuted, substantiate, untrue, disproved, rebutted, adamantly, substantiated, superfluous, overblown, strenuously — the **denial register** as the linguistic marker of negative verdict. This is the axis's genuine other pole, not contamination: in language, the lexical correlate of "rejected outcome" is the rhetorical register of categorical denial.
- **Gating-process vs verdict (from exp64 A3):** attention/focus/decision/judgment vocabulary all load on the (old) A+G subspace (now ATTENTION + INTENTION), NOT on DV. ATT and INT host the verdict-generating process; DV is the verdict-outcome. The naming evolution SELECTION → GATING → DV tracked this disentanglement.

### MB — Markov-blanket / substrate self-other

- **Constructed:** exp64 Part B (`exp64_gating_and_markov.py`, `TARGET_MARKOV_BLANKET_PAIRS`). Sub-thread C2 in the original test queue.
- **Anchor vocabulary (13 pairs, 13/13 in vocab):**
  ```
  (self, other)               (agent, environment)         (internal, external)
  (mine, theirs)              (own, foreign)               (private, public)
  (subjective, objective)     (introspection, perception)  (autonomous, dependent)
  (individual, collective)    (personal, impersonal)       (intrinsic, extrinsic)
  (endogenous, exogenous)
  ```
- **Semantic description:** abstract self/other distinction in the substrate — the agent/environment partition (Markov blanket in Friston's formal sense). Picks up the phenomenal-self vocabulary (self, ego, soul, identity, autonomy) that IO_CLEAN structurally missed.
- **Active-inference reading:** the substrate Markov blanket — the formal partition between agent and environment that grounds active-inference cognition. Where IO_CLEAN captures spatial-container topology, MB captures the abstract agent/environment relation. Both are Markov-blanket-related; they're complementary axes capturing different expressive modalities of the same underlying primitive.
- **Validation:**
  - **Cleanest member of the basis.** Max |cos| with any other axis = +0.137 (with IO_CLEAN).
  - cos with each axis (from exp64 B2): C −0.05, W −0.09, A_orig +0.00, G_orig +0.03, R +0.04, D +0.01, IO_CLEAN +0.14, DV +0.12. All small.
  - **Concept-word coverage:** MB has larger |cos| than IO on 15 of 21 self/other-relevant probes (self +0.32, ego +0.18, soul +0.19, identity +0.12, individuality +0.16, autonomy +0.20, selfhood +0.16, communion +0.09, exile +0.15, interiority +0.08, loneliness +0.09, ...). IO is larger on 6 (belonging +0.29, sovereignty, kinship, isolation, alienation, subjectivity).
  - **Positive pole** (self / agent side): self, mine, autobiography, agent, personal, own. Tail has proper-noun anisotropy (kwalik, parker, k-9, seligson — rare names that cluster pathologically in GloVe).
  - **Negative pole** (other / environment side): external, foreign, environment, globalization, macroeconomic, policy-makers, globalised, over-fishing — clean "external environment" cluster with socioeconomic register-skew (the corpus's most prominent uses of "environment" and "external" are in macroeconomic/policy contexts).

### EPISTEMIC_VALUE — Drive to gather information about hidden states

- **Constructed:** exp73 (`exp73_epistemic_value.py`, `TARGET_EPISTEMIC_VALUE_PAIRS`). Thread NEW, post-exp70 refinement: state-based curiosity vocabulary only, no activity-verbs (investigate / explore / probe / inquire / seek — those load on INTENTION_CLEAN per exp70's empirical split).
- **Anchor vocabulary (10 pairs, 10/10 in vocab):**
  ```
  (curious,     indifferent)   (intrigued,    dismissive)
  (fascinated,  bored)         (inquisitive,  incurious)
  (puzzled,     settled)       (wondering,    knowing)
  (marveling,   dismissing)    (awestruck,    jaded)
  (mystified,   certain)       (engaged,      blase)
  ```
- **Semantic description:** the epistemic-engagement state — being-in-curiosity-toward-the-not-yet-known. Captures *what makes uncertainty valuable to engage with* (the driving signal of exploration), as distinct from the (+ATT, −INT) state-space coordinates of explore-mode itself.
- **Active-inference reading:** the epistemic-value term of expected free energy. With C (pragmatic value) and W (cost) already in the basis, EV completes the first-order EFE decomposition: **EFE = C − W − EV**. This is the quantity that, alongside pragmatic value, drives policy selection under uncertainty about hidden states / model parameters.
- **Validation:**
  - **Max |cos| with the 9-axis basis (pre-EV) = +0.154 (with DV).** Among the cleanest members of the basis, comparable to MB (+0.137).
  - cos with each axis: C +0.023, W −0.108, ATT +0.033, **INT −0.137**, R −0.043, D −0.146, IO +0.037, DV +0.154, MB +0.061.
  - **Falsifiers cleared:** max |cos| < 0.40 (derived state ruled out); cos with INT = −0.137 (activity-verb leakage ruled out — INT screening succeeded).
  - **Concept-word coverage:** state-curiosity words load cleanly on EV — intrigued +0.55, fascinated +0.54, puzzled +0.49, mystified +0.48, marveling +0.45, curious +0.45, wondering +0.39, inquisitive +0.35, awestruck +0.34.
  - **Activity-verbs load on INT, not EV** — investigating +0.32 INT (+0.16 EV), seeking +0.41 INT (−0.10 EV). Confirms the state-vs-activity split from exp70.
  - **Drift-cluster loads on neither** — daydreaming, drifting, lethargic, unfocused all have low EV (≤+0.14) AND low INT (≤−0.27). The active-inference distinction between "active novelty-seeking driven by epistemic value" and "passive drift without that drive" is now empirically separable in the basis.
  - cos(EV, EXPLORATION = unit(ATT − INT)) = +0.113 — small. EV is NOT in the explore-quadrant; it's a separate dimension that *drives* exploration when present in conjunction with high γ and low π.
- **Top positive-pole words:** intrigued, fascinated, marveled, entranced, captivated, curious, puzzled, mystified, wondering, inquisitive, awestruck.
- **Top negative-pole words:** outright, prior, authorization, rejection, previous, pre, approval, customary, failure, fails, failed, payment, rejecting, dismissal. Conceptually clean as the opposite of curious-state ("already-decided / authorized / customary / dismissed"). Note the rejection/approval/authorization/dismissal cluster — this is DV-domain content, which explains the +0.154 coupling. Substrate-real overlap between *incurious-state* and *verdict-rendered-state*.

### ABSTRACT_CONCRETE — Abstract / concrete

- **Constructed:** exp76 (`exp76_coverage_comparison.py`, `ABSTRACT_CONCRETE_PAIRS`). Proposed by Niamh after exp75's modest coverage finding suggested Lakoff-bootstrapped construction was structurally missing some major cognitive operators.
- **Anchor vocabulary (10 pairs, 10/10 in vocab):**
  ```
  (abstract,     concrete)       (theoretical,  practical)
  (conceptual,   physical)       (general,      specific)
  (idea,         object)         (principle,    instance)
  (intangible,   tangible)       (notion,       thing)
  (categorical,  particular)     (ideal,        material)
  ```
- **Semantic description:** abstract / general / conceptual ↔ concrete / specific / physical. The dimension of how distant a representation is from sensory/embodied particulars.
- **Active-inference / cognitive reading:** not a directly-named active-inference quantity, but plausibly the substrate-level expression of representational-distance-from-direct-contact. Captures content that operates at higher levels of generative-model abstraction (abstract pole) vs. content tied to concrete sensory-motor instances (concrete pole).
- **Validation:**
  - **Empirically confirmed as independent primitive in exp81**: regressing ABS onto the full 11-axis basis (including REAL_IMAGINARY) gives R² = 18.4% — **~82% of ABS's variance is outside everything else we have**. ABS captures structure that no combination of other primitives recovers.
  - Cross-bleed (exp77): max |cos| with 10-axis basis = 0.273 (with W — abstract is anti-correlated with somatic-concrete W, as expected). Other notable: ATT +0.134, INT −0.169.
  - Coverage contribution (exp76): adding ABS + MOD to 10-axis basis gave +5.0pp mean coverage across categories, with biggest gain on ABSTRACT_FORMAL (+8.4pp) as predicted; lift was broad-based across all categories (3.6–8.4pp).
  - Subspace orthogonality (exp78 by extension, exp79 directly): ABS has 10.1% variance in Lakoff-12 subspace — at near-random baseline; it lives in the orthogonal-complement subspace where standard cognitive-linguistic decomposition cannot reach.
- **Top positive-pole words (abstract side):** kantian, propounded, conceptions, posits, idealist, materialist, epistemology, posited, hegelian, metaphysics. Clean philosophical-abstract cluster.
- **Top negative-pole words (concrete side):** material, contain, few, damage, specific, enough, identify, inside, additional, amount, actually, find. Concrete/material/specific cluster, slightly muddled with general high-frequency vocabulary.

### REAL_IMAGINARY — Real / imaginary (formerly MODAL_STATUS)

- **Constructed:** exp76 as `MODAL_STATUS_PAIRS`, refined in exp80 with cleaner anchors. Renamed REAL_IMAGINARY in exp80 after the probe test confirmed the axis captures the real/imaginary distinction specifically.
- **Anchor vocabulary (10 refined pairs, 10/10 in vocab):**
  ```
  (hypothetical,   actual)         (imagined,     observed)
  (imaginary,      real)           (fictional,    factual)
  (counterfactual, demonstrated)   (speculative,  confirmed)
  (conjectural,    verified)       (presumed,     proven)
  (notional,       materialized)   (alleged,      documented)
  ```
  Each word appears in at most one pair. No broad-distribution function words (could/can/is from the original were removed). No anchor-word overlap with ABS (theoretical was in both original MOD and ABS; removed from MOD).
- **Semantic description:** representational status — whether content is treated as world-referring (real) or as mind-internal-simulation (imaginary). The substrate-level distinction underlying counterfactual reasoning, fictional content, speculative thought, and policy rollout in active inference.
- **Active-inference / cognitive reading:** the distinction between observed evidence (real) and simulated policy rollouts (imaginary) is one of the foundational operations in any generative model. Distinguishing predictions from observations grounds prediction-error computation. REAL_IMAGINARY is plausibly the substrate-level expression of this in distributional semantics.
- **Validation:**
  - **Probe test (exp80) is the headline confirmation.** 10 real/imaginary word pairs *outside* the anchor list — (imagination, perception), (fantasy, memory), (dream, experience), (fiction, history), (myth, fact), (speculation, observation), (vision, witness), (supposition, evidence), (conjecture, data), (rumor, report). **All 10 are consistent**: the imagined-side word loads more positively on REAL_IMAGINARY than the real-side word, with magnitudes up to +0.56. The axis generalizes to words it wasn't built from.
  - Cross-bleed (exp80): max |cos| with 11-axis basis = +0.32 (with ABS). Under 0.35 threshold; substrate-real overlap (imagined content is more abstract). Other notable: INT −0.22 (committed-to-policy vs imaginary), W −0.13 (somatic vs imaginary).
  - Coverage contribution: as part of (ABS + MOD), contributed to +5.0pp coverage in exp76; per-category lift includes MODAL_ACTUAL +5.0pp and MODAL_HYPOTHETICAL +3.8pp.
  - Rotation-imagining test (exp81): the mean delta vector `v(imagined_i) − v(real_i)` across 14 probe pairs has cos = +0.60 with REAL_IMAGINARY. Also +0.42 with ABS, −0.31 with W, −0.26 with INT. **The "imagining direction" in word-vector space is multi-axis: approximately `unit(0.6·REAL_IMAGINARY + 0.42·ABS − 0.31·W − 0.26·INT)`**. Imagining as a computational operation is a coordinated rotation across multiple basis axes, not single-axis.
- **Top positive-pole words (imaginary side):** imaginary, conjectural, daydreams, sword-and-sorcery, post-modern, steampunk, look-alike (clean cluster) mixed with register noise (shiksa, low-life, guianese, scytalopus, borophaginae, wastebin, 34-acre).
- **Top negative-pole words (real side):** demonstrated, confirmed, showed, verified, noted, achieved, reported, documented, concluded. Beautiful clean evidence/verification cluster.
- **Asymmetry note:** negative pole is cleaner than positive. Consistent with GloVe's corpus skew toward reported-fact register (news + Wikipedia have more direct vocabulary for "verified evidence" than for "imagined content").
- **Naming note:** originally called MODAL_STATUS until probe test confirmed it captures the real/imaginary distinction specifically rather than the broader logician's modal status (which also includes necessity, permission, etc.). REAL_IMAGINARY is empirically more honest.

### UD — Up-down / foundational ordering primitive

- **Constructed:** exp84 (`exp84_ud_refactor.py`) as `target_UD` built from `UP_DOWN_MML` anchors in `lakoff_canonical_vocabulary.py` — the canonical Master Metaphor List UP/DOWN contrastive vocabulary (57 pairs).
- **Anchor vocabulary:** see `UP_DOWN_MML` in `lakoff_canonical_vocabulary.py`. Includes literal vertical pairs (up/down, rising/falling, ascending/descending) plus the MML primary-metaphor family (HAPPY-IS-UP, MORE-IS-UP, GOOD-IS-UP, HOPE-IS-UP, ALIVE-IS-UP, ACTIVE-IS-UP, etc.).
- **Semantic description:** vertical-ordering primitive, with the substantial metaphorical breadth that Lakoff's UP/DOWN schema canonically carries — affective (happy/sad), evaluative (good/bad), quantitative (more/less), aspirational (hope/despair), vital (alive/dead), epistemic (known/hidden), and others.
- **Reading:** UD turns out to be **the most explanatory single primitive in the axis pool** for cognitive content (per exp87 minimal-basis coverage: 18.3% single-axis on random vocab, 4.5× random-1 baseline). Not exclusively a Lakoff schema in behavior — it functions as a foundational ordering / scalar-evaluation primitive that other constructed axes partially borrow from.
- **Cross-bleed with 12-axis basis (exp84):**
  - **High UD-overlap:** C +0.48, DV +0.36, INT +0.30, REAL_IMAG −0.29.
  - **Low UD-overlap:** ATT +0.02, R −0.08, D +0.10, IO +0.16, W −0.15, MB +0.08, EV +0.08, ABS +0.03.
  - The pattern "trouble axes from prior session" = "high UD-overlap axes" was hidden in plain sight until UD's status as primitive surfaced. C, DV, INT, REAL_IMAG had all been iteration-magnets (multiple renames or refinements) earlier; their UD-content was the underlying issue.
- **Validation (exp84):**
  - Used as 13th axis in Cog-13a (original 12 + UD).
  - Used as the *first* basis vector + residualization anchor in Cog-13b (UD primary; C, INT, DV, REAL_IMAGINARY residualized against UD as `A_resid = unit(A − (A·UD)·UD)`).
  - Cosine shifts from residualization: C ~0.88 (substantive), INT ~0.95 (modest), DV ~0.93 (modest), REAL_IMAGINARY ~0.96 (essentially unchanged).
  - Max off-diagonal in Cog-13b: 0.348 (down from 0.478 in Cog-13a). 0/78 pairs exceed 0.35 in Cog-13b.
  - PC1 of Cog-13b refactored basis: `−0.55·UD − 0.47·W + 0.35·C_resid + 0.33·ATT` (var 15.51%). Reads cleanly as "embodied gravity (UD + W) vs cognitive appreciation (C_resid + ATT)."
  - Coverage: Cog-13b at 40.58% mean (+3pp over Cog-12).
- **Residualized-axis probe tests (exp84):** 5 probe pairs per residualized axis, none in the original anchor lists. **20/20 consistent** — each residualized axis still does its primitive work. happiness > suffering Δ +0.39 (C_resid), determined > irresolute Δ +0.60 (INT_resid), endorsed > rejected Δ +0.32 (DV_resid), supposition > evidence Δ +0.54 (REAL_IMAG_resid).
- **Methodological wart (exp85):** residualized axes don't fully reconstruct from anchor changes alone. C_resid: dropping (prospering, declining) from C's anchors gives a reconstruction with cos +0.84 to C_resid but cos(reconstructed, UD) = +0.49 (essentially unchanged from original C-UD). INT_resid, DV_resid, REAL_IMAG_resid show similar partial reconstructions. The UD-content of these axes comes from distributional clustering, not from specific vertical-metaphor anchor-words. **Implication:** the basis construction is either (a) a two-step procedure (anchor-build + residualize against UD) or (b) accepts the partial reconstructions. Flag for any methodology section in a future paper.
- **Relationship to VALENCE:** `cos(VALENCE, UD) = +0.60` in GloVe-300 (exp87). They are not independent foundational dimensions; they overlap heavily. UD captures both vertical-spatial and valence-evaluative content. The (VAL + UD) composite captures 21% of word-vector content alone (vs ~41% for 13-axis cognitive basis), but mostly via shared subspace — the 11 other cognitive axes are largely orthogonal to VAL+UD (>90% orthogonal for 9 of 11; C is the major exception at 32% in VAL+UD subspace).

---

## Section 1.1 — Deprecated original axes (reproducibility of exp52–exp68)

A_RUSSELL_DIAGONAL and G_GOAL_DIRECTED are preserved here. Their anchor vocabularies and pre-screening validation against Lakoff PCs remain the correct reference for any analysis or replication of exp52–exp68. They were replaced in exp69 by ATTENTION_CLEAN and INTENTION_CLEAN above.

### A_RUSSELL_DIAGONAL — DEPRECATED (was the A axis through exp68)

- **Status:** Deprecated as of exp69. Replaced by ATTENTION_CLEAN. Anchor pairs contained intentional content (focused/directed/attentive) that produced the +0.56 entanglement with G; cross-bleed screening showed the entanglement was anchor-vocabulary bias, not substrate-real coupling.
- **Constructed:** exp52 as `TARGET_SALIENCE_PAIRS` (Niamh's valence-balanced "salience" anchors, but the resulting axis turned out to BE Russell's V×A diagonal, not salience). Re-used as the affect axis from exp54 onward.
- **Anchor vocabulary (12 pairs, 11/12 in vocab — `backgrounded` OOV):**
  ```
  (important, unimportant)  (urgent, idle)             (salient, irrelevant)
  (attentive, inattentive)  (focused, unfocused)       (directed, diffuse)
  (prominent, inconspicuous)  (noticeable, unnoticeable)  (foregrounded, backgrounded)
  (highlighted, overlooked)  (conspicuous, unobtrusive)  (pronounced, muted)
  ```
- **Original semantic description:** affect-as-felt-state — Russell's pleasant-calm vs unpleasant-aroused diagonal. *Revised reading:* a mix of attentional engagement and goal-orientation that the anchor vocabulary couldn't separate.
- **Validation against Lakoff PCs (pre-screening, for reproducibility):**
  - `cos(A, PC1) = −0.174`, `cos(A, PC2) = −0.280`. Originally hypothesized to capture PC1=salience; refuted — distributed across AROUSAL (+0.486), COHERENCE (+0.425), BAL (+0.383), FB (+0.363), UD (+0.361).
  - Used as A_RUSSELL_DIAGONAL in exp54's comparator; lost to C_REWARD_COMPOSITE.
  - Top positive-pole words: prominent, including, first, addition, major, notable, noted, also, important, one.
  - Top negative-pole words: valueless, unfocused, uninteresting, inoffensive, pathetically, inattentive, unimaginative, unimportant, inconsequential, inarticulate.

### G_GOAL_DIRECTED — DEPRECATED (was the G axis through exp68)

- **Status:** Deprecated as of exp69. Replaced by INTENTION_CLEAN. Anchor pairs contained attentional content (oriented/targeted) that cross-bled with A's domain. Most of G's content was genuine policy-precision (cos(INT_CLEAN, G_orig) = +0.59); the deprecation is for the attentional-bleed correction, not because G was wrong.
- **Constructed:** exp53 (`exp53_residual_and_goal_directed.py`, `TARGET_GOAL_DIRECTED_PAIRS`).
- **Anchor vocabulary (14 pairs, 14/14 in vocab):**
  ```
  (pursuing, idling)        (aiming, wandering)        (purposeful, aimless)
  (deliberate, accidental)  (motivated, unmotivated)   (intentional, unintentional)
  (resolute, hesitant)      (committed, uncommitted)   (driven, becalmed)
  (oriented, disoriented)   (targeted, untargeted)     (decided, undecided)
  (chasing, dawdling)       (ambitious, complacent)
  ```
- **Original semantic description:** commitment-to-specific-outcome vs uncommitted-action / agency-impairment. Operationalizes policy precision (π) at the meta-level (commitment-to-having-a-policy), not at the sub-decision level (explore/exploit).
- **Validation against Lakoff PCs (pre-screening, for reproducibility):**
  - `cos(G, PC2) = −0.303` — strongest target-axis-to-named-PC hit in the project.
  - PC1-residualization leaves PC2 alignment essentially unchanged: `cos(G_res, PC2) = −0.304` (magnitude retained 99.6%).
  - Top input-axis loadings: BAL +0.546, COHERENCE +0.528, FB +0.429, UD +0.405, AROUSAL +0.390, EXIST +0.368.
  - Positive pole: aim, aims, launched, committed, pursue, promote, strategy, development, against, responsible, aimed, establish.
  - Negative pole: mangxamba, cw96, disoriented, disorientated, inebriated, _____________, tipsy, b***@chron.com, awestruck, a.k.a, rw97, mo96 (note: agency-impaired vocabulary mixed with tokenization junk).

---

## Section 2 — Inter-axis orthogonality (final 7-axis basis)

From `results_exp60.txt`, RAW cosines between the 7 basis axes (before Gram-Schmidt):

|        | C_rew | W_wgt | A_aff | G_pol | R_per | D_cmp | IO_blk |
|--------|------|------|------|------|------|------|------|
| C_rew  | 1.000 | −0.344 | +0.008 | +0.050 | −0.050 | +0.183 | +0.108 |
| W_wgt  | −0.344 | 1.000 | +0.233 | +0.248 | +0.009 | +0.135 | +0.024 |
| A_aff  | +0.008 | +0.233 | 1.000 | +0.561 | +0.113 | +0.023 | −0.015 |
| G_pol  | +0.050 | +0.248 | +0.561 | 1.000 | +0.088 | +0.088 | +0.110 |
| R_per  | −0.050 | +0.009 | +0.113 | +0.088 | 1.000 | +0.051 | −0.114 |
| D_cmp  | +0.183 | +0.135 | +0.023 | +0.088 | +0.051 | 1.000 | +0.086 |
| IO_blk | +0.108 | +0.024 | −0.015 | +0.110 | −0.114 | +0.086 | 1.000 |

Off-diagonal magnitudes worth flagging:

- **A–G = +0.561** (the persistent affect-policy entanglement; see Section 7).
- **C–W = −0.344** (expected: value and cost anti-correlated; structurally meaningful).
- **W–A = +0.233**, **W–G = +0.248** (weight has moderate positive correlation with both affect and goal-directedness; consistent with the cost-of-effort reading).
- **C–D = +0.183** (predictable things are mildly reward-correlated in natural language).
- **R–IO = −0.114** (mild — IO_CLEAN is the cleanest member of the basis).
- All other pairs `|cos| < 0.15`.

For reference, the 6D pre-IO inter-axis matrix from `results_exp57.txt`:

|        | C_rew | A_aff | D_cmp | G_pol | R_per | EE_x |
|--------|------|------|------|------|------|------|
| C_rew  | 1.000 | +0.008 | +0.183 | +0.050 | −0.050 | +0.037 |
| A_aff  | +0.008 | 1.000 | +0.023 | +0.561 | +0.113 | −0.187 |
| D_cmp  | +0.183 | +0.023 | 1.000 | +0.088 | +0.051 | +0.118 |
| G_pol  | +0.050 | +0.561 | +0.088 | 1.000 | +0.088 | −0.139 |
| R_per  | −0.050 | +0.113 | +0.051 | +0.088 | 1.000 | +0.004 |
| EE_x   | +0.037 | −0.187 | +0.118 | −0.139 | +0.004 | 1.000 |

---

## Section 3 — Lakoff PC validation (target-axes vs cluster PCs)

PCA was run on the 11 cluster input axes (VALENCE, AROUSAL, UD, FB, LD, PATH, EXIST, BAL, COHERENCE, SUCCESS, LOSS) — matches exp40's cluster-only run. Variance per PC: PC1 20.6%, PC2 17.3%, PC3 12.4%, PC4 11.2%, PC5 9.3%, PC6 8.7% (cumulative 79.6%).

### Master target-axis × PC table (all targets, raw cosines)

|                       | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 |
|-----------------------|------|------|------|------|------|------|
| target_SALIENCE       | −0.174 | −0.280 | −0.056 | +0.024 | −0.058 | −0.039 |
| target_MOTION         | −0.094 | +0.051 | +0.022 | +0.054 | +0.017 | −0.072 |
| target_GOAL_DIRECTED  | −0.088 | **−0.303** | +0.039 | +0.179 | −0.034 | −0.006 |
| target_EQ_RUN         | +0.290 | +0.011 | **+0.205** | −0.073 | −0.124 | −0.042 |
| A_RUSSELL_DIAGONAL    | −0.174 | −0.280 | −0.056 | +0.024 | −0.058 | −0.039 |
| B_VALUE_PURE          | +0.321 | −0.150 | −0.043 | −0.002 | −0.043 | −0.108 |
| **C_REWARD_COMPOSITE**| **+0.544** | +0.078 | +0.006 | +0.090 | −0.105 | +0.067 |
| D_SURPRISAL           | +0.204 | −0.245 | +0.223 | +0.198 | +0.119 | +0.040 |
| EE_EXPLOIT_EXPLORE    | +0.119 | +0.064 | −0.017 | −0.098 | −0.022 | −0.037 |
| WEIGHT                | **−0.381** | −0.164 | −0.021 | +0.162 | +0.144 | +0.050 |

Note: target_SALIENCE and A_RUSSELL_DIAGONAL share the same anchor pairs and so have identical cosines.

### PC1 — integrated reward axis (exp54 comparator)

|cos with PC1|, sorted:

| candidate           | \|cos(PC1)\| | signed |
|---------------------|-------------|--------|
| C_REWARD_COMPOSITE  | **0.544**   | +0.544 |
| WEIGHT              | 0.381       | −0.381 |
| B_VALUE_PURE        | 0.321       | +0.321 |
| target_EQ_RUN       | 0.290       | +0.290 |
| D_SURPRISAL         | 0.204       | +0.204 |
| target_SALIENCE / A_RUSSELL | 0.174 | −0.174 |
| EE_EXPLOIT_EXPLORE  | 0.119       | +0.119 |
| target_MOTION       | 0.094       | −0.094 |
| target_GOAL_DIRECTED| 0.088       | −0.088 |

**Verdict:** C_REWARD_COMPOSITE wins, but only "moderate capture" (0.4 < |cos| < 0.7) — PC1 has additional content (the W cost-pole at −0.381 is the other half of the expected-free-energy reading).

### Inter-candidate orthogonality matrix (exp54)

|                       | A_RUSSELL | B_VALUE_PURE | C_REWARD | D_SURPRISAL |
|-----------------------|-----------|--------------|----------|-------------|
| A_RUSSELL_DIAGONAL    | 1.000     | +0.387       | +0.008   | +0.023      |
| B_VALUE_PURE          | +0.387    | 1.000        | +0.326   | +0.073      |
| C_REWARD_COMPOSITE    | +0.008    | +0.326       | 1.000    | +0.183      |
| D_SURPRISAL           | +0.023    | +0.073       | +0.183   | 1.000       |

A, C, D nearly-orthogonal pairwise — the empirical signature of Niamh's multi-loss hypothesis.

### PC2 — policy precision axis (exp53)

| target               | cos(PC2) | post-PC1-residualization |
|----------------------|----------|--------------------------|
| target_GOAL_DIRECTED | **−0.303** | −0.304 (99.6% retained) |
| target_SALIENCE      | −0.280   | −0.284                   |
| D_SURPRISAL          | −0.245   | —                        |
| WEIGHT               | −0.164   | —                        |
| B_VALUE_PURE         | −0.150   | —                        |
| EE_EXPLOIT_EXPLORE   | +0.064   | —                        |
| target_MOTION        | +0.051   | +0.051                   |
| target_EQ_RUN        | +0.011   | +0.012                   |

G wins cleanly. EE failed to capture PC2 (cos=+0.064); MOTION failed (cos=+0.051).

### PC3 — perceptual precision axis (exp53)

| target                       | cos(PC3) | post-PC1-residualization | post-V+A-residualization |
|------------------------------|----------|--------------------------|--------------------------|
| target_EQ_RUN                | **+0.205** | **+0.214** (95.7% retained) | **+0.103** (94.2% retained) |
| D_SURPRISAL                  | +0.223   | —                        | —                        |
| target_GOAL_DIRECTED         | +0.039   | +0.039                   | —                        |
| target_SALIENCE              | −0.056   | −0.057                   | —                        |

Partial validation — PC3 alignment survives PC1-residualization cleanly but is dented by V+A-residualization (the "regulation" pole is structurally calm, the "runaway" pole structurally aroused in natural language).

---

## Section 4 — Lakoff-schema decomposition over the 7-axis basis

### 6D (no W) and 7D (with W, but without IO_CLEAN as basis vector) explained-variance

From `results_exp59.txt` (test 6 — 6D basis is C/A/D/R/EE/G; 7D adds W):

| schema    | 6D expl | 7D expl | Δ      |
|-----------|---------|---------|--------|
| UD        | 64.4%   | 65.5%   | +1.1%  |
| FB        | 48.6%   | 48.8%   | +0.2%  |
| LD        | 41.8%   | 46.1%   | +4.3%  |
| IO_CLEAN  | 21.8%   | 22.0%   | +0.2%  |
| PATH      | 31.2%   | 31.2%   | +0.0%  |
| EXIST     | 53.5%   | 54.2%   | +0.7%  |
| FORCE     | 21.7%   | 21.7%   | +0.1%  |
| BAL       | 63.7%   | 65.2%   | +1.5%  |
| **DIFF**  | **36.4%** | **56.1%** | **+19.7%** |
| COH       | 70.8%   | 70.9%   | +0.1%  |
| SUC       | 45.8%   | 48.2%   | +2.4%  |
| LOSS      | 53.5%   | 53.5%   | +0.1%  |

DIFF jumps from 36.4% → 56.1% on adding W. No other schema gains more than 4.3% from the W addition. Confirms: DIFFICULTY-BURDEN is essentially the weight axis plus a smaller reward component.

### 6D-vs-7D (with IO_CLEAN as basis vector) — final pass from exp60

From `results_exp60.txt` (6D = C/W/A/D/R/G; 7D = adding IO_blk):

| schema    | 6D expl | 7D expl | Δ      |
|-----------|---------|---------|--------|
| UD        | 65.5%   | 65.9%   | +0.4%  |
| FB        | 48.8%   | 48.9%   | +0.2%  |
| LD        | 46.0%   | 46.1%   | +0.2%  |
| PATH      | 29.8%   | 33.7%   | +3.9%  |
| EXIST     | 54.1%   | 54.3%   | +0.2%  |
| FORCE     | 21.7%   | 21.7%   | +0.0%  |
| BAL       | 64.7%   | 66.9%   | +2.2%  |
| DIFF      | 56.1%   | 56.2%   | +0.1%  |
| COH       | 70.8%   | 70.8%   | +0.0%  |
| SUC       | 47.3%   | 47.4%   | +0.1%  |
| LOSS      | 53.4%   | 54.6%   | +1.2%  |
| VALENCE   | 63.0%   | 63.2%   | +0.3%  |
| AROUSAL   | 58.0%   | 58.6%   | +0.6%  |

### Final 7D Gram-Schmidt coordinates for each Lakoff schema (exp60)

GS order: C_rew, W_wgt, IO_blk, A_aff, D_cmp, R_per, G_pol.

| schema    | C_rew  | W_wgt  | IO_blk | A_aff  | D_cmp  | R_per  | G_pol  | expl  |
|-----------|--------|--------|--------|--------|--------|--------|--------|-------|
| UD        | +0.478 | +0.019 | +0.106 | +0.368 | +0.012 | −0.091 | +0.227 | 65.9% |
| FB        | +0.183 | +0.096 | +0.058 | +0.350 | +0.040 | +0.064 | +0.255 | 48.9% |
| LD        | +0.208 | −0.090 | −0.040 | +0.319 | +0.068 | +0.144 | +0.182 | 46.1% |
| IO_CLEAN  | +0.108 | +0.066 | +0.992 | +0.000 | −0.000 | +0.000 | +0.000 | 100%  |
| PATH      | +0.139 | +0.067 | −0.133 | +0.143 | −0.048 | −0.048 | +0.217 | 33.7% |
| EXIST     | +0.370 | +0.035 | +0.068 | +0.331 | +0.001 | −0.024 | +0.206 | 54.3% |
| FORCE     | +0.135 | +0.017 | +0.012 | +0.083 | −0.135 | −0.041 | +0.042 | 21.7% |
| BAL       | +0.277 | +0.075 | +0.220 | +0.381 | +0.199 | −0.007 | +0.362 | 66.9% |
| DIFF      | −0.284 | **+0.430** | +0.010 | −0.070 | +0.097 | +0.110 | −0.155 | 56.2% |
| COH       | +0.271 | +0.209 | +0.050 | +0.384 | +0.375 | −0.030 | +0.304 | 70.8% |
| SUC       | +0.400 | −0.118 | +0.029 | +0.185 | −0.073 | −0.020 | +0.097 | 47.4% |
| LOSS      | +0.479 | +0.048 | +0.132 | +0.076 | +0.099 | +0.087 | +0.161 | 54.6% |
| VALENCE   | +0.530 | −0.162 | +0.083 | +0.224 | +0.120 | −0.038 | +0.143 | 63.2% |
| AROUSAL   | −0.232 | +0.216 | +0.084 | +0.451 | −0.107 | −0.045 | +0.136 | 58.6% |

IO_CLEAN's GS-coordinates are degenerate (1.0 on IO_blk axis) because IO_CLEAN IS a basis vector; it's listed for completeness.

### Dominant-axis summary

From `results_exp55.txt` (5D AI-plus space, no W or IO):

| schema     | tier      | dominant axis              | basis-explained |
|------------|-----------|----------------------------|-----------------|
| UD         | cluster   | C_reward (+0.478)          | 64.4%           |
| FB         | cluster   | G_policy_prec (+0.429)     | 48.6%           |
| LD         | cluster   | G_policy_prec (+0.299)     | 41.5%           |
| IO_CLEAN   | Tier-2    | R_percept_prec (−0.114)    | 21.8%           |
| PATH       | cluster   | G_policy_prec (+0.258)     | 29.8%           |
| EXIST      | cluster   | C_reward (+0.370)          | 53.5%           |
| FORCE      | Tier-2    | C_reward (+0.135)          | 21.7%           |
| BAL        | cluster   | G_policy_prec (+0.546)     | 63.3%           |
| DIFF       | Tier-2    | C_reward (−0.284)          | 36.4%           |
| COH        | cluster   | G_policy_prec (+0.528)     | 70.7%           |
| SUC        | cluster   | C_reward (+0.400)          | 44.6%           |
| LOSS       | cluster   | C_reward (+0.479)          | 53.3%           |
| VALENCE    | cluster   | C_reward (+0.530)          | 57.4%           |
| AROUSAL    | cluster   | A_affect (+0.486)          | 57.3%           |

Cluster schemas land 40–70% basis-explained. Tier-2 schemas (IO_CLEAN, FORCE) ~22%. DIFF intermediate at 36% — and then 56% after adding W.

### W's cosines with each Lakoff schema (exp59)

| schema    | cos(W, schema) |
|-----------|---------------|
| DIFF      | **+0.502**     |
| SUC       | −0.249         |
| LD        | −0.156         |
| UD        | −0.147         |
| LOSS      | −0.120         |
| COH       | +0.103         |
| EXIST     | −0.094         |
| FORCE     | −0.031         |
| FB        | +0.027         |
| BAL       | −0.025         |
| IO_CLEAN  | +0.024         |
| PATH      | +0.015         |

DIFF is the standout. All other schemas have `|cos(W, schema)| < 0.25`.

---

## Section 5 — Concept-word projections

### Final 7-axis basis (exp60, Gram-Schmidt order: C, W, IO, A, D, R, G)

#### Agency + clinical states

| word         | C_rew  | W_wgt  | IO_blk | A_aff  | D_cmp  | R_per  | G_pol  | expl  |
|--------------|--------|--------|--------|--------|--------|--------|--------|-------|
| hope         | +0.070 | +0.092 | −0.061 | +0.272 | −0.062 | +0.061 | +0.268 | 41.3% |
| freedom      | +0.024 | +0.095 | −0.012 | +0.208 | −0.006 | +0.027 | +0.225 | 32.3% |
| agency       | −0.048 | +0.192 | +0.001 | +0.252 | +0.078 | +0.014 | +0.190 | 38.1% |
| growth       | +0.057 | +0.139 | −0.012 | +0.195 | −0.089 | −0.019 | +0.236 | 35.3% |
| play         | +0.004 | +0.085 | −0.080 | +0.191 | +0.063 | +0.007 | +0.210 | 31.3% |
| love         | +0.085 | +0.003 | −0.032 | +0.217 | −0.029 | −0.074 | +0.133 | 28.2% |
| psychosis    | −0.076 | +0.018 | +0.068 | −0.104 | −0.083 | +0.026 | −0.041 | 17.5% |
| trauma       | −0.233 | +0.020 | −0.050 | +0.132 | −0.045 | −0.008 | −0.040 | 27.9% |
| depression   | −0.175 | +0.108 | +0.051 | +0.081 | +0.022 | −0.051 | −0.015 | 23.4% |
| grief        | −0.157 | +0.027 | −0.057 | −0.012 | −0.089 | −0.077 | −0.028 | 20.8% |
| anxiety      | −0.182 | +0.016 | +0.024 | +0.050 | +0.024 | −0.064 | +0.005 | 20.2% |
| fear         | −0.176 | +0.189 | −0.006 | +0.093 | −0.021 | −0.112 | +0.175 | 34.5% |
| panic        | −0.189 | +0.047 | −0.025 | +0.062 | −0.067 | **−0.208** | +0.027 | 30.1% |
| shame        | −0.158 | **+0.201** | −0.109 | −0.067 | −0.105 | −0.036 | +0.076 | 31.6% |
| guilt        | −0.123 | +0.142 | +0.000 | −0.063 | −0.033 | +0.109 | +0.045 | 23.3% |
| burnout      | −0.075 | −0.047 | −0.127 | −0.143 | +0.068 | −0.105 | −0.077 | 25.7% |
| dissociation | −0.041 | −0.120 | −0.053 | −0.048 | −0.014 | +0.102 | +0.011 | 17.9% |

#### Affective + contemplative

| word          | C_rew  | W_wgt  | IO_blk | A_aff  | D_cmp  | R_per  | G_pol  | expl  |
|---------------|--------|--------|--------|--------|--------|--------|--------|-------|
| joy           | +0.102 | −0.112 | −0.082 | +0.092 | −0.067 | −0.072 | +0.071 | 23.0% |
| rage          | −0.189 | +0.045 | +0.017 | −0.035 | −0.004 | **−0.249** | +0.092 | 33.2% |
| compassion    | +0.009 | −0.039 | −0.122 | +0.072 | −0.042 | +0.057 | +0.085 | 18.4% |
| boredom       | −0.124 | −0.010 | −0.013 | −0.211 | +0.076 | −0.095 | −0.064 | 28.1% |
| curiosity     | +0.095 | −0.136 | +0.017 | −0.091 | −0.035 | −0.053 | +0.096 | 22.2% |
| awe           | +0.134 | −0.079 | −0.005 | −0.067 | −0.111 | −0.048 | +0.039 | 21.2% |
| sublime       | +0.108 | **−0.214** | −0.015 | −0.061 | −0.138 | −0.081 | +0.092 | 30.9% |
| transcendence | +0.092 | −0.147 | +0.002 | −0.123 | −0.135 | +0.101 | +0.025 | 27.2% |
| ineffable     | +0.078 | **−0.197** | +0.016 | −0.189 | −0.177 | +0.036 | −0.117 | 35.7% |
| presence      | +0.068 | +0.202 | +0.098 | +0.179 | −0.023 | +0.025 | +0.118 | 32.0% |
| ritual        | +0.042 | +0.146 | +0.090 | +0.076 | **+0.226** | +0.112 | +0.040 | 32.0% |
| meditation    | +0.062 | −0.037 | +0.044 | +0.047 | +0.128 | −0.037 | +0.034 | 16.9% |
| creativity    | +0.125 | −0.070 | −0.026 | +0.046 | −0.083 | −0.018 | +0.186 | 25.5% |
| flow          | −0.059 | +0.023 | +0.067 | +0.102 | +0.070 | −0.102 | +0.144 | 23.5% |

#### IO_CLEAN probes (self / other / boundary content)

| word         | C_rew  | W_wgt  | IO_blk | A_aff  | D_cmp  | R_per  | G_pol  | expl  |
|--------------|--------|--------|--------|--------|--------|--------|--------|-------|
| self         | +0.052 | +0.137 | +0.014 | +0.061 | +0.049 | +0.141 | +0.295 | 36.7% |
| selfhood     | −0.040 | −0.203 | +0.063 | −0.239 | −0.087 | −0.005 | −0.064 | 34.0% |
| identity     | −0.041 | +0.137 | +0.067 | +0.163 | +0.065 | +0.099 | +0.149 | 29.7% |
| boundary     | −0.096 | +0.072 | +0.071 | +0.100 | +0.093 | +0.007 | +0.045 | 20.1% |
| intimacy     | +0.127 | −0.168 | +0.036 | −0.073 | −0.123 | +0.099 | +0.081 | 28.7% |
| belonging    | −0.090 | +0.136 | **+0.294** | +0.145 | −0.016 | −0.016 | +0.058 | 37.2% |
| alienation   | −0.134 | −0.001 | −0.017 | −0.058 | −0.096 | −0.060 | −0.046 | 19.1% |
| isolation    | −0.114 | +0.131 | +0.079 | +0.065 | −0.090 | −0.025 | +0.076 | 23.5% |
| communion    | −0.006 | −0.074 | +0.056 | +0.057 | +0.037 | +0.043 | −0.052 | 13.3% |
| exile        | −0.050 | +0.122 | +0.062 | +0.047 | −0.050 | +0.004 | +0.128 | 20.6% |
| membership   | +0.118 | +0.236 | −0.011 | +0.114 | +0.014 | +0.042 | +0.119 | 31.4% |
| kinship      | +0.062 | +0.080 | +0.001 | −0.003 | −0.041 | +0.054 | −0.076 | 14.3% |
| loneliness   | −0.120 | −0.038 | +0.007 | −0.120 | −0.068 | −0.074 | −0.059 | 20.9% |
| togetherness | +0.099 | −0.071 | +0.001 | −0.092 | −0.029 | +0.031 | −0.036 | 16.2% |

Only **belonging** loads strongly on IO_blk (+0.294). Other self/other concepts load weakly — see Section 7.

#### Random nouns (control)

| word     | C_rew  | W_wgt  | IO_blk | A_aff  | D_cmp  | R_per  | G_pol  | expl  |
|----------|--------|--------|--------|--------|--------|--------|--------|-------|
| sausage  | +0.051 | +0.003 | +0.084 | −0.033 | +0.022 | −0.104 | −0.094 | 17.6% |
| pyjamas  | +0.006 | −0.146 | −0.044 | −0.169 | +0.032 | −0.032 | −0.091 | 24.9% |
| marigold | +0.031 | −0.160 | −0.037 | −0.215 | +0.000 | −0.135 | −0.119 | 32.6% |
| stapler  | −0.038 | −0.132 | +0.118 | −0.147 | +0.058 | −0.014 | −0.111 | 26.5% |
| pebble   | +0.061 | +0.037 | −0.059 | −0.079 | +0.010 | +0.019 | +0.048 | 13.2% |

From exp58 (6D, no W/IO): accordion 22.5%, lamppost 27.0%, casserole 19.0%. Random nouns sit higher than expected — they lean negative on A_aff and G_pol, the "passive low-arousal object" signature. **Pebble at 10.5–13.2% is the cleanest baseline.**

### Anxiety / fear / panic comparison (from exp58 6D space)

| word       | C_rew  | A_aff  | D_cmp  | R_per  | EE_x   | G_pol  | expl  |
|------------|--------|--------|--------|--------|--------|--------|-------|
| anxiety    | −0.182 | +0.052 | +0.025 | −0.066 | +0.087 | **+0.011** | 22.0% |
| fear       | −0.176 | +0.138 | +0.014 | −0.119 | −0.083 | **+0.191** | 32.8% |
| panic      | −0.189 | +0.073 | −0.061 | **−0.207** | −0.032 | +0.025 | 29.8% |
| trauma     | −0.233 | +0.134 | −0.050 | −0.003 | +0.037 | −0.043 | 27.9% |
| depression | −0.175 | +0.104 | +0.043 | −0.061 | +0.013 | +0.005 | 21.7% |
| grief      | −0.157 | −0.003 | −0.084 | −0.073 | +0.099 | −0.022 | 21.8% |
| psychosis  | −0.076 | −0.098 | −0.068 | +0.015 | +0.149 | −0.012 | 20.6% |

All low-wellbeing (negative C). Distinguished by G (fear has a policy, anxiety doesn't) and by R (panic has the strongest precision collapse).

---

## Section 6 — Anchors that didn't make it

### A_RUSSELL_DIAGONAL — lost the PC1 comparator

- **Construction:** exp52 / exp54 (same as final A axis, used as a PC1 candidate).
- **Cosine with PC1:** −0.174.
- **Why it failed:** PC1's input-axis loadings (V +0.69, A −0.65, SUC +0.50, LOSS +0.50, UD +0.43) form a composite reward signal, not a pure V×A diagonal. The Russell-diagonal anchors instead landed on PC2 (cos = −0.280) because affect quality is a different primitive than integrated reward.
- **Negative-result content:** affect-as-felt-state and wellbeing-as-integrated-state are independent in word vectors (`cos(A_RUSSELL, C_REWARD) = +0.008`). The Russell axis still survived as the A axis in the basis — it just wasn't PC1.

### B_VALUE_PURE — also lost

- **Construction:** exp54. Anchors: preferred/dispreferred, wanted/unwanted, sought/shunned, cherished/loathed, loved/hated, treasured/abhorred, valued/devalued, welcomed/rebuffed, embraced/rejected, approached/avoided, adored/detested, admired/scorned (11/12 in vocab — `dispreferred` OOV).
- **Cosine with PC1:** +0.321 (2nd place, well behind C at +0.544).
- **Why it failed:** pure-preference vocabulary doesn't capture PC1's composite-reward shape. B has high VALENCE loading (+0.509) and high UD (+0.496) but misses the LOSS, SUCCESS, EXIST loadings that C captures. Inter-target cos(B, C) = +0.326 — they overlap but aren't the same.
- **Negative-result content:** PC1 is integrated wellbeing, not pure preference. Preference is a subspace of reward, not the whole thing.

### EE_EXPLOIT_EXPLORE — failed PC2

- **Construction:** exp56. Anchors: exploit/explore, harvest/forage, specialize/diversify, entrench/venture, optimize/experiment, routine/novelty, consolidate/branch, refined/exploratory, perfecting/probing, capitalize/prospect, rehearsing/discovering, mastery/investigation.
- **Cosine with PC2:** +0.064 (vs G at −0.303).
- **Why it failed:** EE is at a different level of abstraction from PC2. PC2 = policy precision (commitment-to-having-a-policy). Explore-vs-exploit is a sub-decision within having a policy. EE has lower correlation with A_affect (0.187 vs G's 0.561, cleaner) but doesn't capture PC2 at all.
- **Status:** EE was used in the 6D basis (exp55–57) but dropped in exp60. Niamh's call: it was capturing "corporate-optimization vs scientific-investigation register" rather than a cognitive primitive. EE's positive pole (entrench, optimize, microkernels, industrialise, trichogramma) confirms the register-artifact reading.
- **Negative-result content:** EE is a real conceptual axis orthogonal to G in word vectors (cos(EE, G) = −0.139, cos(EE, A) = −0.187, cos(EE, R) = +0.004) — it just lives outside cluster-PCA space.

### target_SALIENCE — refuted PC1 = salience

- **Construction:** exp52 (Niamh's valence-balanced "important / unimportant / urgent / salient / attentive / focused / directed / prominent / noticeable / foregrounded / highlighted / conspicuous / pronounced" set).
- **Cosine with PC1:** −0.174.
- **Why it failed as PC1:** the valence-orthogonal salience anchors distribute across multiple input axes (AROUSAL +0.486, COHERENCE +0.425, BAL +0.383, FB +0.363, UD +0.361) — no single PC home. Best PC alignment is PC2 at −0.280.
- **Negative-result content:** PC1 is NOT salience-as-primitive (orthogonal-to-valence). Salience-without-valence may not be a coherent linguistic concept — Niamh's "salience without valence doesn't exist and neither does valence without salience" observation. The same anchor pairs were retained as the A axis (Russell V×A diagonal).

### target_MOTION — refuted PC2 = motion

- **Construction:** exp52. Anchors: running/sitting, jogging/standing, swimming/lying, flying/crouching, skating/kneeling, sliding/perching, rolling/slouching, leaping/leaning, vaulting/lounging, springing/reclining, lunging/hunching, galloping/slumping.
- **Cosine with PC2:** +0.051.
- **Why it failed:** PC2's nearest-neighbor words on the negative pole include *must, ensure, unable, failure* — modal/constraint/policy vocabulary, not stasis vocabulary. PC2 was never about locomotion-vs-stasis. The locomotion-positive pole even pulled to sports register (skiing, racing, sprint, championships).
- **Negative-result content:** PC2 is goal-directedness / policy precision, not motion. The original "PC2 = motion" naming in exp40 was an inferred-from-nearest-neighbors error.

### Caveat on target_EQ_RUN

Survives as R but with caveats — see Section 7.

---

## Section 7 — Open methodological questions

### A–G entanglement at +0.56 (the persistent affect-policy correlation)

> **Update (2026-05-28): RESOLVED — reading (b), anchor-vocabulary bias.** exp69 cross-bleed screening produced ATTENTION_CLEAN (anchors screened to exclude intentional content) and INTENTION_CLEAN (anchors screened to exclude attentional content). **cos(ATTENTION_CLEAN, INTENTION_CLEAN) = −0.13** vs the original +0.561. The −0.69 swing established that the entanglement was anchor-vocabulary bias, not substrate-real coupling. The cross-bleed screening protocol generalizes — applicable to any contrast-pair axis construction. See ATT_CLEAN and INT_CLEAN entries in §1 above and WRITEUP_v4 §5.1.

- `cos(target_GOAL_DIRECTED, A_RUSSELL_DIAGONAL) = +0.561`.
- target_GOAL_DIRECTED's positive pole has "aim, aims, launched, committed, pursue, promote, strategy, development, against, responsible" — formal/strategic register. Negative pole has "disoriented, inebriated, tipsy, awestruck" mixed with tokenization junk (cw96, mangxamba, b***@chron.com).
- Possible readings (from lab notebook, written pre-exp69): (a) affect and policy-precision are causally coupled in cognition / language; (b) our G anchors introduced spurious correlation from register-bias (strategic vocabulary is calm-positive in the corpus); (c) both.
- Niamh's speculation: PC2 may be capturing "the literal transformation within the algorithm, from tokens to vectors" — frequency / lexical-specificity meta-signal mixed in. (Speculation overtaken by the exp69 resolution above.)

### IO_CLEAN captures only spatial-container vocabulary, not abstract self/other

> **Update (2026-05-28): ADDRESSED by adding MB (Markov-blanket) as a complementary axis.** exp64 Part B built `target_MARKOV_BLANKET` from self/other, agent/environment, internal/external, mine/theirs, own/foreign, private/public, subjective/objective, introspection/perception, autonomous/dependent, individual/collective, personal/impersonal, intrinsic/extrinsic, endogenous/exogenous. MB is the **cleanest member of the basis** (max |cos| with any other axis = +0.137, with IO_CLEAN). On 21 self/other-relevant probes, MB has larger |cos| than IO on 15 of 21 (self +0.32, ego +0.18, soul +0.19, identity +0.12, autonomy +0.20, selfhood +0.16, ...). IO and MB are complementary: IO captures spatial-container topology; MB captures abstract agent/environment partitioning. See MB entry in §1 above. — The target_SELECTION probe (the other proposal in the original note) was constructed as exp63 and ultimately renamed to DV (decision-verdict) after exp64 Part A's pole-vocabulary analysis showed it captures verdict-outcomes rather than active-inference selection. cos(DV, IO_CLEAN) = +0.017 — DV is its own thing.

- IO_blk's loadings on IO-probe concept words (exp60): only **belonging at +0.294** is strong. Other self/other concepts load weakly: self +0.014, selfhood +0.063, identity +0.067, boundary +0.071, intimacy +0.036, alienation −0.017, isolation +0.079, communion +0.056, exile +0.062, membership −0.011, kinship +0.001, loneliness +0.007, togetherness +0.001.
- IO_CLEAN was built from spatial-containment anchors (inside/outside, contained/released, enter/exit, etc.). It captures the **container-topology version** of in/out — not the **abstract phenomenal Markov-blanket** version.
- Open: to test the full Markov-blanket reading, a separate target axis built from self/other/agent/environment vocabulary is needed. Lab-notebook proposes target_SELECTION (selected/rejected, chose/refused, picked/discarded, ...) and target_MARKOV_BLANKET (self/other, agent/environment, internal/external) as Sub-thread C tests. (Both tests have been run — see Update note above.)

### W → COST reframe (computation prior to soma)

> **Update (2026-05-27/28): TESTED, INCONCLUSIVE at original thresholds.** exp72 built target_COST from non-somatic anchors: (expensive, free), (costly, cheap), (demanding, easy), (depleting, sustaining), (consuming, replenishing), (extracting, conserving), (effortful, trivial), (draining, renewing), (intensive, minimal). **Result: cos(COST, W) = +0.289.** This falls between the original thresholds: not "same axis" (would need > 0.70), not "separate primitive" (would need < 0.20). COST and W point in measurably different directions; W carries some cost-flavored content but the axes are not the same. Secondary: cos(COST, DIFF) = +0.329 (vs cos(W, DIFF) = +0.502 — DIFF lexicalizes preferentially as somatic burden, not as process cost). COST's negative pole is anisotropic (generic-positive register rather than clean anti-cost anchors), so the construction is partially undercooked. The cognitive-theory question ("computation prior to soma") is **deferred** to substrate-invariance + causal validation; the single-substrate inconclusive result doesn't license a strong directional claim in either direction. See WRITEUP_v3 Part 5 (the v3 section is the most detailed post-exp72 write-up; v4 reframes it as one of multiple deferred questions, not the central theoretical question).

- W was operationalized via somatic vocabulary (heavy/light, burdensome/effortless, encumbered/unencumbered). But:
  - W's positive pole includes the explicit lexical anchor "costly" (3rd in nearest-neighbor lookup after heavy and burden).
  - cos(W, C_REWARD) = −0.344 — W is specifically anti-reward (cost), not generic anti-anything.
  - cos(W, DIFF) = +0.502 — difficulty in Lakoff = high computational cost.
- Niamh's reframe: the structural primitive being recovered is **cost as computational quantity**, not weight as somatic primitive. Phenomenological felt-heaviness is one of cost's expressive modalities. Active-inference framing: expected free energy = cost − value, so PC1 might literally be the C-vs-W axis.
- Proposed test (Thread B, exp60 backlog): build target_COST from non-somatic anchors — (expensive, free), (taxing, refreshing), (demanding, easy), (laborious, automatic), (depleting, sustaining), (consuming, replenishing), (extracting, conserving), (taxing, restorative). Compare:
  - cos(COST, W) — high if computational and somatic are aspects of the same primitive.
  - cos(COST, DIFF) — should be ≈ +0.50 if cost IS what DIFF tracks.
  - cos(COST, A_affect) — should be lower than W's +0.233 if non-somatic vocabulary reduces affect contamination.
  - (Test was run as exp72 — see Update note above.)

### PC3 regulability survives PC1-residualization but is dented by V+A-residualization

> **Update (2026-05-28): STILL OPEN.** The cross-bleed screening protocol that resolved A-G (exp69) wasn't applied to R directly; R still carries some affect-loading after V+A-residualization. The cleaner-R-construction proposal (homeostat / feedback-control / dynamical-systems vocabulary) hasn't been built. R's geometric prediction against asymmetric collapse held at MML schema level (exp61b: cos(LD, R) = +0.178), but the affect-bound issue is partially structural — the regulation pole is calm-positive in language by default. **Causal validation in Pythia (Thread 6) is the load-bearing test** for whether R captures perceptual-precision content or whether it's substantially affect-content; behavior should differ if R is real. Listed as Thread 4 in WRITEUP_v4 §7.4 and as a candidate for cleaner reconstruction.

- Raw `cos(target_EQ_RUN, PC3) = +0.205`.
- After PC1-residualization: +0.214 (essentially unchanged).
- After V+A-residualization: +0.103 (cut in half).
- Interpretation: regulability is partially affect-independent — there IS real signal after stripping V+A — but the regulation pole is structurally calm-positive in language and the runaway pole structurally aroused-negative. R can't be fully separated from affect in distributional semantics.
- Open: try a cleaner R construction with non-affect-loaded regulation vocabulary (homeostat / feedback-control / dynamical-systems terms). See lab-notebook Thread 4 (Entry 24).

### Random nouns aren't as null as expected

- Pebble at 10.5–13.2% is the cleanest baseline. But marigold at 31.3–32.6%, pyjamas at 23.7–24.9%, accordion at 22.5%, stapler at 22.7–26.5% all sit in the same range as some clinical states (anxiety 20.2–22.0%, depression 21.7–23.4%). Random nouns lean negative on A_aff (low-arousal) and negative on G_pol (no agency) — they have a passive-low-arousal-object signature.
- Implication: basis-explained percentage isn't a clean "is this in agency-space" metric. Need to consider per-axis coordinates, not just aggregate explained.

### "Language has words for the directions but not the diagonals"

- Russell's V and A axes each have dozens of lexicalizations. The diagonal (pure salience as joint of V and A) has very few words. Same pattern for the AI primitives: many words for outputs (committed, focused, drunk, awestruck), no direct word for "policy precision" itself.
- Methodological warrant: PCA over many anchor pairs is the right tool for finding implicit hyperprior-like primitives that lack direct lexicalization. Direct anchor construction wouldn't work; averaging across many anchor pairs each capturing one output of the variable does.

### Niamh's multi-loss hypothesis

> **Update (2026-05-28): STILL SUPPORTED, with A-G entanglement caveat resolved.** The A-G entanglement that compromised the 7-axis basis's orthogonality is now known to have been anchor-vocabulary bias (exp69 → ATT_CLEAN/INT_CLEAN at cos −0.13). The multi-loss orthogonality finding stands and is strengthened: A (now ATT_CLEAN as perceptual precision γ, with a separate Russell V×A retained where needed), C, D have nearly-orthogonal pairwise cosines, and the basis grew to include EV (epistemic value, exp73) completing the first-order EFE = C − W − EV decomposition. The full active-inference EFE quantity is lexically realized across three independently-constructable axes. The "language tracks multiple independent quantities" claim is supported across the larger basis (max off-diagonal in Cog-13b = 0.348).

- A, C, D have nearly-orthogonal pairwise cosines (+0.008, +0.023, +0.183).
- The empirical signature is consistent with multi-objective views of cognition (Sterling allostasis, Berridge wanting-vs-liking, Bayesian model-evidence-vs-utility) and inconsistent with strong-form single-objective active inference. The data shows language tracks three independent quantities.
- The 7-axis basis is "at least 6-dimensional, mostly mutually-orthogonal" — caveat that A–G entanglement is the persistent issue. (Caveat resolved per Update note above.)

### Substrate generalization

> **Update (2026-05-28): PARTIAL CROSS-SUBSTRATE EVIDENCE.** exp98 (All-But-The-Top in GloVe + fastText) found that **the same procedure surfaces interpretable axes in both substrates at K=1 (phenomenological-vs-institutional) and K=2 (virtue-vs-confusion); substrates diverge at K=3.** This supports the weaker "**methodology-reliable across substrates**" claim, not the stronger "axis-universal" claim. Different vocabulary curation would surface different axes. The full-basis fastText replication (with the 13-axis cognitive basis as anchor constructions) is still pending — it's the load-bearing test for whether the specific axes (HARDNESS dominance, cos(W, DIFF) ≈ +0.50, the EFE structure of PC1, etc.) generalize. Listed as Thread 4 / 5 in WRITEUP_v4 §7.4–7.5.

- All findings are in glove-wiki-gigaword-300 (one substrate). The lab notebook flags fastText within-substrate replication as the next decisive test (Thread A). The substrate-invariance claim from Entry 22 was retracted in Entry 23 after exp48–51 showed PC-rank not preserved cross-substrate (word2vec vs Pythia 70m SAE) — best cross-substrate alignment ~0.7 between non-corresponding PCs.

---

## Files

Code:
- `/Users/macn/Documents/embeddingexp/exp45_diverse_triples.py`
- `/Users/macn/Documents/embeddingexp/exp46_all_pcs.py`
- `/Users/macn/Documents/embeddingexp/exp47_triple_coverage.py`
- `/Users/macn/Documents/embeddingexp/exp48_cross_substrate_pc_comparison.py`
- `/Users/macn/Documents/embeddingexp/exp49_matched_substrate_pca.py`
- `/Users/macn/Documents/embeddingexp/exp50_matched_formal_triples.py`
- `/Users/macn/Documents/embeddingexp/exp51_w2v_vs_formal_sae.py`
- `/Users/macn/Documents/embeddingexp/exp52_target_axis_validation.py`
- `/Users/macn/Documents/embeddingexp/exp53_residual_and_goal_directed.py`
- `/Users/macn/Documents/embeddingexp/exp54_pc1_comparator.py`
- `/Users/macn/Documents/embeddingexp/exp55_lakoff_in_ai_space.py`
- `/Users/macn/Documents/embeddingexp/exp56_explore_exploit.py`
- `/Users/macn/Documents/embeddingexp/exp57_lakoff_in_6d_basis.py`
- `/Users/macn/Documents/embeddingexp/exp58_anxiety_random_nouns.py`
- `/Users/macn/Documents/embeddingexp/exp59_weight.py`
- `/Users/macn/Documents/embeddingexp/exp60_final_basis.py`

Result text dumps:
- `/Users/macn/Documents/embeddingexp/results_exp45.txt` through `results_exp60.txt`

Persisted npz arrays (not loaded for this reference, but exist):
- `exp52_results.npz`, `exp53_results.npz`, `exp54_results.npz`, `exp55_results.npz`, `exp56_results.npz`, `exp57_results.npz`, `exp60_results.npz`

Lakoff anchor vocabulary:
- `/Users/macn/Documents/embeddingexp/lakoff_canonical_vocabulary.py` — defines `UP_DOWN_MML`, `IN_OUT_MML_CLEAN`, `FORWARD_BACK_MML`, `PATH_MOTION_MML`, `LIGHT_DARK_MML`, `EXISTENCE_MML`, `FORCE_MML`, `BALANCE_MML`, `DIFFICULTY_BURDEN_MML`.
