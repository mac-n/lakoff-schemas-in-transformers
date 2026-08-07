# Basis Tests TODO

**STATUS AS OF 2026-05-27 (continued session, post-exp73):** Basis is now 10 axes. Full first-order EFE = C − W − EV decomposition is lexically realized at the basis level.

**Current basis (post-exp73):**
```
C, W, ATTENTION (replaces A), INTENTION (replaces G), R, D, IO_CLEAN, DV, MB, EPISTEMIC_VALUE
```

**Top-priority next moves:**
1. **Doc updates pending:** `BASIS_REFERENCE.md` §2 (now 10×10 cosine matrix), §4 (schema decomposition under 10-axis), §5 (concept-word projections including EV findings). The 10×10 matrix is mostly the exp71 9×9 plus the EV row/col from exp73.
2. **`WRITEUP_v3.md` updates** — §4 (basis description, now 10 axes), §3.3 (PC2 → INT, mention EV completes EFE), §5 (computation-prior-to-soma — exp72 sharpened this: W is somatic-burden distinct from process-COST; both contribute to EFE-minimization).
3. **Optional exp74:** consolidated verification pass with the 10-axis basis (10×10 cosine matrix, schema decomposition, concept-word projections under 10-axis). Replaces exp71's results in the canonical reference.
4. **Thread 2 status:** target_COST run as exp72; result was **separate primitive** (cos with W = +0.29), NOT a replacement for W. Could be added as 11th axis if Niamh wants explicit computational-cost-distinct-from-somatic-weight representation; not added in current basis. See lab notebook Entry 11.
5. **Showerthought parked** (Entry 11): the COST/WEIGHT/EFE/DV unification hunch — revisit when Niamh's intuition crystallizes.

**Major findings still on the table:**
- 9×9 → 10×10 matrix is clean at 0.35 threshold across all pairs (exp71 + exp73)
- fear/anxiety/panic distinction survived basis swap (exp71)
- 4 substantive schema dominant-axis shifts under cleaned basis: AROUSAL→INT, PATH→ATT, EXIST→DV, FORCE→DV (exp71)
- psychosis emerges as Friston precision-collapse signature in distributional semantics: high ATT + low INT (exp71)
- Pattern across exp72 + exp73: theory predicts coupling among EFE components, distributional semantics shows separability. Each component has its own lexical neighborhood. **Calibrate predictions downward in this region.**

---

A running test queue for the embeddingexp basis-validation phase. Run free tests first, then the SELECTION construction (which unlocks the rest), then everything else.

Each test has:
- **What** — one-line description
- **Files** — what data to load
- **Predictions** — what the data should show if the hypothesis holds
- **Interpretation** — what to conclude from each outcome

Substrate throughout: `glove-wiki-gigaword-300`. Basis axes loaded from `exp60_results.npz`.

---

## Free Tests (no new axis construction)

### Thread 0 — Asymmetric-collapse → R  **[RUN 2026-05-27, exp61 + exp61b]**

**What:** Project the UP / DOWN / LIGHT / DARK steering directions (from exp1–4b) onto R from exp60. Verify whether the loop-collapse phenomenology at high all-layer strength corresponds to negative R projection.

**Files:**
- Steering directions: reconstruct from word2vec offsets matching exp1-3c (UP = mean(rising/falling, climbing/falling, ...) etc.)
- R axis: from `exp60_results.npz`

**Predictions:**
- cos(UP, R) < 0 — cascade pole (loops at p=4.0)
- cos(DARK, R) < 0 — cascade pole (loops at p=4.0)
- cos(DOWN, R) ≥ 0 — does not loop
- cos(LIGHT, R) ≥ 0 — does not loop

**Interpretation:**
- All four match → R is causally meaningful and the asymmetric collapse IS perceptual-precision collapse. **Closes the project arc from exp4b to exp60.**
- Two match, two don't → R captures the cascade vocabulary but loop phenomenology has other contributors
- None match → R-as-precision-collapse reading is in trouble; revisit

**Time:** ~5 minutes. Highest narrative payoff for cheapest cost.

**RESULT (exp61, exp61b):** Vocabulary-scope split.
- *Narrow 10-word steering vocab* (exp1-4b's actual steering vectors): null. cos(A_UP_DOWN, R) = −0.019, cos(A_LIGHT_DARK, R) = −0.002. Both essentially zero.
- *MML broad schema vocab* (UP_DOWN_MML 57 pairs, LIGHT_DARK_MML 28 pairs): **both predictions PASS.** cos(UD_MML, R) = −0.081, cos(LD_MML, R) = **+0.178** (cleanest hit).
- Pole-anchored cosines (UP/DOWN/LIGHT/DARK individually with R) all clustered at −0.08 to −0.12, looking like 2/4 hits but actually dominated by shared anisotropy.
- Diagnostic: the shared-pole direction decomposes onto the basis as (G +0.18, R −0.14, W +0.13, A +0.10) — coherent "intense / aroused / effortful / committed-dynamic-happening, cascade-side" signature. Not random noise; not the missing TIME axis (cos(shared, proto-TIME) = +0.015).

**Verdict:** R-as-precision-collapse holds at the broad-schema level, arc-closure from exp4b to exp60 holds at that level. Narrow steering vocab was too spatially-literal to register R-aligned content in a simple cosine. Causal validation still deferred to Thread 6 (steer Pythia with R directly).

**Spawned:** Thread 0c — characterize the narrow-vs-broad-vocab gap across other Lakoff schemas (FB, BAL) to map when narrow construction is and isn't enough.

---

### Thread 2.7(a) — A onto (W, R), free stage  **[RUN 2026-05-27, exp62]**

**What:** Regress A on (W, R). Compute regression coefficients α, β and the residual `|A − αW − βR|`.

**Files:** `exp60_results.npz` has all basis axes.

**Predictions:**
- α > 0 (W coefficient positive — pleasant-calm has low W)
- β > 0 (R coefficient positive — pleasant-calm has high R)
- `|A − αW − βR| / |A|` → magnitude of residual relative to A

**Interpretation:**
- Residual < 0.3 → A substantially decomposes into (W, R). The remaining piece is what SELECTION is hypothesized to provide. Half of Thread 2.7 confirmed before Thread 1.
- Residual 0.3–0.5 → A is touched by W and R but has independent content. SELECTION might pick up some; some might remain.
- Residual > 0.5 → A is largely its own thing. Thread 2.7 fails or needs reframing.

**Time:** ~5 minutes.

**RESULT (exp62):** Signs correct, magnitudes refute the W+R reading.
- α = +0.232 (PASS), β = +0.111 (PASS) — both predicted positive ✓
- residual / |A| = 0.966 → variance of A in (W, R) subspace = **6.6%**
- WEAK verdict. A is overwhelmingly NOT in the (W, R) subspace.
- **The residual is 51% aligned with G_pol** — A is fundamentally a G-shaped object with small W and R flavoring. The A-G entanglement at +0.56 isn't a side-effect; it's structural.

**Reframe:** Thread 2.7's basis-collapse hypothesis pivots from "A = W + R + SELECTION" to **"A = G + small flavoring, and G ≈ SELECTION."** The load-bearing test is now Thread 2.7(c) (G onto SELECTION), not 2.7(b). Thread 1 (SELECTION construction) becomes more urgent — both A and G appear to ride on whatever SELECTION is.

---

### Thread 0.5 — Asymmetric explanation test

**What:** Variance-explained both ways.
- Method 1: project Lakoff schema axes onto 7-axis basis, sum variance-explained.
- Method 2: project 7 basis axes onto Lakoff PC space, sum variance-explained.

**Files:**
- Lakoff schema axes: rerun input axes from exp40 (V, A, UD, IO, FB, LD, EXIST, COH, SUC, LOSS, BAL, PATH, IO_CLEAN, FORCE, DIFF)
- 7-axis basis: `exp60_results.npz`
- Lakoff PCs: recompute from exp40 inputs or saved

**Predictions:**
- Method 1 number > Method 2 number → basis is more fundamental than Lakoff PCs

**Interpretation:** Direct falsifiable test of "the basis is the deeper structure, Lakoff PCs are a projection." A small numerical gap is not enough; want a substantial asymmetry.

**Time:** ~30 minutes (mostly recomputation setup).

---

### Thread 0.7 — Basis vs raw word2vec

**What:** Reconstruction error for held-out concept words. For a diverse held-out set, compute fraction of each word vector explained by 7-axis projection.

**Files:** GloVe-300; `exp60_results.npz`; assemble a held-out word list (NOT used in any anchor construction).

**Predictions:**
- Mean basis-explained across diverse held-out words > 5% (vs. 7/300 = 2.3% if basis were uniformly random)
- Cognitive/affective categories (emotions, agency, abstract states) higher than concrete-noun categories
- If mean > 10%, the basis is recovering substantial substrate structure, not just a constructed-offset artifact

**Interpretation:** Tests whether the basis captures word2vec writ large or only the constructed-offset subspace. Resolves the anisotropy-caveat scope question (§1.5).

**Time:** ~1 hour (need to assemble held-out word set, run projection, summarize by category).

---

## Construction-Required Tests

### Thread 0c (NEW) — Anchor cross-bleed screening protocol  **[ESTABLISHED 2026-05-27, exp69]**

**What:** Methodological protocol for constructing new axes. Before naming any inter-axis entanglement as substrate-real, test whether it's anchor-vocabulary bias by rebuilding with screened anchors.

**Protocol:**
1. List anchor pairs for axis A and axis B.
2. Inspect axis A's vocabulary for content that belongs to axis B's domain (and vice versa).
3. Build screened versions: A_clean from anchors that don't carry B-content; B_clean from anchors that don't carry A-content.
4. Compute cos(A_clean, B_clean). If << cos(A_orig, B_orig), the original entanglement was bias. If ≈, substrate-real.

**Origin:** exp69 found cos(ATTENTION_CLEAN, INTENTION_CLEAN) = −0.13 vs cos(A_orig, G_orig) = +0.56. The +0.56 entanglement that had shaped four lab-notebook entries' worth of theory was anchor-vocabulary bias. A_RUSSELL's anchors included intentional vocabulary (focused/directed/attentive); G's anchors included attentional vocabulary (oriented/targeted).

**Apply to:** all future axis constructions. Particular risk areas flagged for inspection:
- target_EPISTEMIC_VALUE vocabulary may bleed into D (compression/surprisal) — both touch novelty
- target_COST vocabulary may bleed into C (reward) — both touch value
- target_TIME vocabulary may bleed into ATTENTION (intention is future-leaning per exp68)

---

### Thread 1 — target_SELECTION (THE HINGE) **[RUN 2026-05-27, exp63, REFRAMED through exp64-69]**

**What:** Build the SELECTION axis from verb-form selection vocabulary. Test inter-axis cosines, concept-word coverage, and the Thread 2.7 reductions.

**Anchor pairs:**
- (selected, rejected), (chose, refused), (picked, discarded)
- (admitted, denied), (accepted, declined), (kept, removed)
- (chosen, eliminated), (preferred, overlooked), (favored, excluded)
- (designated, omitted), (singled-out, ignored), (highlighted, neglected)

Twelve pairs; check vocabulary coverage.

**Tests after construction:**
1. **cos(SELECTION, IO_CLEAN)** — same axis (>0.5), related (0.2–0.4), or distinct (<0.2)?
2. **cos(SELECTION, G)** — does SELECTION subsume G? (Niamh's prediction: yes, G ≈ specific case of policy-selection.)
3. **cos(SELECTION, A)** — does A's residual (from Thread 2.7(a)) match SELECTION's direction?
4. **cos(SELECTION, all 7 axes)** — orthogonality check.
5. **Concept-word coverage:** project self, selfhood, identity, intimacy, alienation, isolation, communion, exile, belonging onto SELECTION. Compare to IO_CLEAN coverage (only BELONGING loaded on IO; do these load on SELECTION?).

**Why this is the hinge:** Thread 2.7's whole basis-collapse hypothesis (4 of 7 axes potentially drop out) is gated on SELECTION behaving as predicted. If SELECTION turns out to be a narrow construct only covering the IO_CLEAN gap, the basis stays at 7 axes. If it behaves as Niamh's intuition predicts — broad, foundational, subsuming G and providing the missing piece of A — basis simplifies dramatically.

**Time:** ~1-2 hours including all the post-construction tests.

**RESULT (exp63):** Basis-collapse hypothesis REFUTED as constructed.
- cos(SELECTION, IO_CLEAN) = +0.017 — does NOT reframe IO
- cos(SELECTION, G_pol) = +0.163 — does NOT subsume G (2.7% of G variance)
- A explained by SEL alone: 4.6% — worse than W+R's 6.6%

**SECOND READING (exp64 Part A):** Niamh's reframe in conversation: "gating. are we looking at a gating primitive?" Re-examination of SELECTION's pole vocabulary (positive: chosen/selecting/preferred; negative: refuted/denied/categorically/vehemently) showed every anchor was a gating verb and the "denial register" on the negative pole was the linguistic marker of the gate's verdict, not contamination. Renamed: SELECTION → GATING.

**THIRD READING (exp64 A3):** Attention/focus/decision/judgment vocabulary loads on G+A (not on GATING). Renamed: GATING → **DECISION-VERDICT (DV)** — the outcome of evaluation processes. DV is now the standing 8th basis axis. Pole vocabulary: validated/chosen vs refuted/denied. Distinct from C (91.6% non-C residual). Distinct from IO (+0.017).

**Sub-thread C2 (MARKOV_BLANKET, run in exp64 Part B):** Sub-thread C2 from original Entry 25. 13 anchor pairs (self/other, agent/environment, internal/external, mine/theirs, etc.). Result: clean 9th-axis primitive. Max |cos| with basis = +0.137 (with IO_CLEAN). Picks up abstract self/other content (self, ego, soul, identity, autonomy, selfhood) that IO_CLEAN structurally missed (only `belonging` loaded on IO at +0.291). **MB is now the standing 9th basis axis.**

---

### Thread 2.7(b),(c) — A and G onto SELECTION (gated on Thread 1) **[REFRAMED 2026-05-27 after exp62]**

After exp62 showed A is mostly G-shaped (not W+R-shaped), the load-bearing test is now (c) — does G itself reduce to SELECTION? — with (b) repurposed to characterize A's reduction once both G and SELECTION are available.

**(b — repurposed) — A onto SELECTION alone, then A onto (G, SELECTION):**
- Project A onto SELECTION alone. Compute `|A − γ·SELECTION|`. Compare variance-explained vs the 6.6% bar set by (W, R) in exp62.
- Project A onto (G, SELECTION). Compute `|A − γ·SELECTION − δ·G|`. If small, A is fully captured by G and SELECTION together.
- Bonus: A onto (W, R, G, SELECTION) — the full four-axis decomposition, to see what (if anything) is left of A.

Interpretation:
- A onto SELECTION explains >> 6.6% → SELECTION already a better A-axis than W+R combined.
- A onto (G, SELECTION) residual < 0.2 → **A drops from basis** as a G+SELECTION combination.
- Substantial residual remains → A has independent content beyond all candidate primitives.

**(c) — G onto SELECTION:**
- Project G onto SELECTION
- Compute residual `|G − δ·SELECTION|`
- If residual < 0.2: **G drops from basis** — it was always policy-selection specifically
- If moderate residual: G keeps some distinct precision-over-policy content beyond pure selection

**Time:** ~30 minutes after Thread 1.

---

### Thread NEW-0 — Is EPISTEMIC_VALUE a derived state?  **[RUN 2026-05-27, exp70]**

**Hypothesis:** Epistemic value may not need its own basis axis. It might be the low-INTENTION + high-ATTENTION quadrant — explore-mode emerging from precision dynamics rather than a separate quantity.

**RESULT (exp70): Refuted as stated, but informatively.**

```
                        mean cos with EXPL = unit(ATT - INT)
CURIOSITY (n=31)        -0.074
PURSUIT   (n=26)        -0.103
DRIFT     (n=17)        +0.134   ← drift, not curiosity, dominates the quadrant
```

- "Curiosity" vocabulary splits: goal-directed-investigation words (investigate, seek, search, inquire) load on INTENTION; pure-curiosity-state words (inquisitive, intrigued, fascinated, puzzled) fit the (+ATT, −INT) prediction.
- The (+ATT, −INT) quadrant conflates "active novelty-seeking" with "passive drift." Active-inference distinguishes them (explore needs epistemic value driving it); distributional semantics doesn't lexicalize the distinction.
- **Implication:** target_EPISTEMIC_VALUE construction IS still warranted but anchors should be state-based (curious, intrigued, inquisitive, puzzled) not activity-based (investigate, search, seek). The axis would capture what differentiates active novelty-seeking from passive drift in the same quadrant.

Proceed with Thread NEW (target_EPISTEMIC_VALUE) using refined anchor selection.

---

### Thread NEW (priority) — target_EPISTEMIC_VALUE  **[RUN 2026-05-27, exp73 — CONFIRMED as 10th-axis primitive]**

**What:** Build the gathering-information / curiosity-as-primitive axis. Active inference's second EFE term (after pragmatic value = C). Niamh's gap-noticing in conversation; EE failed in exp56 because it operated at wrong abstraction level. The derived-state hypothesis (Thread NEW-0) showed pure-curiosity-state and goal-directed-investigation are split in vocabulary, with the quadrant conflated with drift — so a dedicated axis IS needed to capture the "what makes novelty-seeking valuable" content.

**Anchor pairs (REVISED after exp70 — state-based only, avoiding activity-vocab that loads on INTENTION):**
- (curious, indifferent), (intrigued, dismissive), (fascinated, bored)
- (inquisitive, incurious), (puzzled, settled), (wondering, knowing)
- (marveling, dismissing), (awestruck, jaded), (mystified, certain)
- (engaged, blasé)

**REMOVED from original list (these load on INTENTION, not curiosity):**
- (investigating, ignoring), (probing, conceding), (exploring, settling), (inquiring, dismissing)
  — These are goal-directed-investigation verbs that carry intentional commitment in language.

**Apply Thread 0c screening:** check anchors aren't bleeding into INTENTION (avoid action-verbs); aren't bleeding into C (curiosity should be epistemic, not pragmatic positive valence); aren't bleeding into D (familiarity/novelty is in D's anchors — avoid using novelty terms directly).

**Cross-bleed screening required against:**
- D (compression/surprisal) — D's anchors include familiar/unfamiliar; check EPISTEMIC anchors aren't tracking surprise directly.
- C (reward) — epistemic value is non-pragmatic in active inference (the part of EFE you'd minimize even without preferences).
- ATTENTION_CLEAN — curiosity overlaps with attention but is more specific.
- INTENTION_CLEAN — investigating is intentional but epistemic-specific.

**Predictions (current Claude, committed before run):**
- cos(EV, ATT) in [+0.10, +0.30]
- cos(EV, INT) in [-0.10, +0.10] (near zero; >+0.20 = screening failed)
- cos(EV, C)   in [+0.10, +0.25]
- cos(EV, D)   in [+0.15, +0.30]
- max |cos| in [0.25, 0.35], most likely on D or C

**RESULT (exp73):** **Confirmed as 10th-axis primitive.**
- max |cos| = **+0.154 (with DV)** — among cleanest basis members, comparable to MB (+0.137)
- cos with each axis: C +0.023, W −0.108, ATT +0.033, INT **−0.137**, R −0.043, D **−0.146**, IO +0.037, DV +0.154, MB +0.061
- All falsifiers cleared (max |cos| < 0.40; cos with INT not > +0.20)
- Pole vocabulary excellent: positive = intrigued/fascinated/marveled/entranced/captivated/curious/puzzled/mystified; negative = outright/prior/authorization/rejection/customary/dismissal ("already-decided" cluster)
- State-curiosity words load 0.34–0.55 on EV; activity-verbs (investigating, seeking) load on INT not EV; drift-cluster loads on neither — exp70's distinction now empirically separable
- cos(EV, EXPLORATION = unit(ATT−INT)) = +0.113 — EV is NOT in the explore-quadrant; it's the *driving signal* that distinguishes active novelty-seeking from passive drift

**Prediction-calibration miss (on record):** all four predicted cosines missed in the same direction — I overestimated overlap with related primitives. Same pattern as exp72. Active-inference theory predicts coupling; distributional semantics shows EFE components are lexically much more separable than the theory predicts. Each component (value / cost / γ / π / epistemic-value / surprisal) has its own distinct lexical neighborhood.

**Significance:** with EV added, the full first-order EFE = C − W − EV decomposition is lexically realized at the basis level. Basis is now 10 axes.

---

### Thread 2 — target_COST (the W → COST reframe test)

**What:** Build COST from non-somatic computational-cost vocabulary. Test whether W is recovering computational cost via somatic surface vocabulary.

**Anchor pairs:**
- (expensive, free), (taxing, refreshing), (demanding, easy)
- (laborious, automatic), (depleting, sustaining), (consuming, replenishing)
- (extracting, conserving), (taxing, restorative)
- Possibly add: (costly, gratis), (effortful, automatic), (burdening, freeing)

**Predictions and interpretation:**
- cos(COST, W) > 0.7 → computational-cost IS the primitive; somatic-weight is one expressive modality. **Empirical evidence against strong embodied-cognition.** Rename W to COST in basis.
- cos(COST, W) ≈ 0.3-0.6 → mixed; cost and weight related but distinct
- cos(COST, W) < 0.2 → separate primitives; embodied-cognition reading partially vindicated for this case

**Secondary tests:**
- cos(COST, DIFF) — should be ≈ +0.50 if cost IS what DIFF tracks
- cos(COST, A_affect) — should be < +0.23 (W's value) if non-somatic vocabulary reduces affect coupling

**Time:** ~1 hour.

---

### Thread 2.5 — target_TIME

**What:** Build TIME from temporal-ordinal vocabulary. Quick-pass first with existing temporal words, then full construction.

**Quick-pass (no construction):** project temporal words onto current 7-axis basis. Words: yesterday, tomorrow, past, future, ancient, modern, historic, prospective, before, after. Do they cluster on D or LD-aligned content, or near-zero everywhere?

**Full anchor pairs:**
- (past, future), (yesterday, tomorrow), (before, after), (earlier, later)
- (ancient, modern), (old, new), (remembered, anticipated)
- (begun, pending), (completed, planned), (precedes, follows)
- (was, will-be — handle stop-word issue), (historical, upcoming)

**Predictions:**
- cos(TIME, D) — moderate positive (past more compressible than future)
- cos(TIME, LD) — moderate positive if Lakoff LIGHT-IS-NEW / DARK-IS-PRIMORDIAL coupling is real in word2vec
- cos(TIME, A_affect) — should be small
- cos(TIME, all other axes) — should be small
- Max |cos| with existing 7 axes < 0.3 → TIME is an 8th primitive candidate

**Bonus:** test whether TUE-WED → LD finding (+0.81 in Pythia SAE) replicates in word2vec. Independent question about substrate-dependence of LD's temporal coupling.

**Time:** ~1 hour.

---

### Thread 7 — Sham-sweep

**What:** Build ~20 candidate shams from diverse categories. Project each onto 7-axis basis + Lakoff schemas. Score schematicity.

**Candidate shams (categories):**
- Numerical: (5, 8), (two, seven), (twelve, fifty)
- Geographic: (Boston, Chicago), (Paris, Berlin), (Lagos, Cairo)
- Names: (Sarah, Michael), (Aisha, David), (Mei, Carlos)
- Taxonomic-arbitrary: (poodle, labrador), (sparrow, robin), (oak, maple)
- Made-up nonsense: (zlort, grebbit), (flooble, snarp) — test in vocabulary first
- Already-tested-historical: (coffee, tea), (Tuesday, Wednesday), (apple, orange) — for reproducibility check

**Scoring:** for each sham, compute max |cos| with any of the 7 basis axes or 9 Lakoff schemas. High max → schema-loaded. Low max → clean sham.

**Predictions:**
- If most/all shams have max |cos| > 0.2 → "you cannot construct non-schematic contrasts" holds. Niamh's working hypothesis confirmed.
- If a meaningful fraction have max |cos| < 0.1 → claim fails; we learn which categories escape schematicity.

**Time:** ~2 hours including making and testing nonsense-word vocabulary check.

---

## Substrate Transfer

### Thread 4 — fastText replication (DECISIVE substrate test)

**What:** Rerun the full 7-axis basis construction + Lakoff schema projection + concept-word projections in fastText. Compare to GloVe-300 findings.

**Files:** fastText (Facebook — Wikipedia + Common Crawl, ~2M vocab); same anchor vocabularies; same exp40, exp59, exp60 protocols ported.

**Predictions:**
- 7 axes reconstruct from fastText anchors with cos > 0.7 to GloVe versions (each axis is the same direction in both substrates)
- cos(W, DIFF) ≈ +0.50 in fastText too (DIFFICULTIES-ARE-BURDENS substrate-invariant)
- cos(G, PC2) ≈ −0.30 in fastText Lakoff space (policy-precision reading holds)
- Clinical fear/anxiety/panic distinction reappears in fastText concept-word projections
- A-G entanglement: if also +0.56 in fastText, substrate-real coupling. If substantially different, anchor-bias.

**Interpretation:** If yes: the active-inference reading isn't GloVe-specific. If no: most of the project's findings are GloVe-artifact.

**Time:** ~half a day (lots of script porting).

---

### Thread 6 — Pythia steering with new basis (closes loop with Part 1)

**What:** Inject C, W, R, G (or post-Thread-1 simplified basis: C, COST, R, D, SELECTION) as steering directions during Pythia generation. Test for theoretically-predicted behavioral effects.

**Predictions:**
- Positive C steering → outputs more wellbeing-positive (cleaner than UP-steering, no high-strength collapse if C is purer than the UP composite)
- Negative R steering → K-collapse threshold drops (loops appear at lower α) — **direct behavioral validation of R-as-precision-collapse**
- Positive R steering → K-collapse threshold rises (more coherence preserved at higher α)
- Positive G (or SELECTION) steering → outputs more goal-committed / agentic
- Negative C steering → outputs more suffering-themed

**Interpretation:** Closes the project arc — started by steering Pythia with Lakoff schemas (UP/DOWN), end by steering with the active-inference primitives. Same methodology, right axes, cleaner predictions.

**Time:** ~1-2 days (lots of generation runs, careful baseline comparisons).

---

## Status

- Free tests (Thread 0, 2.7(a), 0.5, 0.7): all unblocked, can run now
- Thread 1 (SELECTION): unblocked, ~1-2 hours, **highest priority new construction** because Thread 2.7(b)(c) depend on it
- Thread 2 (COST), Thread 2.5 (TIME): unblocked, independent of Thread 1
- Thread 7 (sham-sweep): unblocked, independent
- Thread 4 (fastText), Thread 6 (Pythia steering): bigger investments, do once basis is stable

**Suggested order for a fresh test session:**
1. Thread 0 (5 min, satisfying first win)
2. Thread 2.7(a) (5 min, partial validation of A decomposition)
3. Thread 1 (1-2 hours, the hinge)
4. Thread 2.7(b),(c) (30 min after Thread 1, completes the basis-collapse story or refutes it)
5. Thread 2 (COST) and Thread 2.5 (TIME) (parallel-runnable)
6. Thread 0.5 + Thread 0.7 (somewhere in the middle, low-priority but informative)
7. Thread 7 (sham-sweep) — when basis is stable
8. Thread 4 (fastText) — when basis is stable and we want substrate-robustness
9. Thread 6 (Pythia steering) — final causal validation

---

*See WRITEUP_v3.md for full theoretical context. See BASIS_REFERENCE.md for current axis numbers, anchor vocabularies, and concept-word projections.*
