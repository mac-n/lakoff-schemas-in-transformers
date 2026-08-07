# PRE-REGISTRATION — exp161: GPT-2 BALANCE ↔ attention-entropy
Written 2026-06-12, BEFORE any code or model run. (Claude; Niamh to
review before execution.)
v1.1 amendment (pre-run, logged): the first draft's prompt set
contained BALANCE axis-definers ("level"×2, "settled", "steady"×3) —
caught by an automated overlap check (whose own first version was
buggy and checked the wrong text region; rerun correctly). Six prompts
rewritten; C5 axis-word exclusion added. No model had been run.

## Provenance / why this needs a prereg
exp160 (LIGHT-DARK→entropy, pre-registered, correctly predicted null)
produced a POST-HOC find: partial r(BALANCE_proj, attn_entropy | pos,
norm) negative and monotone in depth, strongest in GPT-2-medium
(L8 −0.21, L12 −0.26, L16 −0.30) — the model with NO BALANCE↔norm
coupling. exp160b showed entropy is not norm (cos(d_ent, d_norm)
0.25–0.76, not the 0.99 collapse that killed load), but both analyses
were post-hoc on the same 40 prompts, with BALANCE selected from an
8-schema × 3-model × 5-layer table after looking. Winner's curse and
forking paths fully apply. exp161 is the confirmatory test.

## Hypothesis under test
H: GPT-2-medium grounds the BALANCE schema axis in attention
concentration (entropy) — a second embodiment, in a different regulated
variable than Pythia's norm grounding. Multiple realizability,
demonstrated rather than conjectured, if it survives.

## Design
1. FRESH DATA: 40 new generic prompts (frozen in this file, Appendix A),
   written before any model run, same genre as exp160's (everyday
   scenes, varied register), NOT stuffed with balance/stability
   vocabulary — the BALANCE projection must vary naturally.
2. Models: GPT-2-medium PRIMARY (decision layers 8/12/16).
   pythia-410m and Llama-3.2-1B secondary (cross-substrate table).
3. Per-layer machinery as exp160 (attn_entropy_lib): isolated-word
   residuals → aniso+freq strip → 8 schema axes; per-query normalised
   attention entropy; per-token unit-residual projections.
4. d_norm: HELD-OUT estimation per exp154c protocol (estimate on half
   the vocab, carrier check out-of-sample). Full-set d_norm is circular
   and is not used anywhere in the verdict.

## Control stack (the point of the experiment)
- C1 baseline (exp160's): partial r(BAL, ent | position, scalar norm).
- C2 explicit norm-orthogonalisation, BOTH sides:
    covariates = position, scalar norm, AND the token's d_norm_ho
    projection; axis = BALANCE projected ⊥ d_norm_ho.
    C2 is the HEADLINE statistic.
- C3 schema-orthogonalisation: as C2 but axis additionally ⊥ the other
  7 schema axes (Gram–Schmidt, BALANCE last). Distinguishes
  BALANCE-unique coupling from shared schema-space variance.
- C4 specificity: all-8-schema partial table (C2 covariates), BALANCE
  rank by |r|.
- Inference honesty: tokens cluster within prompts, so pooled n (~500)
  overstates evidence. Report prompt-level cluster bootstrap 95% CI
  (resample prompts, 1000 reps) alongside pooled r. Gate uses pooled
  point estimates (comparability with exp160); CI is reported context.
- (Token-count purity, exp157b lesson: n/a here — entropy is per-query
  within prompts, not a per-word last-token scalar; position covariate
  carries the sequence-structure confound. Stated so the omission is
  deliberate, not forgotten.)
- C5 axis-word exclusion (NEW vs exp160, v1.1): any prompt token whose
  lowercased string matches an axis-defining word for ANY of the 8
  schemas is EXCLUDED from all correlations. Axis-defining tokens
  project onto their own axis trivially; exp160 did not do this (its
  prompts contained "light", "still", "open", ...) — protocol
  improvement, logged.

## Precondition (gate validity, exp158 lesson)
Held-out d_norm carrier (out-of-sample corr) ≥ 0.50 at every GPT-2
decision layer. If it fails, verdict = INVALID (the norm-
orthogonalisation is meaningless), NOT a null. Selftest must include
the "precondition fails + headline null" case.

## Committed predictions (before code; calibration context: my
deflationary point predictions have been reliable, mechanism stories
0-for-7 — but this is a replication-with-controls, not a mechanism)
- P1 (replication): C1 partial r NEGATIVE at all three GPT-2 decision
  layers on fresh prompts, |r| in 0.10–0.30 (attenuated vs exp160 by
  winner's curse, but real).
- P2 (norm controls cheap in GPT-2): C2 differs from C1 by < 0.05 at
  each GPT-2 decision layer — cos(BALANCE, d_norm) ≈ 0 there, and the
  entropy-side overlap is carried by the covariates.
- P3 (specificity, deflationary): BALANCE top-2 by |r| at a majority of
  decision layers but NOT top-1 at all three. Top-1 sweep = stronger
  than I expect.
- P4 (schema-orth, deflationary): C3 attenuates the coupling by ≥ half
  relative to C2 — most of it is shared schema-space variance. If C3
  |r| ≥ 0.15 survives, the BALANCE-unique claim is live and that is
  bigger news than I am betting on.
- P5 (cross-substrate signature): under C2, Pythia's BALANCE-entropy
  |r| < GPT-2's at matched decision layers (Pythia's version is largely
  absorbed by the d_norm controls). This is the multiple-realizability
  fingerprint: same concept, different carrier per body.

## Decision rule (GPT-2 only; precondition must pass)
- V1 SECOND-EMBODIMENT CANDIDATE STANDS: C2 negative with |r| ≥ 0.15 at
  ALL GPT-2 decision layers AND BALANCE top-2 (C4) at a majority.
    V1a if also C3 |r| ≥ 0.15 at a majority (BALANCE-unique);
    V1b otherwise (real, but shared schema variance).
  → next step: causal test (BALANCE-steer → entropy shift), exp165+.
- V2 DEAD: C2 |r| < 0.10 at all GPT-2 decision layers. exp160's lead
  was selection/prompt-set noise; return to the carrier-first list.
- V3 AMBIGUOUS: anything else. Diagnose before any causal work.
- INVALID: precondition failed. Fix d_norm estimation before reading
  any number.
- C3 is explicitly NON-GATING (Niamh, 2026-06-12): the schema axes are
  known to be linearly non-orthogonal (exp162: PR 6.2–7.0, top-2 var
  39–44%), and whether other schemas relate to BALANCE NONLINEARLY has
  never been measured — so "fails ⊥-schemas" cannot be read as "not
  BALANCE" until S1 (below) is understood.

## S1 — declared secondary analysis (non-gating, same data)
Pairwise nonlinear dependence of BALANCE on each of the other 7 schema
projections, on the exp161 token cloud (C5-filtered), per model at
decision layers: Pearson r vs Spearman rho vs distance correlation
(dCor), with a permutation null (1000 shuffles) for the NONLINEAR
EXCESS statistic dCor² − r². Question: do any schemas predict BALANCE
through structure that linear r misses? No predictions committed — this
is measurement, not a bet; any finding here is a lead for its own
prereg, not a result.

## What I will NOT do
- No swapping in the old 40 prompts if fresh ones look null.
- No promoting a different schema if BALANCE fails C4 — that would be
  a new post-hoc lead for a future prereg, logged as such.
- No quoting C1 as the result if C2 is weaker. C2 is the headline.

## Appendix A — frozen fresh prompt set (40)
1.  The kettle clicked off and she poured the water over the leaves.
2.  Halfway through the meeting nobody could remember who was chairing it.
3.  The carpenter measured twice and the shelf fit on the first try.
4.  Loose papers slid from the pile every time the door swung open.
5.  The orchestra tuned to a single note and the hall went quiet.
6.  His suitcase burst at the airport and his clothes went everywhere.
7.  She filed the receipts by month and closed the drawer with a click.
8.  The toddler stacked the blocks until the tower leaned and toppled.
9.  The ferry crossed on schedule despite the morning swell.
10. Three alarms went off at once and he silenced none of them.
11. The gardener pruned the hedge into a clean straight line.
12. The spreadsheet totals refused to match no matter who re-added them.
13. A heron stood motionless at the edge of the reservoir.
14. The debate drifted off topic within the first five minutes.
15. The baker weighed the flour and the dough came out the same as always.
16. Her headphones tangled into a knot at the bottom of the bag.
17. The surveyor set the tripod and the bubble came to rest in the centre.
18. Rumours of the merger changed shape with every retelling.
19. The night train ran quiet and true across the plain.
20. The committee's minutes contradicted the recording in three places.
21. He laced his boots, checked the map, and set off at first light.
22. Paint dripped from the ladder onto the new carpet below.
23. The accountant reconciled the books before the end of the quarter.
24. Half the streetlights were out and the road kept changing under him.
25. The juggler kept four clubs aloft in an unbroken arc.
26. The printer jammed again and the queue of documents kept growing.
27. The librarian reshelved the returns before the doors opened.
28. Wind gusted through the market and the stalls flapped and strained.
29. The pilot trimmed the aircraft and the nose held its line.
30. The recipe doubled badly and the sauce split in the pan.
31. She watered the plants on the same morning every week.
32. The scaffolding clattered as the storm pulled at its joints.
33. The clockmaker set the pendulum and the ticking evened out.
34. Boxes from the move still blocked the hallway a month later.
35. The rowers found their rhythm and the boat ran flat and fast.
36. His password expired mid-task and the form deleted everything.
37. The nurse charted the doses in a neat, unhurried hand.
38. Gravel spilled across the lane where the wall had given way.
39. The tide came in exactly as the tables said it would.
40. The signal cut out each time the speaker reached her point.
