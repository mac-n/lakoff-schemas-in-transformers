# PRE-REGISTRATION — exp172: outlier-dimension robustness of the BALANCE–norm coupling
Written 2026-07-10 ~00:10, BEFORE code/model run. (Claude Fable 5,
night session; provoked by the Opus disagreement injector's
massive-activation challenge, relayed & verified via CONSULT_LOG
23:55 item 3.)

## The question
exp155 shows mean inflectional Δnorm jumping −7.0 (L4) → −534.7 (L8):
the norm story partly lives in a massive/outlier-activation regime
(rogue dimensions, a documented transformer artifact family). Attack:
"BALANCE reads out norm physiology" is really "BALANCE reads out a
few outlier dims." Does the coupling survive clipping them?

## Design (Pythia 410M, layers {4,8,12,16,20}, exp154 word set/strip)
- Identify outlier dims per layer: dims whose mean |activation|
  across the word set exceeds a robust threshold of the per-dim
  distribution (report count; expect a handful).
  [AMENDMENT, pre-data, same night: threshold is MEDIAN + 5·(1.4826·MAD),
  not MEAN + 5·SD — the synthetic self-test showed mean/SD detection
  self-masks (outlier dims inflate the SD used to find them, planted
  world (a) went undetected). Changed before any model data was seen.]
- Rebuild, with those dims ZEROED in all residuals before any other
  step: BALANCE axis, d_norm (exp154 construction), suffix dirs.
- Measures per layer: (1) n_outlier dims and their share of total
  norm variance; (2) cos(BALANCE_clip, d_norm_clip); (3) inflectional
  mean sink before/after clipping; (4) control: clip an equal number
  of RANDOM dims, 20 seeds, same measures.

## RULE PARAMETERS (frozen)
  OUT_Z = 5.0   SURVIVE = 0.60   N_RAND = 20

## Decision rule
- ROBUST: cos(BALANCE_clip, d_norm_clip) ≥ 0.60 × unclipped value at
  ≥4/5 layers AND sink retained ≥ 0.60 × unclipped, both outside the
  random-clip band. → concession sentence in blog gets its answer.
- ARTIFACT: coupling drops < 0.30 × unclipped at ≥3/5 layers while
  random-clip band stays high. → "reads out norm physiology" demotes
  to "reads out outlier-dimension structure"; blog/paper F5 rewritten.
- MIXED: anything else → report both components, keep the concession.

## Committed predictions
- P1 ROBUST: **55%** (the sink survived held-out norm-strip at
  L8–L20, suggesting distributed structure; but the L8 Δnorm scale
  jump is genuinely outlier-flavoured)
- P2 ARTIFACT: **20%**
- P3 outlier dims number < 12 per layer at OUT_Z=5: **70%**

## Integrity
- Synthetic self-test: plant (a) coupling carried by 3 outlier dims →
  ARTIFACT; (b) coupling distributed across dims with outliers
  present but orthogonal → ROBUST. Must discriminate before model.
- nice -n 10, free bench only, after exp170/171.

## RESULT + GRADES (graded 2026-07-10 03:32, grade-only protocol)
Integrity: self-test PASS (after pre-data amendment to median+MAD detection).
Frozen rule output: **ROBUST** (coupling retained ≥0.6 at 5/5 layers; sink 5/5; artifact 0/5).
Raw: outlier dims per layer 10/13/12/9/8 (L4 differs: 10 dims at **42.2%** of residual
energy; L8–L20 share a stable recurring set — 130, 357, 752, 966 et al. — carrying
**99.2–99.5%**); after zeroing them:
[CORRECTION (this line replaced an earlier transcription error made minutes prior: I had
recorded "13/12/12/9/8" and "99.2–99.5% at all layers", having read the table from L8 down
and missed L4's row. Caught on re-grep before any injector; raw file is authoritative.)]
cos(BALANCE, d_norm) retains 90–99% (+0.729→+0.721 at L8, +0.701→+0.673 at L12,
+0.718→+0.659, +0.729→+0.657); sink retains 94–111%; random-clip bands unmoved (±0.001–0.014).
GRADES vs committed odds:
- P1 ROBUST (55%): **HIT**
- P2 ARTIFACT (20%): did not occur — consistent.
- P3 outlier count < 12 per layer at OUT_Z=5 (70%): **MISS as written** (13 at L8, 12 at L12).
  [SECOND CORRECTION: this line originally read "13 at L4, 12 at L8/L12" — the same
  L8-down misread as the top row, caught by Opus injector #3 AFTER my first correction
  claimed to have fixed it. Grade unchanged; the lesson about transcription under
  time pressure is now twice-taught and goes in the morning report.]

## INJECTOR ANNOTATIONS (03:43, Opus injector #3 — verdict AFFIRMED; two live catches booked)
- Verdict re-derived: ROBUST correct, all quoted figures match raw, P1 HIT / P3 MISS stand.
- CATCH 1: P3 justification carried the same layer-misread my first correction claimed to
  fix (now corrected above, marked). One caught error DID mean others.
- CATCH 2 (framing): "clipped dims carry 99% of energy" overstates stringency — the
  massive dims are near-DC (high mean |activation|), and BALANCE is a difference-of-means
  whose DC component cancels in the cosine regardless. The honest characterization:
  outlier-clip measurably hurts (drops cos 1-10%, vs ~0% for random-clip — so the dims
  carry SOME real contrast) and the coupling survives anyway. Modest-but-real robustness,
  not 99%-energy-survival heroics. Morning pass must use this framing.
- Code-vs-rule gap: the ROBUST branch doesn't gate on the outside-random-band clause
  (frozen text requires it); clause HOLDS if enforced (outlier-clip cos below every
  random band) — verdict unaffected, operationalization debt booked.
- Amendment provenance affirmed via git: median+MAD locked 22:01, data 03:31. Stale
  docstring line (still says mean+SD) noted for cleanup.
- L4 nuance: L4 shares the recurring dim set; only its ENERGY share differs (42.2%).
