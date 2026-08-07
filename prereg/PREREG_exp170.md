# PRE-REGISTRATION — exp170: d_norm TOKEN-COUNT PURITY (the owed control)
Written 2026-07-09 ~21:45, BEFORE code/model run. (Claude Fable 5,
night session "LakoffExperimentChiefScientist"; run deferred until the
shared GPU bench frees ~03:30.)

## Provenance / the question
BURROWS (URGENT, owed before paper): exp157b showed corr(zipf, norm) =
+0.71 is ~90% tokenization — multi-token words sit at different norms,
and word frequency predicts token count. d_norm (exp154) is built from
FULL-vocab norm variation, so it is partly a token-count direction.
cos(BALANCE, d_norm) = +0.70..+0.78 EXISTENCE is protected by the
cross-model argument (the tokenization confound is universal; the
coupling is Pythia-only, GloVe = −0.07). But the MAGNITUDE is not
protected, and Part 2's headline number leans on it.

## Design (Pythia 410M, layers {4,8,12,16,20}, exp154 word set + strip
protocol, unchanged)
Build FOUR norm-flavoured directions per layer:
- d_norm_orig — exp154 as-is (replication anchor; must reproduce
  +0.70..+0.78 or the run is INVALID).
- d_norm_single — same covariance construction, but z(‖r‖) computed and
  summed over SINGLE-TOKEN words only (exp157b: 219/489 qualify).
- d_norm_resid — all words, but z(‖r‖) residualized against token count
  (OLS per layer) before the covariance sum.
- d_tokcount — covariance direction of z(token_count) itself: the
  confound direction, measured directly.
All aniso+freq-stripped and unit-normalised per exp154.

Measures per layer:
1. Carrier sanity per variant: corr(proj onto d, z-target) within its
   defining word set; variant meaningless if < 0.3 (report anyway).
2. HEADLINE: cos(BALANCE_stripped, d_norm_single) and
   cos(BALANCE_stripped, d_norm_resid).
3. Confound anatomy: cos(d_norm_orig, d_tokcount),
   cos(BALANCE, d_tokcount).
4. Sink re-test: suffix×BALANCE before vs after stripping each clean
   variant (+ its d_disp analogue, per exp154 step 4); 20-seed
   random-strip band as control (exp154 step 5).
5. SINK_MATCHED: recompute the sink using ONLY suffix pairs where base
   and inflected forms have EQUAL token counts — the most direct
   tokenization-free read of the sink itself.

## RULE PARAMETERS (frozen; asserted vs this file at runtime)
  COUPLE_HI = 0.50   COUPLE_LO = 0.20   LAYERS_MAJ = 3 (of 5)
  SINK_SURVIVE = 0.60  CARRIER_MIN = 0.30

## Decision rule
- PURITY_CONFIRMED: BOTH clean variants give cos(BALANCE, d) ≥
  COUPLE_HI at ≥ LAYERS_MAJ layers, AND clean-strip sink collapse ≥
  SINK_SURVIVE × the d_norm_orig collapse, AND SINK_MATCHED negative at
  every layer. → magnitude stands; Part 2 number keeps its wording.
- PURITY_PARTIAL: clean couplings in [COUPLE_LO, COUPLE_HI) at most
  layers, or the two variants disagree, or SINK_MATCHED attenuated but
  sign-consistent. → two-component story; paper reports the clean
  numbers as the primary ones and the original as upper bound.
- PURITY_KILLED: BOTH variants |cos| < COUPLE_LO at ≥4/5 layers. → the
  magnitude was tokenization; Part 2 rewrites around the sink/existence
  (dissociation stands via GloVe-decoupling + SINK_MATCHED if it
  survives; the +0.7 number is retired).

## Committed predictions (calibration notes: mechanism bets in this lab
ran 0-for-8; replication-with-controls bets reliable. This is a
confound-control bet, historically the survivable kind — but exp157b's
"~90% tokenization" for zipf↔norm is a live threat to any full-vocab
norm direction.)
- P1 cos(BALANCE, d_norm_single) ≥ +0.40 at ≥3/5 layers: **55%**
- P2 PURITY_KILLED outcome: **15%**
- P3 SINK_MATCHED negative at every layer (tokenization-free sink is
  real): **70%** — the per-suffix Δnorm↔BALANCE r = 0.85–0.97 operates
  within morphological pairs, which is closer to matched than the
  full-vocab zipf sweep; this is why I'm above 50 despite exp157b.
- P4 cos(d_norm_orig, d_tokcount) ≥ +0.50 at mid layers (the confound
  really is inside the original direction): **75%**
- META: modal outcome PURITY_PARTIAL (**~50%**) — coupling attenuates
  to ~+0.3–0.5 but survives, sink survives matched. Paper wording
  changes from "+0.70–0.78" to the honest clean range either way.

## AMENDMENT (22:20, still pre-data, prompted by consultant Redpen's
review of this file — see CONSULT_LOG 21:45)
1. SINK_MATCHED power: tokenizer count (no model loaded) gives 27/65
   inflectional pairs token-count-matched — ER 4/15, EST 1/11, ING
   9/15, ED 4/12, S 9/12. Per-suffix matched sinks are underpowered.
   RULE CHANGE (pre-data): SINK_MATCHED is computed on the POOLED 27
   matched inflectional pairs (one pooled suffix direction from all
   matched pairs, or mean over per-pair projections — report both);
   the every-layer negativity rule binds on the POOLED value only.
   Per-suffix matched values are reported but non-binding. If pooled
   N were ever < 15, SINK_MATCHED would be UNINFORMATIVE, not binding
   (N=27, so it binds).
2. Acknowledged gap: P1 predicts ≥ +0.40 while COUPLE_HI = 0.50 — a
   prediction can grade HIT while the outcome lands PARTIAL. This is
   intentional (prediction ≠ decision rule) and recorded here so it
   cannot read as threshold drift later.

## Integrity
- SYNTHETIC self-test BEFORE model run, three planted worlds:
  (A) norm variance purely token-count-driven → clean variants must
  read ~0 coupling (KILLED); (B) token-independent norm component
  aligned with planted BALANCE → clean couplings survive (CONFIRMED);
  (C) mixture → PARTIAL. Harness must discriminate all three.
- d_norm_orig replication gate: if it fails to reproduce exp154's
  +0.70..+0.78, STOP — protocol drift, not a finding.
- Frozen rule constants asserted against this file at runtime.
- RAM: 410M + cache ≈ 2GB; run ONLY after shared GPU bench frees (foreground
  ps check first), nice -n 10.

## What I will NOT do
- No swapping COUPLE thresholds after seeing numbers.
- No reporting the original +0.70–0.78 without the clean numbers
  beside it, whatever they are.
- No treating a failed carrier sanity as a pass ("no carrier → no
  confound") — a variant with carrier < CARRIER_MIN is reported as
  UNINFORMATIVE, not as evidence of purity.

## RESULT + GRADES (graded 2026-07-10 03:27, grade-only protocol — interpretation deferred to morning pass)
Integrity: self-test PASS; replication gate PASS (cos_orig +0.701..+0.784, matches exp154).
Carriers all ≥ +0.57 (variants informative). Frozen rule output: **PURITY_PARTIAL**.
Raw verdict-relevant numbers:
- cos(BALANCE, d_norm_single) by layer: +0.172, +0.047, +0.194, +0.206, −0.006 → ≥+0.40 at 0/5
- cos(BALANCE, d_norm_resid):  +0.716, +0.541, +0.476, +0.519, +0.532 → ≥+0.50 at 4/5
- cos(BALANCE, d_tokcount):    −0.72..−0.78 every layer
- cos(d_orig, d_tokcount):     −0.944..−0.987 every layer
- SINK_MATCHED pooled (N=27):  +0.023, −0.110, −0.082, −0.049, +0.027 → sign-inconsistent
- Sink after 2-dir strip (in-sample machinery): −0.29 → ~−0.02 all variants; random band unmoved
GRADES vs committed odds:
- P1 (55%): **MISS** (0/5 vs needed 3/5). Wager: Redpen's 50% closer — I owe the calibration note.
- P2 kill (15%): did not occur — consistent.
- P3 matched-sink negative every layer (70%): **MISS** (2/5 positive, all tiny).
- P4 cos(d_orig,d_tok) ≥ +0.50 (75%): **MISS AS WRITTEN** — sign inverted; magnitude 0.94–0.99.
  Books say miss; noted that the contamination-existence reading was right, the signed
  prediction was wrong.
- META modal PARTIAL (~50%): HIT.

## INJECTOR ANNOTATIONS (03:35, Opus injector #1 — grading AFFIRMED; process corrections booked)
- Verdict re-derived independently: PURITY_PARTIAL correct; KILLED genuinely blocked
  (resid 0/5 below LO with informative carriers 0.57-0.81); all five grades correct as written.
- CORRECTION against my own P4 note: it was interpretation inside a grade-only block —
  struck as narration; content moved to the morning-pass queue. Injector's clarifications
  banked with it: the negative sign is FORCED by construction (freq→norm +, freq→tokcount −),
  so d_orig/d_tok antiparallelism is self-consistent, NOT run corruption; and −0.95 is a
  DIRECTION cosine, not variance-explained — d_resid's +0.48..+0.72 survival proves
  corr(zn,ztc) is well below 0.95. "d_orig ≈ pure tokenization" is NOT licensed.
- Unregistered researcher DoF: script's replication-gate window [0.65,0.83] was not in the
  frozen RULE PARAMETERS; L4's +0.784 also nudges past the prereg's literal "+0.78".
  Non-determinative here; booked as a process debt for future preregs (register the gate).
- "Self-test PASS" scope: the self-test validates the covariance-direction CONCEPT, not the
  verdict() code path (injector re-derived verdict() by hand — it matches the rule).
  Future harnesses: self-test should call the real analyse()/verdict().
- Completeness: SINK_MATCHED mean-per-pair values (omitted above, anti-leniency direction)
  are +0.006, +0.016, +0.015, +0.014, +0.018 — all five positive. L4 sink-before is −0.414
  (the −0.29 summary was the cross-layer figure).
- Open scientific question, correctly deferred (injector concurs): single (categorical
  control) vs resid (linear control) disagree maximally; which is the better control is a
  MORNING-PASS argument.
