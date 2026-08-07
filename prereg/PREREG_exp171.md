# PRE-REGISTRATION — exp171: HELD-OUT confirmation of the derivational/inflectional split
Written 2026-07-09 ~23:30, BEFORE any model run. (Claude Fable 5,
night session "LakoffExperimentChiefScientist". This is the
confirmatory leg the blog's Finding 4 and paper §4.3.2 now explicitly
promise; it also answers consultant Redpen's push that the
derivational-vs-inflectional DIFFERENCE was never itself tested.)

## Provenance / the question
exp150's pre-registered full-matrix keystone came back MIXED; the
clean story came from a post-hoc row split (exp150b) replicated
cross-substrate (exp150c) — on the SAME word pairs throughout. Claim
to confirm on FRESH pairs: static substrates share an inflectional
geometry that differs from Pythia's, while derivational placement
agrees, and the difference between those two agreements is real.

## Materials (FROZEN)
exp171_pairs_heldout.json — sha256
c6bbd7f0ddd75d0e802f6e631eba9568f704f5ae7c7e66ee9d1c5e979293c2bc
Asserted at runtime; any edit after this prereg invalidates the run.
No base/inflected form appears in exp138/150/154 SUFFIX_PAIRS.
RUNTIME EXCLUSION (programmatic, reported): drop any pair whose base
or inflected form appears in LAKOFF_SCHEMAS_MML anchors or in the
exp138 pair set; drop OOV pairs per substrate. Report post-exclusion
N per suffix per substrate. A suffix with N < 8 in any binding
substrate is NON-BINDING there (reported anyway).

## Design (exp138/150/150c protocol, unchanged)
- Substrates: Pythia 410M (L12 primary; L4, L20 reported), GloVe
  (glove-wiki-gigaword-300) and word2vec (BINDING), fastText
  (SUPPLEMENTARY — subword architecture, per its own exp150 caveat).
  Load substrates sequentially, free memory between (16GB machine).
- Per substrate: suffix×schema matrix (7×8) from held-out pairs,
  aniso+freq strip per exp138 (All-But-The-Top top-3 PCs for static,
  per exp102 precedent).
- Statistics, DECLARED:
  T1 (ordering): r(deriv rows vs Pythia) > r(infl rows vs Pythia),
     point estimates, per binding substrate.
  T2 (the difference test — the new content): bootstrap over word
     pairs within suffix (2000 resamples), CI95 on
     Δr = r_deriv − r_infl. Count binding substrates where CI95
     excludes 0.
  T3 (static-static): GloVe↔word2vec inflectional-only r ≥ +0.50.
  T4 (sink replication): all 5 inflectional suffixes BALANCE-negative
     in Pythia at L12 on held-out pairs.

## RULE PARAMETERS (frozen; asserted vs this file)
  BOOT_N = 2000   CI = 95   MIN_PAIRS = 8   T3_MIN = 0.50

## Decision rule
- SPLIT_CONFIRMED: T1 both binding substrates AND T2 CI excludes 0
  in ≥2/2 AND T3 AND T4. → blog/paper upgrade from
  "descriptive-but-replicated" to "held-out confirmed".
- SPLIT_PARTIAL: T1 both + T2 in 1/2 (or T2 2/2 with T3 or T4
  failing). → report as directionally confirmed, difference test
  underpowered/mixed; keep "descriptive" language for whatever failed.
- SPLIT_NOT_CONFIRMED: T2 0/2 with T1 still both positive. → the
  ordering replicates but the difference is not distinguishable at
  this N; paper must say so plainly.
- SPLIT_REFUTED: T1 fails (reversed) in ≥1 binding substrate. → the
  post-hoc split was an artifact of the original pairs; Claim 3's
  keystone paragraph gets rewritten around the failure, prominently.

## Committed predictions (calibration: replication-with-controls bets
have been reliable in this lab; but T2 is a NEW statistic on few rows
— wide CIs are the honest expectation)
- P1 T4 sink replicates on fresh pairs: **85%** (the sink is the
  sturdiest object in the whole project)
- P2 T1 ordering holds in both binding substrates: **70%**
- P3 T2 CI excludes 0 in ≥2/2: **45%** (this is the hard one; N per
  row is small and Δr CIs will be wide)
- P4 T3 static-static inflectional ≥ +0.50: **80%**
- META modal outcome: SPLIT_PARTIAL **~40%**, CONFIRMED ~30%,
  NOT_CONFIRMED ~20%, REFUTED ~10%.

## Integrity
- Checksum of pairs file asserted at runtime BEFORE loading models.
- SYNTHETIC self-test first: three planted worlds — (i) split real
  (deriv aligned, infl divergent) → CONFIRMED; (ii) all rows equally
  aligned → REFUTED/NOT; (iii) split real but noisy → PARTIAL
  machinery exercises. Harness must discriminate before touching
  models.
- Sequential substrate loading; vm_stat check between; nice -n 10.
- Bench: run ONLY after the other project frees (foreground ps check).

## What I will NOT do
- No editing the pairs file after this prereg (checksum binds it).
- No swapping T2's bootstrap for a friendlier test after seeing CIs.
- No reporting fastText as binding, whichever way it goes.
- No upgrading blog/paper language unless the rule says CONFIRMED.

## RESULT + GRADES (graded 2026-07-10 03:31, grade-only protocol)
Integrity: self-test PASS; pairs checksum asserted; runtime exclusions reported; all
suffixes ≥ MIN_PAIRS in binding substrates (vocab hits 515-518/518).
Frozen rule output: **SPLIT_CONFIRMED**.
Raw verdict numbers (all at Pythia L12 vs static, held-out pairs only):
- T1 ordering: GloVe r_deriv +0.711 > r_infl +0.551; w2v +0.551 > +0.355 → both TRUE
- T2 Δr CI95: GloVe +0.177 [+0.044,+0.322]; w2v +0.206 [+0.045,+0.358] → 2/2 exclude 0
  (fastText, supplementary: +0.190 [+0.024,+0.346] — same shape)
- T3 static-static inflectional: +0.791 (≥0.50)
- T4 held-out sink, Pythia L12 BALANCE column: −0.332, −0.200, −0.258, −0.310, −0.229 → all negative
GRADES vs committed odds:
- P1 sink replicates (85%): **HIT**
- P2 T1 ordering both (70%): **HIT**
- P3 T2 CI excludes 0 in ≥2/2 (45%): **HIT** — the hard one landed
- P4 T3 ≥ +0.50 (80%): **HIT**
- META modal SPLIT_PARTIAL (~40%): **MISS** — actual outcome CONFIRMED (which I priced at 30%)
Per frozen rule: blog/paper language upgrades from "descriptive-but-replicated" to
"held-out confirmed" — the UPGRADE ITSELF IS MORNING-PASS WORK, not tonight's.

## INJECTOR ANNOTATIONS (03:40, Opus injector #2) — VERDICT DEMOTED PENDING CORRECTIVE RUN
Injector findings, all verified by me against the cited lines:
- F1 MATERIAL: prereg line 33-34 froze "All-But-The-Top top-3 PCs for static (exp102
  precedent)"; the script ran exp138's mean+freq strip on ALL substrates including the
  binding static ones. Weaker strip on the substrates carrying T1/T2/T3. The frozen design
  was also internally contradictory ("exp138/150/150c protocol unchanged" vs the ABTT
  parenthetical) — the script silently resolved to the weaker reading.
  **STATUS CHANGE: SPLIT_CONFIRMED is DEMOTED to CONFIRMED-UNDER-DEVIATION. The frozen
  upgrade clause (blog/paper language) does NOT fire on this run.** Corrective run with
  ABTT-3 static strip (the specific frozen commitment) launching immediately — same frozen
  pairs, same rule; if CONFIRMED under ABTT too, the ambiguity is moot; if not, the
  deviated run's result is reported as protocol-sensitive.
- F2: L4/L20 promised "reported," never printed — corrective run prints them.
- F3: RESULT block's "Δr" figures were bootstrap MEANS (+0.177/+0.206), not plug-in point
  estimates (+0.160/+0.196); T2 lower bounds are razor-thin (+0.044/+0.045). Corrective
  run reports plug-in + CI; the fragility is now on the record.
- Affirmed by injector: checksum valid, numbers faithful, rule logic faithful, MIN_PAIRS
  claim true, no leak vs exp150/154 (identical 147-word set), schema-direction
  contamination refuted, bootstrap calibrated (null world straddles 0), META graded
  honestly.

## CORRECTIVE RUN + FINAL ADJUDICATION (03:45, ABTT-3 static strip per frozen commitment)
**The prereg-conforming verdict is SPLIT_PARTIAL**, superseding the deviated run's CONFIRMED.
Raw (ABTT run, exp171_output_abtt.txt):
- T1 ordering: GloVe +0.630>+0.512, w2v +0.553>+0.412 → both TRUE (survives the stronger strip)
- T2 Δr plug-in: GloVe +0.118 CI[−0.010,+0.291] → includes 0; w2v +0.141 CI[+0.004,+0.320]
  → excludes, barely. 1/2 → PARTIAL branch. (fastText suppl.: +0.259 CI[+0.084,+0.444] —
  the strongest, consistent with injector #1's note that its demotion was conservative.)
- T3 +0.743 TRUE; T4 unchanged TRUE (Pythia side identical).
- L4/L20 secondaries (now reported per frozen promise): the split is LAYER-SENSITIVE —
  GloVe L4 +0.129 / L20 −0.003; w2v L4 −0.004 / L20 +0.033. L12 is where it lives.
REVISED GRADES under the prereg-conforming analysis:
- P1 sink (85%): HIT (unchanged). P2 ordering (70%): HIT (holds under ABTT).
- P3 difference-CI 2/2 (45%): **MISS** (1/2). [Deviated run had graded it HIT.]
- P4 T3 (80%): HIT. META modal PARTIAL (40%): **HIT** — my modal call was right once the
  analysis matched the prereg.
Per frozen rule: NO blog/paper upgrade; "descriptive-but-replicated, difference test
directionally consistent but not confirmed at this N under the frozen strip" is the
morning-pass language. Both runs stay on the record; the deviated run is reported as
protocol-sensitive evidence, not as the finding.
