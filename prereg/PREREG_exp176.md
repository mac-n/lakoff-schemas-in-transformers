# PRE-REGISTRATION — exp176: PYTHIA'S BALANCE–ENTROPY TEST, WITH A START TOKEN

Written 2026-10-02 17:27 IST, BEFORE any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh asked for
it at 17:26 ("yes prereg it"). Frozen at the git commit that carries
this text.

## Plain summary

exp166 (June) found that BALANCE is coupled to attention entropy in
GPT-2 and Llama, and not in Pythia. Today's start-token bug
(START_TOKEN_BUG_2026-10-02.md) means Pythia was the only one of the
three measured without a start token. This experiment reruns Pythia's
side of exp166 with a start token, changing nothing else, to see
whether Pythia joins the other two.

## Design

Everything is exp166's, imported and not re-typed: the 40 frozen
prompts (checksum 4d54ff4297bd7e2c), the 489-word vocabulary and its
held-out split, `collect_residuals`, `build_layer_dirs`,
`attn_entropy_per_query`, the quadratic control stack
(`covar_stacks`), `partial_corr`, the prompt-level cluster bootstrap
(1000 resamples, seed 166), the C5 axis-word exclusion, and the
classifiers `band_status` / `control_status`. The statistic is C2q.

The one change: for Pythia, `model.cfg.default_prepend_bos` is set to
True, so both the bare words that build the axes and the prompts
start with the start token. Asserted at runtime by checking that
"the" tokenises to two tokens.

Three runs, on CPU (exp175 is using the GPU; exp166 ran on the GPU, so
the gates below double as a device check):

1. **Replication gate.** Pythia WITHOUT a start token, exp166's layers
   [5, 11, 12, 18]. Must reproduce exp166's C2q (−0.047, −0.007,
   −0.048, +0.007) and carriers (0.96, 0.92, 0.91, 0.89), each within
   REPL_TOL, with n = 473. Otherwise STOP: harness drift.
2. **Positive control.** GPT-2-medium, untouched (it already has a
   start token), exp166's layers [3, 8, 12, 16]. Must return PASS
   under exp166's `control_status`, and its C2q must match exp166's
   (−0.322, −0.266, −0.320, −0.145) within REPL_TOL. Otherwise the run
   is INVALID.
3. **The test.** Pythia WITH a start token, layers
   [3, 5, 8, 11, 12, 16, 18, 20].

## RULE PARAMETERS (frozen)

  DECISION_LO = 0.10   CARRIER_MIN = 0.50   N_BOOT = 1000
  REPL_TOL = 0.02

## Decision rule

PRIMARY: layers {8, 12, 16}, the same three layers and the same rule
exp166 used for GPT-2 (`control_status`): the coupling is present if
at least 2 of the 3 layers have C2q negative, bootstrap CI excluding
0, and |C2q| >= DECISION_LO, with the carrier precondition met. These
were exp161's original decision layers for both 24-layer models,
chosen before any Pythia entropy data existed.

- **JOINS**: primary returns PASS. Pythia shows the BALANCE–entropy
  coupling once it has a start token. Three of three.
- **NULL**: primary returns FAIL. Pythia does not show it. The claim
  stays at two of three and Pythia's carrier stays unknown.
- **INVALID_CARRIER**: the held-out d_norm carrier is below
  CARRIER_MIN at a primary layer. This is likely with a start token,
  because the old carrier was the position-0 effect. In that case the
  same rule is applied without the carrier condition and reported as
  a second tier, labelled **CARRIERLESS_JOINS** or
  **CARRIERLESS_NULL**. Scalar norm and its square are still
  controlled in C2q; only the direction-level norm control is empty.
  This fallback is registered here so it cannot be a rescue after the
  fact.

SECONDARY (reported, cannot change the verdict): exp166's own Pythia
gate, band {11, 12} under `band_status`. That band came from a
24-layer scan of data taken without a start token, which is why it is
not primary.

Reported beside the verdict: C2q at every run layer with and without
the start token; n; carriers; and cos(BALANCE, d_norm_ho) with a
start token (the held-out version of today's diagnostic number).

## Committed predictions

This is a replication with a control changed, the kind of bet that
has been reliable in this lab, though exp166's own strong bet on
Pythia lost.

- P1 replication gate passes: **90%**
- P2 GPT-2 positive control passes on CPU: **95%**
- P3 carrier falls below 0.50 at a primary layer with a start token:
  **45%**
- P4 JOINS or CARRIERLESS_JOINS: **40%**
- P5 secondary band {11, 12} fires: **30%**

## What I will NOT do

- No moving the primary layers or thresholds after seeing numbers.
- No reading a context layer (3, 5, 18, 20) as a result.
- No calling a carrierless result by the plain name.
- No conclusion about Finding 5 beyond this one test: the causal
  result (exp168) and Llama are not rerun here.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 17:44 IST, 2026-10-02)

Raw output: exp176_output.txt. Numbers: exp176_results.json. Run on
CPU. No deviations.

- **Replication gate PASS.** Pythia without a start token reproduces
  exp166 to three decimals at all four layers (C2q −0.047, −0.007,
  −0.048, +0.007; carriers 0.96, 0.92, 0.91, 0.89; n 473).
- **Positive control PASS.** GPT-2-medium reproduces exp166 to three
  decimals (C2q −0.322, −0.266, −0.320, −0.145).
- **The test, Pythia with a start token (n 513):**

  | layer | C2q | 95% CI | carrier | role |
  |---|---|---|---|---|
  | 3  | +0.264 | [+0.167, +0.355] | 0.85 | context |
  | 5  | −0.007 | [−0.100, +0.087] | 0.76 | context |
  | 8  | −0.196 | [−0.268, −0.121] | 0.71 | PRIMARY |
  | 11 | −0.062 | [−0.151, +0.020] | 0.73 | secondary |
  | 12 | −0.009 | [−0.105, +0.087] | 0.72 | PRIMARY + secondary |
  | 16 | −0.016 | [−0.122, +0.082] | 0.77 | PRIMARY |
  | 18 | +0.084 | [−0.019, +0.176] | 0.80 | context |
  | 20 | +0.176 | [+0.072, +0.261] | 0.84 | context |

  Without a start token the primary layers read +0.066, −0.048,
  +0.018.

**VERDICT: NULL.** One of the three primary layers clears the rule
(layer 8); the rule needs two. Carriers are valid at every primary
layer, so the carrierless tier does not apply. Secondary band
{11, 12}: NULL.

Pythia does not join GPT-2 and Llama under the registered rule. The
claim stays at two of three. Pythia's carrier is unknown.

Recorded, not results (per "What I will NOT do"):
- Layer 8 moved from +0.066 to −0.196 when the start token was added.
  Layers 6, 7, 9 and 10 were not run, so whether it is one cell or a
  band is unknown. A band there would need its own prereg.
- Context layers 3 and 20 are POSITIVE with CIs excluding zero.
- cos(BALANCE, d_norm_ho) with a start token: +0.460, +0.425, +0.314
  at layers 3, 5, 8; −0.041, −0.053, +0.014, −0.078, −0.122 at layers
  11–20. The held-out method agrees with diag_bos_rerun.py: nothing
  from layer 11 up, something at the early layers. GPT-2 reads
  +0.491, +0.258, +0.181, −0.264 at layers 3, 8, 12, 16.

### Grades
- P1 replication gate passes (90%): HIT.
- P2 GPT-2 control passes on CPU (95%): HIT.
- P3 carrier below 0.50 at a primary layer (45%): did not occur
  (0.71–0.77).
- P4 JOINS or CARRIERLESS_JOINS (40%): did not occur.
- P5 secondary band fires (30%): did not occur.
