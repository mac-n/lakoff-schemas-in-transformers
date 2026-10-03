# PRE-REGISTRATION — exp178: WHAT PRODUCES THE BALANCE–ENTROPY COUPLING?

Written 2026-10-02 18:19 IST, BEFORE any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh asked for
it at 18:04 ("yeah do 2 and 3"); this is item 2, and the causal
rerun (item 3) follows as exp179. Frozen at the git commit that
carries this text.

Seen before freezing: exp177's full results in both models, and
tokenizer-only counts of the word lists below (no model).

## Plain summary

exp177 showed that the published BALANCE–entropy coupling depends on
how the BALANCE axis is built, and left open why. Two tests settle
it.

- **Part 1, the matched control.** Take random words that have
  nothing to do with balance and build fake axes with the same token
  split as the real one (in GPT-2: nine single-token words and six
  split words on one side, fifteen split words on the other). If the
  fake axes show the coupling, the token split alone produces it. If
  the real BALANCE axis stands apart from them, something specific to
  balance is there.
- **Part 2, a properly built clean axis.** Put balance and imbalance
  words inside neutral sentences, single tokens only, with a longer
  list (27 balance words, 22 imbalance words, the same in both
  models). Ask whether that axis is coupled to attention entropy,
  in either direction, and whether it stands apart from random clean
  axes.

## Common machinery

exp166's, imported as in exp176 / exp177: the 40 frozen prompts
(checksum 4d54ff4297bd7e2c), vocabulary and held-out split,
`build_layer_dirs`, `attn_entropy_per_query`, the quadratic control
stack, `partial_corr`, C5 axis-word exclusion, the classifiers.
Statistic: C2q. Models and judged layers as exp177: gpt2-medium
{8, 12, 16} (context 3); Llama-3.2-1B {5, 6, 7} (context 13). Each
model gets its own verdicts. Replication gate per model: the
original analysis must reproduce exp166's C2q within REPL_TOL.

## Part 1 — token-structure-matched random axes (original protocol)

Bare words, exactly as exp166. Pool: the 434 vocabulary words that
are not BALANCE words and not the frequency-anchor words. In each
model the real BALANCE poles' token structure is read off at runtime
(GPT-2: positive 9 single + 6 split, negative 0 single + 15 split;
Llama: 10 + 5 and 1 + 14).

- MATCHED: N_RAND random axes, each drawing words from the pool with
  exactly that structure, built and norm-orthogonalised exactly as
  the real axis.
- SINGLE-ONLY: N_RAND random axes with 15 single-token words on each
  side.

For each axis: C2q over the same token cloud as the original
analysis. Reported per layer: the real value; the median, 5th and
95th percentile of each random family; the real value's percentile
within MATCHED.

Decision rule (per model, judged layers, 2 of 3):
- **BALANCE_SPECIFIC**: the real BALANCE C2q is below the 5th
  percentile of MATCHED at >= 2 judged layers.
- **TOKEN_STRUCTURE_SUFFICIENT**: not BALANCE_SPECIFIC, and the
  MATCHED median is <= −DECISION_LO at >= 2 judged layers. The split
  alone gives a coupling of the published size and sign.
- **NEITHER**: anything else.

## Part 2 — clean in-sentence axis, longer lists

Words (each one token with a leading space in BOTH models; frozen):
- balance (27): balanced stable steady even level equal firm solid
  settled upright centered aligned poised tight grounded anchored
  secure fixed still regular uniform composed sturdy rooted steadfast
  flat square
- imbalance (22): unstable uneven unequal tilted tipping shaky skewed
  slack fallen rocky precarious loose leaning tipped stumbling
  slipping sliding collapsing staggering erratic volatile irregular

Frames (four; a word's state is its mean over the four):
1. she wrote the word ___ on the board before the lesson began that morning
2. the next word on the list is ___ and then we move on to the rest of the page
3. after the move the old table looked ___ and nobody in the room said anything about it
4. by the end of the week the whole arrangement seemed ___ to everyone who came to see it

Every vocabulary word is run in the same four frames, so the
anisotropy, frequency and held-out norm directions all come from
in-sentence states. The clean axis is the mean of the 27 minus the
mean of the 22, stripped and norm-orthogonalised as usual.

Cloud: whole-word tokens only (PRIMARY), with tokens that are any of
the 49 list words also excluded. All tokens reported beside it.
C2q with the prompt-level bootstrap (1000, seed 166).

CLEAN-RANDOM: N_RAND random axes, 27 against 22, drawn from pool words
that are a single token with a leading space, built the same way.

Decision rule (per model, judged layers):
- **CLEAN_COUPLING_PRESENT**: the model's exp166 classifier returns
  PASS / FIRES (negative coupling, as published).
- **CLEAN_COUPLING_REVERSED**: C2q >= +DECISION_LO with CI excluding
  zero at >= 2 judged layers. (Registered because exp177's Llama
  analysis C came out positive; that was seen before this freeze.)
- **CLEAN_COUPLING_ABSENT**: anything else.
- SPECIFIC flag, attached to PRESENT or REVERSED: the real clean
  axis lies outside the 5th–95th percentile range of CLEAN-RANDOM at
  >= 2 judged layers. Without the flag, a coupling of that size is
  what random clean axes give.

## RULE PARAMETERS (frozen)

  DECISION_LO = 0.10   CARRIER_MIN = 0.50   N_BOOT = 1000
  REPL_TOL = 0.02      N_RAND = 500         SEED_RAND = 178

## Committed predictions

- P1 both replication gates pass: **90%**
- P2 GPT-2 Part 1: TOKEN_STRUCTURE_SUFFICIENT **55%**,
  BALANCE_SPECIFIC **20%**, NEITHER **25%**
- P3 Llama Part 1: TOKEN_STRUCTURE_SUFFICIENT **40%**,
  BALANCE_SPECIFIC **15%**, NEITHER **45%**
- P4 GPT-2 Part 2: ABSENT **70%**, PRESENT **15%**, REVERSED **15%**
- P5 Llama Part 2: REVERSED **45%**, ABSENT **45%**, PRESENT **10%**
- P6 SPECIFIC flag raised in either model: **20%**

## What I will NOT do

- No changing lists, frames, layers or thresholds after seeing
  numbers.
- No reading a PRESENT or REVERSED without the SPECIFIC flag as a
  finding about balance.
- No conclusion about the causal result here; that is exp179.

## Deviations and stubs

None at freeze. Device: the Mac GPU (as exp166 was); the replication
gate doubles as the device check.

## RESULT + GRADES (graded 18:38 IST, 2026-10-02)

Raw output: exp178_output_<model>.txt. Numbers:
exp178_results_<model>.json. Mac GPU. No deviations. Both replication
gates PASS (exp166 reproduced to three decimals).

### Part 1 — random axes with BALANCE's token split

GPT-2-medium (real poles: 9 single + 6 split against 0 + 15):

| layer | real BALANCE | MATCHED median [5%, 95%] | real's percentile | SINGLE-ONLY median [5%, 95%] |
|---|---|---|---|---|
| 3 | −0.322 | −0.258 [−0.374, −0.121] | 17th | +0.007 [−0.128, +0.139] |
| 8 | −0.266 | −0.223 [−0.350, −0.061] | 30th | +0.012 [−0.214, +0.238] |
| 12 | −0.320 | −0.256 [−0.400, −0.063] | 24th | −0.003 [−0.198, +0.221] |
| 16 | −0.145 | −0.021 [−0.179, +0.146] | 12th | −0.003 [−0.154, +0.152] |

Llama-3.2-1B (real poles: 10 single + 5 split against 1 + 14):

| layer | real BALANCE | MATCHED median [5%, 95%] | real's percentile | SINGLE-ONLY median [5%, 95%] |
|---|---|---|---|---|
| 5 | −0.139 | −0.326 [−0.492, −0.048] | 87th | +0.023 [−0.313, +0.327] |
| 6 | −0.168 | −0.387 [−0.565, −0.066] | 86th | +0.017 [−0.358, +0.363] |
| 7 | −0.202 | −0.260 [−0.372, −0.106] | 77th | +0.014 [−0.203, +0.205] |
| 13 | −0.268 | −0.248 [−0.404, −0.004] | 41st | −0.004 [−0.273, +0.255] |

**Part 1 verdict, both models: TOKEN_STRUCTURE_SUFFICIENT.** Axes
built from random words with nothing to do with balance reproduce the
published coupling, as long as they have BALANCE's split of whole
tokens against chopped-up words. The real axis is unremarkable among
them (in Llama it is weaker than the typical fake). Axes with single
tokens on both sides centre on zero.

### Part 2 — clean in-sentence axis (27 against 22)

| model | judged layers | clean C2q | CLEAN-RANDOM [5%, 95%] | verdict |
|---|---|---|---|---|
| GPT-2 | 8 / 12 / 16 | −0.041 / −0.178 / −0.169 | about ±0.17 / ±0.26 / ±0.31 | CLEAN_COUPLING_PRESENT, no SPECIFIC flag |
| Llama | 5 / 6 / 7 | +0.092 / +0.134 / +0.062 | about ±0.30 / ±0.33 / ±0.24 | CLEAN_COUPLING_ABSENT |

- GPT-2's clean axis passes exp166's classifier at layers 12 and 16,
  but sits at the 15th–34th percentile of random clean axes. Per this
  prereg, without the SPECIFIC flag that is not a finding about
  balance.
- Llama's positive values from exp177 (+0.32 / +0.43 / +0.21 on nine
  imbalance words) shrink to +0.06..+0.13 with the longer list in
  sentences, well inside the random range. The flip was noise.
- **A general point that outlives this experiment:** an arbitrary
  word-contrast axis correlates with attention entropy at |r| of 0.2
  to 0.3 in this 40-prompt cloud (the middle 90% of random clean axes
  runs to about ±0.3). exp166's criterion (CI excluding zero and
  |r| >= 0.10) is met by a large share of random axes. Any future
  coupling claim needs a random-axis null, not a test against zero.

### Grades
- P1 both gates pass (90%): HIT.
- P2 GPT-2 Part 1 TOKEN_STRUCTURE_SUFFICIENT (55%): HIT.
- P3 Llama Part 1: TOKEN_STRUCTURE_SUFFICIENT occurred (given 40%;
  NEITHER was the favourite at 45%).
- P4 GPT-2 Part 2: PRESENT occurred, unflagged (given 15%; ABSENT was
  the favourite at 70%).
- P5 Llama Part 2: ABSENT occurred (given 45%, level with REVERSED).
- P6 SPECIFIC flag in either model (20%): did not occur.
