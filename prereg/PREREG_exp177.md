# PRE-REGISTRATION — exp177: IS THE BALANCE–ENTROPY COUPLING A WORD-FRAGMENT EFFECT?

Written 2026-10-02 17:53 IST, BEFORE any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh asked for
it at 17:51 ("yeah we should test the word fragments reading").
Frozen at the git commit that carries this text.

Seen before freezing: tokenizer-only counts of how the BALANCE pole
words split into tokens (no model). Nothing else.

## Plain summary

The BALANCE axis is "mean state of the balance words minus mean state
of the imbalance words", each word run on its own. In GPT-2, 9 of the
15 balance words are a single token and NONE of the 15 imbalance
words is ("unbalanced", "wobbly" and the rest are split into pieces,
and the state taken is the last piece). In Llama it is 10 of 15
against 1 of 15. So the axis may partly be a "whole word versus last
fragment of a word" axis, and the coupling with attention entropy
could come from that and not from balance.

Written with a leading space, as words appear inside sentences, 14 of
the 15 balance words and 9 of the 15 imbalance words are single
tokens in GPT-2 (15 and 9 in Llama). That makes a fragment-free
version of the axis possible, and this experiment asks whether the
coupling survives it.

## Design

exp166's machinery, prompts (checksum 4d54ff4297bd7e2c), vocabulary,
control stack, bootstrap (1000 resamples, seed 166) and classifiers,
imported as in exp176. Statistic: C2q.

Models: gpt2-medium at layers [3, 8, 12, 16], judged by exp166's
`control_status` on {8, 12, 16}; Llama-3.2-1B at layers [5, 6, 7, 13],
judged by exp166's `band_status` on {5, 6, 7}. Each model gets its own
verdict. Pythia is not included: it showed no coupling to explain.

Every token in the sentence cloud is typed: WHOLE (a word that is one
token), INITIAL_PIECE (first piece of a split word), CONT_PIECE (a
later piece), OTHER (punctuation, digits, anything not purely
letters).

Analyses, per model:
- **P0, replication gate.** exp166 exactly. Must reproduce exp166's
  C2q within REPL_TOL at every layer. Otherwise STOP.
- **A, whole words only.** The original axis; the cloud restricted to
  WHOLE tokens.
- **B, tokenization direction controlled.** A held-out direction
  d_tok_ho is built like d_norm_ho, on the same estimation half of
  the vocabulary, from "this word is more than one token" in place of
  norm. The BALANCE axis is projected off both d_norm_ho and d_tok_ho.
  Each token's projection on d_tok_ho and its type (three dummies)
  join the control stack.
- **C0, leading-space protocol, all pole words.** Every vocabulary
  word is run as start token + " word". Everything else as P0.
  Reported to show what the change of protocol does by itself.
- **C, fragment-free axis (BINDING).** The leading-space protocol,
  with the BALANCE axis built ONLY from pole words that are a single
  token (GPT-2: 14 against 9; Llama: 15 against 9), projected off that
  protocol's d_norm_ho. Cloud restricted to WHOLE tokens. In this
  analysis neither the axis nor the cloud contains a fragment.

Reported beside them: how many cloud tokens are of each type; mean
entropy and mean BALANCE projection by type; cos(original BALANCE,
d_tok_ho); cos(original BALANCE, fragment-free BALANCE); C2q and CI
for every analysis at every layer.

## RULE PARAMETERS (frozen)

  DECISION_LO = 0.10   CARRIER_MIN = 0.50   N_BOOT = 1000
  REPL_TOL = 0.02      MIN_POLE = 5

"Survives" means the model's own classifier returns PASS (GPT-2) or
FIRES (Llama) for that analysis.

## Decision rule (per model)

- **FRAGMENT_READING_REJECTED**: C survives. The coupling holds for a
  BALANCE axis with no whole-word / fragment contrast in it, among
  whole-word tokens.
- **FRAGMENT_READING_SUPPORTED**: C fails while C0 survives. Removing
  the fragments, and nothing else, removes the coupling.
- **UNRESOLVED**: C fails and C0 fails. The change of protocol breaks
  the coupling before fragments are removed, so the failure cannot be
  pinned on fragments.
- An INVALID from the classifier (carrier below CARRIER_MIN) in C is
  reported as UNRESOLVED_CARRIER, with the carrierless reading beside
  it as in exp176.

A and B cannot change the verdict. If C survives while A or B fails,
that is flagged as INCONSISTENT in the report.

## Committed predictions

- P1 both replication gates pass: **90%**
- P2 GPT-2: A survives **80%**; B survives **65%**
- P3 GPT-2 verdict: REJECTED **55%**, SUPPORTED **25%**,
  UNRESOLVED **20%**
- P4 Llama verdict REJECTED: **45%**

## What I will NOT do

- No changing the binding analysis, layers or thresholds after seeing
  numbers.
- No adding words to the BALANCE poles to rescue a thin axis.
- No reading the causal result (exp168) into or out of this: it is
  not rerun here. If the verdict is SUPPORTED, exp168 needs its own
  rerun with the fragment-free axis before it can be cited.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 18:18 IST, 2026-10-02)

Raw output: exp177_output_gpt2-medium.txt, exp177_output_Llama-3.2-1B.txt.
Numbers: exp177_results_<model>.json. Run on CPU. No deviations.
Both replication gates PASS (exp166 reproduced to three decimals).

### GPT-2-medium (judged layers 8 / 12 / 16; L3 is context)

| analysis | n | L3 | L8 | L12 | L16 | classifier |
|---|---|---|---|---|---|---|
| P0 exp166 exactly | 513 | −0.322 | −0.266 | −0.320 | −0.145 | PASS |
| A original axis, whole words only | 329 | −0.467 | −0.446 | −0.346 | −0.256 | PASS |
| B tokenization direction controlled | 513 | −0.121 | −0.076 | −0.134 | −0.150 | PASS |
| C0 leading space, all pole words | 513 | −0.150 | −0.067 | −0.239 | −0.106 | FAIL |
| **C fragment-free axis, whole words** | 329 | −0.036 | −0.047 | −0.098 | +0.010 | **FAIL** |

Verdict: **UNRESOLVED** (C fails and C0 fails).

### Llama-3.2-1B (judged layers 5 / 6 / 7; L13 is context)

| analysis | n | L5 | L6 | L7 | L13 | classifier |
|---|---|---|---|---|---|---|
| P0 exp166 exactly | 504 | −0.139 | −0.168 | −0.202 | −0.268 | FIRES |
| A original axis, whole words only | 334 | −0.093 | −0.130 | −0.068 | −0.189 | NULL |
| B tokenization direction controlled | 504 | +0.132 | +0.109 | +0.054 | −0.046 | NULL |
| C0 leading space, all pole words | 504 | +0.259 | +0.300 | +0.021 | −0.239 | NULL |
| **C fragment-free axis, whole words** | 334 | +0.318 | +0.431 | +0.211 | +0.018 | **NULL** |

Verdict: **UNRESOLVED** (C fails and C0 fails).

### What the tables say, beyond the labels
- In neither model does the published negative coupling survive a
  BALANCE axis with no whole-word / fragment contrast in it.
- GPT-2: fragments in the sentences are not the cause (A is
  stronger than P0). The axis is: same whole-word tokens, original
  axis −0.45 / −0.35 / −0.26, fragment-free axis −0.05 / −0.10 /
  +0.01.
- Llama: the coupling weakens under every control, and with the
  fragment-free axis it is POSITIVE at the judged layers (+0.32,
  +0.43, +0.21, CIs excluding zero). That reversal was not a
  registered outcome. It is one observation on nine imbalance words
  and needs its own test before it means anything.
- The original BALANCE axis is substantially a tokenization axis:
  cos(BALANCE, d_tok_ho) is −0.44..−0.54 in GPT-2 and −0.32..−0.39
  in Llama.
- "Unresolved" is about attribution only: the change of protocol
  (leading space) already breaks the coupling before fragments are
  removed, so the rule cannot say which of the two did it. That the
  coupling depends on how the axis was built is not in doubt.
- exp168 (the causal result) was not rerun. A post-hoc check of
  whether each schema's exp168 slope tracks its pole token gap was
  inconclusive (r −0.44 over 8 schemas, driven by BALANCE).

### Grades
- P1 both replication gates pass (90%): HIT.
- P2 GPT-2 A survives (80%): HIT. B survives (65%): HIT, narrowly.
- P3 GPT-2 verdict: UNRESOLVED occurred (given 20%; REJECTED was the
  favourite at 55% and did not occur).
- P4 Llama REJECTED (45%): did not occur.
