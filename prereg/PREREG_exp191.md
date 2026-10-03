# PRE-REGISTRATION — exp191: DOES THE SUFFIX FINDING SURVIVE A CLEAN TEST?

Written 2026-10-02 22:01 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 22:00:
"i guess we should check if the suffix finding survives too." Frozen
at the commit carrying this text.

Seen before freezing: today's diagnostic that the sink roughly halves
when Pythia is given a start token (diag_bos_rerun.py, old protocol
otherwise), and tokenizer-only counts of the pairs below.

## Plain summary

Finding 4 says inflected words (walked, taller, cats) are shifted
toward the "imbalance" side of the BALANCE axis in Pythia, and that
GloVe does not do this. In the old protocol the base words were
mostly single tokens and the inflected words mostly split, and the
BALANCE axis itself pitted single-token words against split ones. So
the "shift" may have been tokens against fragments on both sides.

Inside sentences most of these words are single tokens. This reruns
the finding using only pairs where BOTH forms are a single token, a
BALANCE axis with no fragments in it, a start token, and a comparison
with axes built from random words.

## Design

Pythia 410M, real start token. Layer 12 is PRIMARY (as exp171);
layers 4, 8, 16, 20 reported. Every word's state is taken at its own
position inside exp180's eight neutral sentences and averaged.

Suffix pairs: exp154's original pairs plus exp171's held-out pairs,
kept only if both forms are one token with a leading space (counts
before exclusions: -er 26, -est 15, -ing 30, -ed 24, plural 27, un-
18, re- 9). A pair is dropped if either form is on any axis list. A
suffix with fewer than MIN_PAIRS pairs is reported but not binding.
Suffix direction: the mean over pairs of (unit inflected state minus
unit base state), anisotropy and frequency directions projected out
(exp154's construction).

Axes, all from single-token words, built the same clean way:
- **BALANCE**: exp178's 27 balance words minus 22 imbalance words.
- FORWARD-BACK and the other schemas: the lab's lists as filtered to
  single tokens in exp175_stimuli.json; UP from exp175's spatial
  lists.
- **RANDOM-WORD NULL**: N_NULL axes, 27 random words minus 22 others,
  from single-token pool words that are not on the BALANCE lists and
  not in any suffix pair.

The sink: for each inflectional suffix (-er, -est, -ing, -ed,
plural), the cosine between its direction and the BALANCE axis; and
their mean. The same mean is computed for every random-word axis.

GloVe (glove-wiki-gigaword-300), the same words, pairs and lists,
with the static strip exp171 froze (all-but-the-top, three
components): the same sink and the same null.

## RULE PARAMETERS (frozen)

  MIN_PAIRS = 8   N_NULL = 300   SEED_NULL = 191

## Decision rule (Pythia, layer 12)

- **SINK_SURVIVES**: every binding inflectional suffix is negative on
  BALANCE AND their mean is below the null's 5th percentile.
- **SINK_WEAK**: the mean is below the null's 5th percentile but not
  every suffix is negative.
- **SINK_REVERSED**: the mean is above the null's 95th percentile.
- **SINK_GONE**: the mean is inside the null's 5th–95th range.

The transformer-versus-static claim: **DISSOCIATION_SURVIVES** only if
Pythia returns SINK_SURVIVES and GloVe's mean sits inside GloVe's own
null range. Otherwise **DISSOCIATION_NOT_SUPPORTED**.

Reported: each suffix on each of the eight clean axes at every layer;
-ed on FORWARD-BACK with its percentile among random axes; un- and
re- the same way.

## Committed predictions

- P1 SINK_GONE **60%**; SINK_SURVIVES **15%**; SINK_WEAK **15%**;
  SINK_REVERSED **10%**
- P2 DISSOCIATION_SURVIVES: **8%**

## What I will NOT do

- No changing pairs, lists, layer or the rule after seeing numbers.
- No reading a negative cosine against zero without the random-word
  null.

## Deviations and stubs

Flagged at freeze: this tests the sink and the Pythia-against-GloVe
contrast only. It does not rerun exp150's full suffix-by-schema
matrix comparison across three static substrates, or exp153's
layer-by-layer emergence.

## RESULT + GRADES (graded 00:11 IST, 03 Oct)

Raw output: exp191_output.txt. Numbers: exp191_results.json. CPU. No
deviations. Pairs after exclusions: -er 19, -est 10, -ing 26, -ed 23,
plural 27, un- 17, re- 9 (all binding). 673 words; GloVe had all of
them.

### The sink (mean of the five inflectional suffixes on the clean BALANCE axis)

| substrate | sink | random-word axes 5th..95th | percentile | verdict |
|---|---|---|---|---|
| Pythia layer 4 | −0.079 | −0.121..+0.117 | 16th | SINK_GONE |
| Pythia layer 8 | −0.077 | −0.132..+0.147 | 20th | SINK_GONE |
| **Pythia layer 12 (PRIMARY)** | **−0.093** | −0.131..+0.134 | 15th | **SINK_GONE** |
| Pythia layer 16 | −0.084 | −0.133..+0.138 | 17th | SINK_GONE |
| Pythia layer 20 | −0.082 | −0.239..+0.235 | 32nd | SINK_GONE |
| GloVe | −0.027 | −0.087..+0.079 | 33rd | SINK_GONE |

**VERDICT: SINK_GONE (Pythia, layer 12). DISSOCIATION_NOT_SUPPORTED.**

### Per suffix, Pythia layer 12 (cosine with the clean BALANCE axis; percentile among random axes)
-er −0.086 (20th), -est −0.007 (44th), -ing −0.266 (3rd), -ed −0.035
(41st), plural −0.069 (15th); un- −0.456 (0th), re- −0.100 (14th).

### What this says
- With single-token pairs on both sides, a fragment-free BALANCE
  axis and a start token, the old −0.38..−0.47 for -er is −0.09, and
  the mean over the inflectional suffixes sits inside what random
  word-contrast axes give, at every layer. GloVe is the same. The
  transformer-against-static contrast (Pythia −0.38, GloVe −0.01) is
  not there once both are built the same clean way.
- One suffix is below the random range on its own: -ing (−0.27 at
  layer 12, 3rd percentile; 4th–6th at layers 4–16; 9th in GloVe).
  Not a registered test; one suffix of five; noted as a lead only.
- un- is strongly BALANCE-negative in both substrates (−0.46 Pythia,
  −0.45 GloVe, 0th percentile) and equally LIGHT-DARK-negative. The
  clean BALANCE list still contains three un- words (unstable,
  uneven, unequal), so the un- direction overlaps the axis by
  construction; and it is the same in GloVe, so it is not a
  transformer effect. Same lesson as exp186.
- "-ed sits on FORWARD-BACK-negative (the past is behind you)":
  Pythia −0.11 at layer 12, 26th percentile, inside the random range.
  GloVe −0.21, at its 5th percentile. If anything the static space
  shows it more.

### Grades
- P1 SINK_GONE (60%): HIT.
- P2 DISSOCIATION_SURVIVES (8%): did not occur.
