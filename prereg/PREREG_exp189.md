# PRE-REGISTRATION — exp189: DOES THE UP READOUT TRACK THE MODEL'S SURPRISE?

Written 2026-10-02 20:42 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 20:41:
"yes add it to the queue pls". Frozen at the commit carrying this
text. What had been seen beforehand is in the hypothesis note.

## Plain summary

Token by token on ordinary sentences: is a token's position on the UP
axis related to how surprised the model was by that token, once
valence, position in the sentence, word frequency, the size of the
state and the kind of token are held fixed? And is that relation any
different from what axes built from random words show?

## Design

Models: Pythia 410M and GPT-2-medium, each analysed separately. Real
start token. Text: the lab's 120 everyday sentences.

Axes, built the clean way at layers {4, 8, 12, 16, 20} (single-token
words inside exp180's eight neutral sentences; anisotropy and
frequency directions projected out):
- **UP**: exp180's 31 UP words minus their 31 DOWN opposites.
- VALENCE: exp180's 20 positive words minus their 20 opposites (used
  as a control, and reported).
- **RANDOM-WORD NULL**: N_NULL axes, 31 random words minus 31 others,
  from the pool of all exp180 and exp185 pair words.

Per token (every position after the start token; tokens that are UP
or DOWN list words are left out):
- **surprisal (PRIMARY)**: minus the log probability the model gave
  this token, in nats;
- predictive entropy (secondary): the entropy of the model's guess
  for the NEXT token at this position;
- readout on each axis: the unit state at layer L projected on it.

Statistic, per layer: the partial correlation between the UP readout
and surprisal, holding fixed the VALENCE readout at that layer,
position, word frequency (wordfreq zipf; a flag for non-words), the
state's norm, and token type (whole word, first piece, later piece,
other). Prompt-level cluster bootstrap CI (N_BOOT). The same partial
correlation for every random-word axis.

## RULE PARAMETERS (frozen)

  N_NULL = 200   SEED_NULL = 189   N_BOOT = 1000   LAYERS_MAJ = 3 (of 5)

## Decision rule (per model)

At a layer, UP is "above random" if its partial correlation is
positive, its CI excludes zero, and it is above the null's 95th
percentile; "below random" is the mirror image.
- **UP_TRACKS_SURPRISE**: above random at >= LAYERS_MAJ layers.
  (Harder-to-predict tokens read as more UP: "unknown is up".)
- **UP_TRACKS_SURENESS**: below random at >= LAYERS_MAJ layers.
- **NOT_BEYOND_RANDOM**: anything else.
A label counts as a result only if both models return it. The same
table is reported for predictive entropy.

## Committed predictions

- P1 Pythia: NOT_BEYOND_RANDOM **70%**, UP_TRACKS_SURPRISE **20%**,
  UP_TRACKS_SURENESS **10%**
- P2 GPT-2: NOT_BEYOND_RANDOM **75%**, UP_TRACKS_SURPRISE **15%**,
  UP_TRACKS_SURENESS **10%**
- P3 the same non-null label in both models: **8%**

## What I will NOT do

- No reading a correlation against zero without the random-word null.
- No picking layers after the fact.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 20:51 IST, 2026-10-02)

Raw output: exp189_output_<model>.txt. Numbers:
exp189_results_<model>.json. Mac GPU. No deviations. 1554 tokens
(Pythia) and 1536 (GPT-2) from the 120 sentences.

### Surprisal (PRIMARY): partial correlation with the UP readout

| layer | Pythia UP [CI] | Pythia random 5–95% | UP's percentile | GPT-2 UP [CI] | GPT-2 random 5–95% | UP's percentile |
|---|---|---|---|---|---|---|
| 4 | −0.091 [−0.144, −0.034] | −0.164..+0.158 | 20th | −0.030 [−0.091, +0.025] | −0.084..+0.098 | 30th |
| 8 | −0.051 [−0.113, +0.008] | −0.134..+0.119 | 34th | −0.029 [−0.092, +0.032] | −0.112..+0.110 | 38th |
| 12 | −0.063 [−0.120, −0.006] | −0.161..+0.147 | 34th | −0.069 [−0.130, −0.006] | −0.137..+0.123 | 25th |
| 16 | −0.067 [−0.132, −0.008] | −0.169..+0.160 | 36th | −0.060 [−0.122, +0.003] | −0.159..+0.144 | 37th |
| 20 | −0.026 [−0.090, +0.034] | −0.077..+0.060 | 33rd | −0.064 [−0.124, −0.003] | −0.165..+0.160 | 38th |

**VERDICT: NOT_BEYOND_RANDOM in both models.** The UP readout's
relation to surprisal is slightly negative everywhere (more
surprising tokens read a little less UP) and sits well inside what
axes built from random words show.

### Predictive entropy (secondary)
Slightly positive in both models (+0.00 to +0.15), inside the random
range at every layer (highest: Pythia layer 4 at the 92nd
percentile). NOT_BEYOND_RANDOM.

### Notes
- The afternoon's hint (scrambled text reading more UP at later
  layers in exp175) does not carry over to tokens in ordinary text.
- Again a correlation that clears zero (several CIs exclude it) and
  is unremarkable among random axes.

### Grades
- P1 Pythia NOT_BEYOND_RANDOM (70%): HIT.
- P2 GPT-2 NOT_BEYOND_RANDOM (75%): HIT.
- P3 the same non-null label in both models (8%): did not occur.
