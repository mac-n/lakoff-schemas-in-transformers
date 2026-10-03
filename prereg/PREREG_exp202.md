# PRE-REGISTRATION — exp202: IS THE GOAL SPENT BY PROGRESS, OR DOES IT JUST FADE WITH DISTANCE?

Written 09:50 IST, 03 Oct 2026, before any code or data, by Claude (Opus 5.5). Niamh,
09:49: "yes please claude" (to freezing the follow-up proposed
after exp201). Frozen at the commit carrying this text.

## Why
exp201: in answers that finished, the goal state set up by an
instruction declines while the model writes (0.17 -> 0.07 of its
prompt-end strength) and settles to a floor after. That decline could
be (PROGRESS) the goal used up as the task is done, which is Niamh's
"it would tail off" as I translated it, or (DISTANCE) the instruction
simply getting further away. exp201 cannot tell them apart.

## The test
Ask for the same form at two lengths. A long answer and a short one
are at the same FRACTION DONE at different token distances, and at
the same TOKEN DISTANCE at different fractions done.
- PROGRESS predicts: at the same fraction done, long = short; at the
  same token distance, long > short (the long one has more left to do).
- DISTANCE predicts: at the same token distance, long = short; at the
  same fraction done, long < short (the long one is further away).

## Model and stimuli
Llama-3.2-1B-Instruct, as exp201 (HF, float32, MPS, chat template,
greedy). MAX_NEW tokens; answers that hit it are dropped as unfinished.
4 forms x 2 lengths x exp201's 20 topics = 160 instructions:
- haiku: "Write one haiku about {t}." / "Write three haikus about {t}."
- limerick: "Write one limerick about {t}." / "Write three limericks about {t}."
- facts: "Write a numbered list of three facts about {t}." / "... of eight facts about {t}."
- story: "Write a two-sentence story about {t}." / "Write a six-sentence story about {t}."
READER control as exp201: "Write something about {t}." followed by the
same answer, teacher-forced.

## Measure
Layers: 2 (PRIMARY: the layer exp201's rule chose) and 8 (reported).
Goal directions u_f from prompt-end states over the 4 forms (both
lengths pooled), leaving the prompt's own topic out (as exp201).
Per answer token k: D(k) = signal(instructed) − signal(reader), with
signal(h, f) = h·u_f − mean over the other forms of h·u_g; in units of
the mean prompt-end signal S0 of the instructed runs.

A PAIR = the short and long answer for one form and topic, both
finished, long at least LEN_RATIO/10 times the short in tokens.
- Fraction axis: 10 bins of k/n. Delta_p = mean over bins 2-10 of
  (G_long − G_short).
- Token axis: K_f = median short length for form f; 10 bins of width
  K_f/10 over k in [0, K_f). Delta_k = mean over those bins of
  (G_long − G_short) (bins a short answer does not reach are skipped
  for that pair).
Both averaged over pairs within form, then over forms. Bootstrap:
N_BOOT resamples of topics.

## RULE PARAMETERS (frozen)

  MAX_NEW = 600   LEN_RATIO = 15   N_BOOT = 2000   SEED = 202

## Decision rule (layer 2; precedence top to bottom)
1. **NO_DECLINE**: the slope of G over fraction-done bins 1-10, pooled
   over all finished answers, has CI including or above 0 (nothing to
   explain).
2. **PROGRESS**: Delta_k CI above 0 AND Delta_p CI includes 0.
3. **DISTANCE**: Delta_p CI below 0 AND Delta_k CI includes 0.
4. **BOTH**: Delta_k CI above 0 AND Delta_p CI below 0.
5. **UNRESOLVED**: anything else.
Reported: per form, layer 8, number of pairs kept, the length ratio
achieved, and POST (the "Thank you!" turn, as exp201).

## Committed predictions
- NO_DECLINE 5%, PROGRESS 25%, DISTANCE 35%, BOTH 20%, UNRESOLVED 15%.
- At least 40 pairs kept: 75%.

## What I will NOT do
- No changing the layer, bins or pairing after seeing data.
- No reading a per-form result as the verdict.

## RESULT + GRADES (graded 10:15 IST, 03 Oct 2026)

Output: exp202_output.txt; exp202_results.json; caches exp202_cache.npz,
exp202_proj.npz. All 160 answers finished; 76 pairs kept (haiku 20,
limerick 17, facts 19, story 20); length ratios 2.3-3.7.

**Layer 2: UNRESOLVED. Layer 8: UNRESOLVED.**
The goal signal declines over the answer again (slope CI −0.007 to
−0.005 at layer 2), replicating exp201. But long answers sit LOWER than
short ones on BOTH axes:
- same fraction done: long − short = −0.056 (CI −0.063 to −0.048)
- same token distance: long − short = −0.058 (CI −0.069 to −0.047)
Progress predicts the second to be positive; distance predicts it to
be zero. Neither happened.

### Post-hoc (not graded)
1. Asking for more weakens the goal at the start. The prompt-end
   signal for "three haikus" is 0.35 of that for "one haiku" at layer
   2 (0.76 at layer 8); limerick 0.90, story 0.90, facts 1.04.
2. Scaling each length by its own starting strength does not rescue
   either reading: long still lower on both axes (layer 2, all forms:
   −0.025 [−0.034, −0.013] and −0.023 [−0.039, −0.005]; same pattern
   without facts and at layer 8).
3. Lists of facts carry almost no goal signal beyond what the text
   shows at any point (per token +0.005 / −0.001). For the other three
   forms, the total signal summed over the answer is 1.7-2.0x larger
   for the long version at layer 2, against 2.3-3.7x the tokens: in
   between "a fixed amount spread over the task" (1x) and "a fixed
   amount per token" (the length ratio). No clean law.

### Standing
The decline over an answer replicates. What drives it is not settled:
a bigger task holds less goal signal per token everywhere, at every
point, which is neither "spent by progress" nor "fades with distance"
as I set them up. My test assumed the two lengths start from the same
goal, and they don't. A fairer design would hold the instruction fixed
and vary length some other way (e.g. a topic that needs more words),
or compare at matched prompt-end strength. Not attempted.

### Grades
NO_DECLINE 5 / PROGRESS 25 / DISTANCE 35 / BOTH 20 / UNRESOLVED 15:
UNRESOLVED occurred. At least 40 pairs kept (75%): occurred (76).
