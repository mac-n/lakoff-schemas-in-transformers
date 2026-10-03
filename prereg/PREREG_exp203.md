# PRE-REGISTRATION — exp203: DOES HAPPY IS UP SURVIVE WITHOUT THE TWO VERTICAL WORDS IN THE HAPPINESS SCORE?

Written 11:43 IST, 03 Oct 2026, before any data, by Claude (Opus 5.5). Frozen at the
commit carrying this text.

## Why
The affect measure in exp116 / exp187 / exp194 is log P(happy word) −
log P(sad word) at the next token after four prompts ("Today I am
feeling" ...). The candidate words are
POS: hopeful, optimistic, excited, happy, elated, uplifted
NEG: anxious, sad, worried, depressed, hopeless, low
"uplifted" and "low" are themselves vertical words. Steering UP could
raise "uplifted" and lower "low" lexically, with no change in
happiness. Found while preparing a video of the HAPPY IS UP result
(Niamh, 11:42, asked for one). Not yet checked.

## Design
Re-run exp187's affect cells (Pythia 1.4B, layer 12) and exp194's
(GPT-2 medium, layers 6, 12, 18) at strengths +8 and −8 with the
candidate lists minus "uplifted" and "low" (5 vs 5). Same directions
(CLEAN UP, CLEAN VALENCE, and the same random-word null directions
with the same seeds), same prompts, same scoring. Built by importing
those scripts' direction construction; nothing else changes.

## Decision rule (per cell)
SURVIVES: CLEAN UP's affect effect (+8 minus −8) is above the null's
95th percentile, as before. FALLS: it is not.
Overall: HOLDS if Pythia L12 and GPT-2 L12 both SURVIVE; PARTIAL if
one; FALLS if neither.

## Predictions
Pythia L12 SURVIVES 70%; GPT-2 L12 SURVIVES 65%; HOLDS 50%.

## RESULT + GRADES (graded 11:53 IST, 03 Oct 2026)

Outputs: exp203_output_pythia.txt, exp203_output_gpt2.txt;
exp203_results_*.json. Affect scored on single-token candidates only,
as before: 4 happy (hopeful, optimistic, excited, happy; "elated" is
multi-token and was already excluded) vs 5 sad.

| cell | CLEAN UP | random-word 95th | random-word max | before (with uplifted/low) | verdict |
|---|---|---|---|---|---|
| Pythia 1.4B L12 | **+1.348** | +0.702 | +0.965 | +1.291 | SURVIVES |
| GPT-2 L6 | **+0.465** | +0.311 | +0.372 | (passed) | SURVIVES |
| GPT-2 L12 | **+0.589** | +0.397 | +0.513 | +0.586 | SURVIVES |
| GPT-2 L18 | **+0.252** | +0.128 | +0.196 | +0.255 | SURVIVES |

**HOLDS.** Removing the two vertical words changes nothing: the UP
effect is the same size or slightly larger, and in every cell it beats
not only the 95th percentile but every one of the random-word
directions. HAPPY IS UP is not "uplifted" and "low".

### Grades
Pythia L12 SURVIVES (70%): occurred. GPT-2 L12 SURVIVES (65%):
occurred. HOLDS (50%): occurred.
