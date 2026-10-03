# PRE-REGISTRATION — exp199: DO INSTRUCTIONS USE THE SOURCE/GOAL ROLES? (Llama-3.2-1B base and instruct)

Written 02:45 IST, 03 Oct 2026, before any data for this experiment, by Claude (Fable
5.1). Niamh at 02:41: "so i think one thing id like to try for
tomorrow might be on llama instruct. whether the instructions use
source path goal"; at 02:43 she chose reading (A) of the hypothesis
file: "i mean A, actually could you build it and queue it tonight?"
Frozen at the commit carrying this text.

## Plain summary

An instruction names an input and an asked-for output: "turn this
draft into a summary". If the model understands a task as a journey
(PURPOSES ARE DESTINATIONS), the same probe that places the barn and
the river in a walk should place the draft as SOURCE and the summary
as GOAL. Base and instruct are compared because instruction tuning
is where tasks become the model's own business.

## Design

Llama-3.2-1B and Llama-3.2-1B-Instruct, CPU, start token, layers 3,
6, 8, 10, 13 of 16. Probe trained on exp198's LITERAL-CF minus the
three sentences containing "into" / "out of" (29 sentences, 58
points), ridge lambda 10, source = 0, goal = 1.

Test sets (32 instructions each, order balanced 16/16):
- **INSTRUCT**: plain instructions with an input noun and an output
  noun, no from/to ("Turn this draft into a summary", "Write a report
  based on these notes", "Produce a chart out of this table").
- **SWAPPED**: the same with the two nouns exchanged.
- **NO-TASK**: the same nouns with no task ("The draft and the summary
  are both attached"), scored with the task labels.
- Secondary: INSTRUCT wrapped in the Llama-3 chat template as a user
  turn, instruct model only.
Chance: N_SHUF label-shuffled probes tested on INSTRUCT.

## RULE PARAMETERS (frozen)

  N_SHUF = 300   SEED = 199   PRIMARY_LAYER = 6

## Decision rule (per model, layer 6; exp198's)

TRANSFERS: INSTRUCT above the shuffle 95th, SWAPPED above the 95th,
NO-TASK inside 5th–95th. CUE_OR_NOUN: INSTRUCT above but a condition
fails. NO_TRANSFER: INSTRUCT inside the range.

## Committed predictions

- Base TRANSFERS: **45%**; Instruct TRANSFERS: **50%**
- Instruct INSTRUCT accuracy above base's at layer 6: **55%**
- Chat template changes the instruct result by more than 0.05: **35%**

## What I will NOT do
- No changing the layer after the numbers; no dropping instructions.
- "Into" and "out of" appear in INSTRUCT; with the three literal
  sentences removed, the probe never sees them in training.

## RESULT + GRADES (graded 03:53 IST, 03 Oct 2026)

First launch crashed on a substring ("code" inside "pseudocode" in a
swapped sentence); noun finder made whole-word, re-run. Outputs:
exp199_output_Llama-3.2-1B.txt, exp199_output_Llama-3.2-1B-Instruct.txt.
Probe trained on 29 cue-free literal journeys (no from/to/into/out of).

| | layer | within-literal (LOO) | INSTRUCT | SWAPPED | NO-TASK | shuffle 95th | chat template |
|---|---|---|---|---|---|---|---|
| base | 3 | 0.71 | 0.58 | 0.58 | 0.44 | 0.62 | – |
| base | **6** | 0.97 | **0.75** | **0.78** | 0.52 | 0.66 | – |
| base | 8 | 1.00 | 0.58 | 0.56 | 0.50 | 0.66 | – |
| base | 13 | 0.84 | 0.53 | 0.52 | 0.52 | 0.62 | – |
| instruct | 3 | 0.74 | 0.72 | 0.77 | 0.41 | 0.66 | 0.70 |
| instruct | **6** | 0.97 | **0.72** | 0.66 | 0.53 | 0.67 | 0.69 |
| instruct | 8 | 1.00 | 0.58 | 0.53 | 0.50 | 0.69 | 0.53 |
| instruct | 13 | 0.88 | 0.50 | 0.50 | 0.50 | 0.62 | 0.50 |

- **Base, layer 6: TRANSFERS.** A probe that has only ever seen walks
  reads "turn this draft into a summary" with the draft as the start
  and the summary as the end, 75% of the time (chance tops out at
  66%); swap the two nouns and it follows them (78%); the same nouns
  with no task sit at chance (52%).
- **Instruct, layer 6: CUE_OR_NOUN**, by one hundredth: INSTRUCT 0.72
  clears, SWAPPED 0.66 against a 0.67 threshold. At layer 3 all three
  conditions pass (0.72 / 0.77 / 0.41), but 3 was not the named
  layer. Wrapping the instruction in the chat template changes
  little (0.69 at layer 6).
- The readout lives early (layers 3–6) and is gone by layer 10 in
  both models, where exp200 found "this word is in an instruction"
  instead.
- Instruction tuning did not sharpen it: instruct is not above base.

### Standing
In Llama-3.2-1B base, the source/goal roles learned from literal
journeys read the input and output of an instruction. Same result
in the instruct model one hundredth short of the rule at the named
layer and clear at the layer before it. Her question of 02:41, "do
the instructions use source path goal": yes for the two ends, in the
early layers, in both models (one by the frozen rule, one just
short). Together with exp200: what an instruction asks for is placed
at the goal end relative to its input, but a lone asked-for thing is
not marked as a goal against rest; the role is relational.

### Grades
- Base TRANSFERS (45%): occurred.
- Instruct TRANSFERS (50%): did not occur (CUE_OR_NOUN by 0.01).
- Instruct INSTRUCT above base's at layer 6 (55%): did not occur
  (0.72 vs 0.75).
- Chat template changes instruct by more than 0.05 (35%): did not
  occur (0.03).
