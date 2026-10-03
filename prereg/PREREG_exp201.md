# PRE-REGISTRATION — exp201: WHAT HAPPENS TO THE GOAL OF AN INSTRUCTION WHILE THE MODEL CARRIES IT OUT?

Written 09:31 IST, 03 Oct 2026, before any code or data for this experiment, by Claude
(Opus 5.5). Frozen at the commit carrying this text.

Niamh, 09:25: "i was hopign we could find out something about how it
represents actual goals in instructions"; 09:30: "yes please" (to
freezing steps 1-3 after the literature check,
LIT_goals_in_instructions_2026-10-03.md).

Her prediction, made on 2 Oct about a different quantity (the UP
lead) and recorded in HYPOTHESIS_UP_grounding_2026-10-02.md line 208:
"i think you would actually see it being spent, if its potential
energy. so i think it would tail off!" Applied here, with her
permission to be checked: the goal is SPENT as it is reached. That is
my translation, not her words about this experiment.

Rival shape from the literature (Dong et al. 2025, Fig. 7): the plan
reads strongly at the start, dips mid-response, returns near the end
(U). Third shape: HELD (flat through the response, drops after it).

## Model and stimuli

Llama-3.2-1B-Instruct (cached unsloth copy), HF transformers, float32,
MPS, its own chat template. Greedy decoding, at most MAX_NEW tokens,
stop at end-of-turn.

6 forms x 20 topics = 120 instructions:
- haiku: "Write a haiku about {t}."
- limerick: "Write a limerick about {t}."
- list: "Write a numbered list of five facts about {t}."
- email: "Write a short email to a friend about {t}."
- joke: "Tell a short joke about {t}."
- story: "Write a three-sentence story about {t}."
Topics: rain, the sea, a cat, coffee, winter, a train, friendship,
the moon, gardening, a lost key, mountains, bread, a library,
thunder, a bicycle, autumn leaves, a lighthouse, homework, a
birthday, the city at night.
READER condition (step 3 control): the user turn is replaced by
"Write something about {t}." and the model's OWN response to the
form instruction is teacher-forced after it. Same tokens, no asked-for
form.

## Step 1 (gate; replicates Dong et al.): is the goal held before the answer starts?

State at the last prompt token (the end of the assistant header),
layers 2-14. Ridge one-vs-rest classifier (lambda 10, standardised),
6 forms, cross-validated with TOPIC held out (leave-one-topic-out), so
topic cannot carry it. LAYER = the layer with the highest CV accuracy
(ties: earlier). Gate: accuracy at LAYER above the 95th percentile of
N_PERM label-permuted runs. Chance = 1/6.

## Step 2 (gate; replicates task-vector work): is that state causal?

For each target prompt i, source j = same topic, next form (cyclic).
Replace i's LAYER state at the last prompt position with j's; teacher-
force j's own response y_j after i's prompt; effect = mean per-token
log-prob of y_j, patched minus unpatched. Control: the same with a
source of i's OWN form, next topic. Gate: mean (real − control) > 0
with 2000-sample bootstrap CI over prompts excluding 0.

## Step 3 (the finding): the goal signal through and after the answer

Goal directions (at LAYER, prompt end): u_f = unit (mean state of form
f − grand mean), computed leaving out the prompt's own topic.
For a state h, signal(h, f) = h·u_f − mean over g ≠ f of h·u_g, with h
first centred by the mean over all prompts of the same condition at
the same position bin.

Positions: the response (excluding end-of-turn) split into 10 bins by
normalised position (bin = floor(10 k / n)); responses under 10
tokens are dropped and counted. POST bin: the end-of-turn token plus
the tokens of "<|start_header_id|>user<|end_header_id|>\n\nThank
you!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
appended after the response.

G(bin) = [signal in the instructed run − signal in the READER run]
averaged over tokens in the bin, divided by the mean prompt-end signal
G0 of the instructed runs. So G is "how much of the original goal
state is present beyond what the text itself shows", in units of its
starting strength.

Bootstrap: 2000 resamples of prompts.

## RULE PARAMETERS (frozen)

  MAX_NEW = 100   N_PERM = 100   SEED = 201   N_BOOT = 2000

## Decision rule for step 3 (precedence top to bottom)

1. **NO_GOAL_STATE**: CI of G(bin 1) includes 0 or lies below it.
2. **U_SHAPED**: mean G(bins 4-7) below G(bin 1) AND below G(bin 10),
   both difference CIs excluding 0.
3. **SPENT**: the slope of G over bins 1-10 has CI below 0 AND point
   G(bin 10) < 0.5 x G(bin 1).
4. **HELD**: slope CI includes 0 or is above 0, AND mean G(bins 1-10)
   minus G(POST) has CI above 0 with point G(POST) < 0.5 x that mean.
5. **OTHER**: none of the above (described as found).
If step 1 or step 2 fails its gate, step 3 is still run and reported
but labelled UNGATED.

## Committed predictions

- Step 1 gate passes: **90%**. Step 2 gate passes: **65%**.
- Step 3: SPENT **30%**, U_SHAPED **25%**, HELD **15%**, NO_GOAL_STATE
  **10%**, OTHER **20%**.
- Reported, not graded: whether each form was actually produced
  (simple text checks), and G by form.

## What I will NOT do
- No changing LAYER after step 3 is seen (it is set by step 1's rule).
- No dropping prompts except by the under-10-token rule.
- No reading HELD or SPENT into OTHER.

## RESULT + GRADES (graded 09:46 IST, 03 Oct 2026)

Output: exp201_output.txt; exp201_results.json; caches exp201_cache.npz,
exp201_bins_L2.npz.

- **Step 1: PASS.** The asked-for form is read perfectly (1.00, topic
  held out; permutation 95th 0.23) at every layer 2-14. The tie rule
  therefore picked LAYER = 2, the earliest; a very early layer for a
  "goal", but that is what the rule said.
- **Step 2: PASS.** Swapping in another form's prompt-end state at
  layer 2 raises the log-prob of that form's answer by +0.0105 nats
  per token over a same-form swap (CI +0.004 to +0.017). Small, but
  causal, at one position in one layer.
- **Step 3: SPENT** by the frozen rule. G (goal signal beyond what the
  text shows, in units of its strength at the prompt end):

| bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | POST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 120 | +0.15 | +0.09 | +0.11 | +0.09 | +0.07 | +0.05 | +0.07 | +0.07 | +0.04 | +0.05 | +0.07 |

  slope −0.009 (CI −0.010 to −0.008); G(10) is a third of G(1).

### Two problems, found post-hoc, that limit what SPENT means
1. **48 of 120 answers never finished.** MAX_NEW = 100 was too small:
   all 20 lists, 14 emails and 12 stories were cut off. For those,
   the last bin is not the goal being reached. On the 72 answers that
   DID finish (haiku, limerick, joke and some emails and stories;
   mean 38 tokens): G falls from +0.17 to +0.07 (slope CI −0.012 to
   −0.008), and after the answer is complete it does NOT fall further
   (POST +0.08; bin 10 − POST CI −0.02 to 0.00). So: it declines while
   the work is done, then levels off at a floor once it's done.
2. **Progress and distance are not separated.** Splitting by length,
   longer answers sit lower from bin 2 onward (long half +0.02 to
   +0.06; short half +0.07 to +0.17), which is what fading with
   DISTANCE from the instruction predicts, not progress toward done.
   But the long half is mostly the cut-off forms, so form and length
   are tangled; within haiku the lengths barely differ (17 vs 19), so
   that check is empty.
Also: my form check for lists was wrong (the model writes "1. **Fact**:"
and the check missed it); report-only, no rule depends on it.

### Standing
Within the 72 completed answers, the goal state an instruction sets
up declines while the model writes, and settles to a floor (about half
its early level) once the answer is done, without vanishing. That
matches Niamh's "it would tail off" for the working phase; her "spent
and tail off after the sentence is finished" is not seen: nothing more
is lost after completion. Whether the decline tracks progress or just
distance from the instruction is NOT settled here.

### The follow-up this needs
Same form, different lengths asked for: "a haiku" vs "three haikus",
"three facts" vs "ten facts", MAX_NEW 400 so everything finishes. If
the goal is spent by PROGRESS, curves overlay when plotted against
fraction done; if it fades with DISTANCE, they overlay against tokens
since the instruction. That is the test that answers her question.

### Grades
- Step 1 gate (90%): passed. Step 2 gate (65%): passed.
- Step 3: SPENT (30%) occurred by the rule; graded as occurred, with
  the two problems above attached to it.
