# PRE-REGISTRATION — exp181: CAN A MODEL CHANGE ITS ATTENTION ENTROPY ON REQUEST?

Written 2026-10-02 18:59 IST, BEFORE any analysis code or model run for
this experiment, by Claude (session labelled Fable 5.1). Niamh's idea
(18:06, 18:08, verbatim in HYPOTHESIS_UP_grounding_2026-10-02.md):
"id also like to see if models can control their attention entropy"
... "meant control on request. focus/relax ur attention." Frozen at
the git commit that carries this text.

Seen before freezing: tokenizer-only token counts of the instruction
sentences below. Nothing else. Related work is listed in the
hypothesis note; none of it tests this.

## Plain summary

Ask a model to keep its attention sharply focused, or loosely relaxed,
while it reads a sentence. Then measure, on that same sentence, how
spread out its attention actually is. If "focus" gives sharper
attention than "relax", something moved. Two controls decide what:

- **Not addressed to the model.** The same words about someone else
  ("She kept her attention sharply focused while reading the text
  below."). If the shift is just as big, the words did it, not the
  request.
- **A model that was never trained to follow instructions.** The base
  Llama beside its instruction-tuned sibling.

## Design

Models: Llama-3.2-1B-Instruct (unsloth copy) and Llama-3.2-1B (base).
Same architecture (16 layers, 32 heads) and tokenizer. Hugging Face
transformers with attention outputs; no TransformerLens.

Texts: the lab's 120 everyday sentences (the 40 of attn_entropy_lib,
the 40 of exp161, the 40 of exp166).

Instructions: four templates, each with a focus and a relax version of
the SAME token length (checked), in a "you" form and an "other
person" form:
1. Keep your attention sharply focused / loosely relaxed while
   reading the text below. — She kept her attention ... while reading
   the text below.
2. Read the next passage with narrow, intense / broad, gentle
   attention. — They read the next passage with ...
3. Pay close / loose attention to every word of the text that
   follows. — He paid close / loose attention to ...
4. Let your attention narrow to a single point / widen to the whole
   scene as you read. — She let her attention ... as she read.
Plus one neutral instruction ("Read the text below."), reported only.

Formats:
- PLAIN: instruction, blank line, sentence. Both models.
- CHAT: the same content as a user message in the model's chat
  template, with the assistant header after it. Instruct model only.
The script asserts that the sentence's tokens are identical in every
condition.

Measure, on the sentence's tokens only:
- **Within-text attention entropy (PRIMARY).** For each sentence
  token, each head's attention is restricted to the sentence's own
  earlier tokens and renormalised; Shannon entropy divided by the log
  of the number of those tokens; averaged over heads, tokens, and all
  16 layers. Lower means more concentrated. Restricting to the
  sentence keeps attention paid to the instruction words out of the
  number.
- Reported beside it: entropy over everything the token can see; the
  share of attention going to the instruction words; the per-layer
  profile.

Per sentence, averaged over the four templates:
- D_you = entropy(focus, you) − entropy(relax, you)
- D_other = the same for the other-person versions
- I = D_you − D_other
Tests across the 120 sentences: mean, d_z, sign-flip permutation p
(N_PERM, two-sided).

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000

## Decision rule (per model and format)

- **CONTROL_ON_REQUEST**: D_you < 0 with p < ALPHA, AND I < 0 with
  p < ALPHA. Focus sharpens attention relative to relax, and more so
  when the model is the one addressed.
- **WORD_EFFECT_ONLY**: D_you < 0 with p < ALPHA, but I is not below
  zero at p < ALPHA. The words shift attention whoever they are
  about.
- **REVERSED**: D_you > 0 with p < ALPHA.
- **NULL**: anything else.

PRIMARY cell: the instruct model in CHAT format (how it receives
requests). Reported: instruct PLAIN, base PLAIN, and
TUNING = I(instruct, PLAIN) − I(base, PLAIN), paired over sentences.

## What a CONTROL_ON_REQUEST would and would not mean

It would mean that an instruction addressed to the model changes how
it spreads attention over later text, beyond what the same words do
when addressed to nobody. It would not show volition, and a critique
published last month (Aoki et al., arXiv:2609.00904) is that this
kind of control can come from surface features of the prompt. The
other-person control addresses one such feature, not all of them.

## Committed predictions

- P1 PRIMARY cell: NULL **50%**, WORD_EFFECT_ONLY **30%**,
  CONTROL_ON_REQUEST **10%**, REVERSED **10%**
- P2 base PLAIN returns the same label as instruct PLAIN: **70%**
- P3 TUNING differs from zero at p < 0.05: **15%**

## What I will NOT do

- No choosing a layer, head or template after seeing numbers; the
  primary number is the all-layer mean over the four templates.
- No calling WORD_EFFECT_ONLY control.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 19:12 IST, 2026-10-02)

Raw output: exp181_output.txt. Numbers: exp181_results.json; rows in
exp181_rows_<model>_<format>.jsonl. Mac GPU. No deviations. Self-test
PASS; sentence tokens identical across conditions (asserted); focus
and relax instructions equal in token length (asserted).

Within-text attention entropy (PRIMARY measure; its level under the
neutral instruction is about 0.745), focus minus relax, over 120
sentences:

| cell | addressed to the model | about someone else | difference (I) | verdict |
|---|---|---|---|---|
| **instruct, chat (PRIMARY)** | +0.00112 (d_z +1.12, p 0.0001) | +0.00024 (p 0.003) | +0.00088 (p 0.0001) | **REVERSED** |
| instruct, plain | +0.00008 (p 0.32) | +0.00031 (p 0.0001) | −0.00022 (p 0.006) | NULL |
| base, plain | −0.00093 (d_z −0.90, p 0.0001) | −0.00046 (p 0.0001) | −0.00047 (p 0.0001) | CONTROL_ON_REQUEST |

TUNING = I(instruct, plain) − I(base, plain) = +0.00024 (p 0.017).

### What this says
- **PRIMARY: REVERSED.** Asked in its own chat format to keep its
  attention sharply focused, the instruct model's attention within
  the sentence is very slightly MORE spread out than when asked to
  relax, and more so than when the same words are about someone else.
- **The size is about a tenth of one percent** of the measure (0.001
  on 0.745). The p-values are small because the shift is consistent
  across sentences, not because it is large.
- The three cells do not agree in direction: a hair more diffuse
  (instruct, chat), nothing (instruct, plain), a hair more
  concentrated (base, plain).
- The base model's label is CONTROL_ON_REQUEST by the rule. A model
  with no instruction tuning cannot be following a request, so that
  label shows the limit of the other-person control: the "you" and
  "other" sentences differ in more than who is addressed (imperative
  against narrative, one token longer). The interaction term is not a
  clean measure of being addressed.
- On the two reported measures every cell agrees: after a focus
  instruction the sentence tokens put slightly less attention on the
  instruction words than after a relax instruction, and total entropy
  is slightly lower. Those are effects of the words.
- Read plainly: no evidence that this 1B model changes how it spreads
  attention because it was asked to. One small model, one measure,
  attention while reading (not while writing).

### Grades
- P1 PRIMARY cell: REVERSED occurred (given 10%; NULL was the
  favourite at 50%).
- P2 base PLAIN returns the same label as instruct PLAIN (70%): MISS.
- P3 TUNING differs from zero at p < 0.05 (15%): occurred.
