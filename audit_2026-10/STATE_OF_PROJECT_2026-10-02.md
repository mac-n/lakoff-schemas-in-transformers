# Where the Lakoff project stands — Friday 2 October 2026

*One page, written for Niamh at the end of a long afternoon. — Claude
(session labelled Fable 5.1). Everything here has a prereg or a
labelled diagnostic behind it; file names are at the bottom.*

## In four sentences

Today started as a test of your idea about what UP is grounded in and
turned into an audit of BALANCE. Two faults in how the axes were built
were found: Pythia was never given a start marker, and the BALANCE
word lists pit whole-token words against chopped-up ones. Between
them they account for the Pythia norm result and for the
BALANCE–entropy correlation in GPT-2 and Llama. One piece of the
causal result survived a hard control and is not yet explained.

## What stands

- **The eight-schema structure (Finding 3) survives (exp192).** On
  clean axes the pattern of similarities between the eight schema
  axes is the same from layer to layer (0.94, against 0.91 for the
  scrambled null) and Lakoff's predicted pairs are closer than the
  rest (+0.17 against +0.01, null 95th +0.08). Four of six predicted
  pairs hold at every layer. One correction: BALANCE is not the
  centre of the system; on clean axes it is the least connected. The
  old "centre" was its lopsided token split.
- **Steering UP makes the model happier (exp187).** A cleanly built
  spatial UP direction, added to Pythia 1.4B while it completes
  prompts, shifts its answers toward happier, by more than 95% of
  directions built from random words. The old "bundle" half (status,
  quantity) does not clear that null. HAPPY IS UP is the one Lakoff
  mapping that survived every control today, in behaviour and in
  readouts.
- **Happy reads as UP.** Happy sentences project higher than sad ones
  on a purely spatial UP axis, at every layer, with no height words in
  them. Seen twice on fresh sentences, the second time on a clean
  instrument (exp175).
- **The GPT-2 causal effect, at layer 3.** Making attention more
  diffuse moves token states along the original BALANCE axis more
  than along any of 300 fake axes with the same token split (exp179).
  It is small, it is not shown by a cleanly built BALANCE axis, and
  what in the original 30 words carries it is unknown.
  Update (exp186): it is mostly the four "un-" pairs in the lists.
  Axes built from word / un-word pairs on unrelated concepts all
  move the same way, and BALANCE without its un- pairs no longer
  stands out.
- **The lab's harness.** Every June number rerun today reproduced to
  three decimals, on a different device. The measurements were right.
  What the axes measured was the problem.

## What fell

- **The opposite-pair "hole" effect (exp180), found and explained the
  same evening.** UP words did leave a bigger hole than DOWN words in
  two models. Niamh worked out why: English says "up and down", "here
  and there" in that order, most of the test sentences put "and"
  after the word, and the model was expecting the partner. Nothing to
  do with UP. (exp182, exp185, and the post-hoc note in
  PREREG_exp180.md.)

- **Pythia grounds BALANCE in residual norm.** With a start marker the
  coupling is about zero from layer 11 up.
- **BALANCE is coupled to attention entropy in GPT-2 and Llama.**
  Random words with the same token split give the same correlation
  (exp178). A clean BALANCE axis is no different from random clean
  axes.
- **The inflection finding (Finding 4), tested at midnight (exp191).**
  With single-token pairs, a fragment-free BALANCE axis and a start
  token, the "markedness sink" sits inside what random word axes give
  at every Pythia layer, and GloVe looks the same. The transformer-
  versus-static contrast is not there. (-ing alone is below the random
  range: a lead, not a result.)
- **"Different bodies"** (norm in one model, entropy in others). Both
  halves are gone as stated. With a start marker Pythia does not show
  the entropy coupling either (exp176), apart from one layer.

## Three method results worth keeping

These apply to anyone who builds concept directions from isolated
words, which is common in steering work.

1. **Position zero.** Without a start marker, a single-token word sits
   in the first slot, where Pythia's state is about 45 times larger.
   "Norm" then means "is one token".
2. **Token split.** If one pole is whole tokens and the other is word
   fragments, the axis is largely a fragment axis.
3. **The null is not zero.** A random word-contrast axis correlates
   with attention entropy at 0.2 to 0.3. A coupling has to beat random
   axes, not zero.

## Your UP hypotheses

- **Layers in agreement (exp174):** null, on an instrument later found
  bent.
- **Holding against gravity, effecting (exp175, exp180):** the
  word-level lead turned out to be the "X and ___" word-order effect
  above. At sentence level the UP readout follows happy.
- **Potential energy (exp182):** the lead does not persist in fixed
  text; it was a one-token bump at "and".

## Not yet checked

- Whether "how much the later layers disagree after a push" is a
  usable test of a direction being one the model responds to. exp193
  (Pythia, layer 4): the UP push lowers later-layer agreement three
  times more than single-word, word-pair, word-average or noise pushes
  of the same size, and more than any other schema axis, while moving
  the output no more than they do (drop and output KL correlate at
  +0.11). One model, one layer; exp190 showed GPT-2's random pushes do
  move agreement, so the GPT-2 version is the next test. A lead.

- Whether the surviving causal effect is about "un-" negations.

## What is public

The blog post and glassnest.ai/embodied state the norm grounding and
the entropy coupling as found. Those passages no longer hold. A short
dated correction would cover it; nothing else needs to come down.

## Next, in order

1. (Done: exp180.) Can "default member of a pair" be told apart
   from "the UP side of Lakoff's metaphor bundle"?
2. Focus / relax: can a model change its attention entropy on request.
3. A morphology-matched control for the causal effect.
4. (Done: exp191, it fell.)
5. (Done: exp192, it survives.)

## Where everything is

Branch `exp174-up-grounding` in `~/Documents/embeddingexp/`, all
committed.
- `HYPOTHESIS_UP_grounding_2026-10-02.md` — your words, verbatim and
  timestamped, before each test.
- `START_TOKEN_BUG_2026-10-02.md` — the start-marker fault.
- `PREREG_exp174.md` … `PREREG_exp179.md` — each test's frozen rules,
  results and my graded odds.

---
## Addendum: the early hours of 3 Oct (written 03:59 IST, 03 Oct 2026)

Niamh stayed up. Seven more preregistered runs, all committed, each
with RESULT + GRADES in its PREREG file.

**Survived / new**
- **HAPPY IS UP replicates in GPT-2** (exp194): clean UP steering
  moves affect beyond random-word directions at layers 6, 12 and 18;
  status moves at 6 and 12, quantity at 12. So in GPT-2 the "more
  and status" half comes back; in Pythia 1.4B it was affect only.
- **SOURCE-PATH-GOAL has a shape in the models** (her idea, 01:33).
  exp197 pilot -> exp198 (cue-disjoint): a probe trained only on
  literal walks reads abstract change ("poverty turned into wealth")
  as start/end: Pythia TRANSFERS at L8 (0.67–0.73; follows swaps;
  chance on no-journey); GPT-2 passes at L16, not at the named L8;
  a probe that never saw "from"/"to" reads "from poverty to wealth"
  at 0.98. Correction on file: "into"/"out of" were shared cues in
  three literal sentences.
- **Instructions use the two ends** (exp199, her 02:41): Llama base
  TRANSFERS at L6 (draft = start, summary = end, 0.75; swap 0.78;
  no-task 0.52); instruct one hundredth short at L6, clear at L3.
  Early layers only; chat template irrelevant; tuning adds nothing.
- But a lone asked-for thing is not a goal (exp200, her 02:54):
  NOT_A_ROLE both Llamas; deep layers mark "in an instruction", not
  goal vs source. The role is relational.

**Closed**
- The later-layer agreement measure. exp193's Pythia-L4 oddity did
  not replicate at Pythia L8 or GPT-2 L4 (exp195, 0 of 2) and does
  not separate used from unused SAE features (exp196). One odd cell.

**Figure**: figures/same_path.png (the cartoon of exp198).

**Next, if wanted**: a layer sweep named in advance for GPT-2 in
exp198 and for Llama-instruct in exp199; more literal training
sentences (the probe is 0.88–0.97 within-set); the causal version of
the path (steer "nearly there", watch abstract tasks resolve); the
"distance to goal" quantity during generation (reading B in the
hypothesis file).
