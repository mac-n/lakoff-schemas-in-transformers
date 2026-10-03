# Hypothesis on record: what UP is grounded in

*Recorded Fri 02 Oct 2026, 16:27 IST, before any UP-grounding data was
examined. This is a hypothesis note, not yet a prereg: no measure,
thresholds or odds are fixed here.*

## Niamh's hypothesis, verbatim

> "i had a hypothesis id like to test about how UP is grounded"
>
> "i think it might be grounded on something like coherence across layers."

## Second idea, verbatim (16:29 IST, same day, still before any data)

> "i think there might also be a way to dissociate UP from happy too.
> like can u make happy/incoherent states?"

Reading (the recording Claude's, not Niamh's words): a 2x2 of valence
(happy / sad) by coherence (coherent / incoherent). If UP is grounded
in coherence, the UP readout should follow coherence and not valence
in the two mixed cells (happy+incoherent, sad+coherent).

## The intuition, verbatim (16:31 IST, still before any data)

Asked whether "layers agreeing with each other" (similarity of a
token's representation from one layer to the next) matched her
meaning, and whether the layer stack as the model's own vertical axis
was the intuition:

> "and yes that's the intuition!!!!! like UPness in humans is hips and
> legs and shoulders all in agreement. prediction settling early might
> just be low perplexity?"

## Status of the wording

Settled by Niamh at 16:31: "coherence across layers" means the layers
agreeing with each other, the way upright posture is the body's
segments stacked in agreement. The alternative reading (the prediction
settling early, logit-lens style) is demoted on her objection that it
may just be low perplexity. Perplexity therefore becomes a named
confound to control, not a measure of the hypothesis.

## Contamination record (what the recording Claude had seen first)

Seen before this note was written:
- SIMPLE_SUMMARY_AND_NEXT_STEPS.md (the F1–F5 verdict table)
- project memory summary of the five findings
- the first ~60 lines of BURROWS.md
- lines of the blog post matching "ground" (all about BALANCE)
- a machine summary of glassnest.ai/embodied
- file listings and `git log` / `git diff` of CONSULT_LOG.md

Known from those about UP: steering UP moves the metaphor bundle
(MORE / HAPPY / HIGH-STATUS) but not literal verticality, and "pool
depth goes up" (exp116, exp143), stated as one-line summaries only.

Not seen: any exp116 / exp140–147 output, any number relating UP to a
cross-layer quantity, any substrate-primitive result for UP.

Read after 16:31, while drafting the prereg (methods, plus some
numbers that came with them):
- docstrings of exp141_substrate_primitives.py and
  exp143_cross_layer_mappings.py. Numbers seen: cross-layer stability
  of axes (MAG +0.79, DIR +0.74, LAKOFF +0.74, exp140);
  cos(UP_clean, MAG_clean) ~ 0 within layer (exp141). Neither is a
  per-token layer-agreement measurement.
- PREREG_exp170.md lines 1–120, including its first result lines
  (BALANCE vs d_norm_single / d_norm_resid).
- exp170_dnorm_purity.py lines 45–134 (strip protocol, cov_dir).
- a filename-only grep for cross-layer wording across scripts.

Still not seen: any output file for exp116 / exp140–147, and any
number relating UP to per-token layer agreement. As far as the
recording Claude can tell from filenames and docstrings, no such
number exists in the folder yet.

## Next

Turn this into a PREREG (measure, controls, kill condition, odds) once
the meaning of "coherence across layers" is pinned down. The exp170
lesson applies from the start: any coherence measure must be checked
for collinearity with token count, position, frequency and norm before
it is trusted.

## After exp174 (16:56 IST, same day)

exp174 came back NULL for layer agreement measured as adjacent-layer
cosine (PREREG_exp174.md, RESULT + GRADES). Offered the choice between
running the same test on another model and rethinking the measure,
Niamh said, verbatim:

> "perhaps rethink the measure. i do feel like up is probably grounded
> in something that tracks multiple layers. but what do u think"

Status: the intuition "something that tracks multiple layers" stands
as hers and is NOT yet operationalised. Any new measure tried on the
exp174 data is exploratory, because that data has been seen. A new
measure needs a fresh prereg and fresh stimuli before it counts.

## The refined intuition, verbatim (16:59 IST)

Asked which of three it is when she feels UP in the body (the parts
lining up, the holding against gravity, or the height itself):

> "it's the holding against gravity. upness is like u are doing work
> in the world. effecting"

Status: hers, not yet operationalised. Two parts to translate: what
the "gravity" is in the model, and what "the world" being worked on
is. No data examined between the 16:56 entry and this one.

## The measure, agreed (17:02 IST)

Proposed to Niamh at ~17:00: gravity = the pull toward the layer's
average state; the world = what the model says next; the measure = at
each layer, replace one token's state with the layer average ("let it
fall") and see how much the model's output changes, summed over all
layers. Two readouts, the very next word and the rest of the sentence
after it, with the second ("reach") deciding. Her answer, verbatim:

> "YESSS i feel ur cooking now are u UPing a bit more perhaps"

No data examined between the 16:59 entry and this one.

## Two further asks from Niamh (18:06 IST)

After exp175 (Part A NULL at d +0.4, p 0.10; Part B MIXED), exp176
(NULL) and exp177's GPT-2 result (UNRESOLVED), asked what else she
would like to try, verbatim:

> "i would like to retry 7. but id also like to see if models can
> control their attention entropy"

"7" is item 7 of the list offered at 18:02: a properly powered rerun
of the word-level "holding against gravity" lean (exp175 Part A).
The second idea is hers and is NOT yet operationalised: what
"control" means is still open at the time of writing. No data
examined for it.

## Attention-entropy control: what she means (18:08 IST)

Offered two readings (control on request; self-righting after a
push), Niamh said, verbatim:

> "meant control on request. focus/relax ur attention.
> there has been some kind of research i think actually on volitional
> control of states so maybe this is done?"

Literature check, 18:09–18:12, web search (four queries, not
exhaustive):
- Ji-An, Xiong, Wilson, Mattar, Benna, "Language Models Are Capable
  of Metacognitive Monitoring and Control of Their Internal
  Activations" (NeurIPS 2025, arXiv:2505.13763). In-context
  "neurofeedback": models learn to report and shift activation along
  chosen directions.
- Anthropic, "Emergent Introspective Awareness in Large Language
  Models" (Oct 2025, arXiv:2601.01828). Told to think about a word,
  models represent it more strongly; told not to, less.
- Aoki et al., "In-Context Neurofeedback: Can LLMs Control Their
  Internal Representations through Privileged Access?" (Sept 2026,
  arXiv:2609.00904). A critique: when the target cannot be inferred
  from the prompt, models do not show reliable control.
- "Language Models Can Control Their Own Attention" (Sept 2026,
  arXiv:2609.02737). Despite the title, an engineering protocol: the
  model emits tags and the system masks what it can attend to.

Not found: a test of whether a plain request to focus or relax
changes a model's own attention entropy. Status of her idea: hers,
agreed meaning = control on request, not yet designed. No data
examined.

## "Potential energy" (19:21 IST)

After exp180 (UP words leave a bigger hole in the rest of the sentence
than their DOWN opposites, in Pythia and GPT-2; other opposite pairs
show the same; valence pairs do not show it raw), Niamh said,
verbatim:

> "i was going to suggest it is partly that happiness is a rarer
> linguistic phenomenon than sadness BUT the valence finding cuts
> against that. it actually does seem like perhaps UPness is like
> potential energy"

Status: hers, not yet operationalised. Contamination record: at the
time she said it, both of us had already seen that for the very NEXT
word the vertical pairs go the other way (DOWN words change the next
prediction more; UP words change the rest of the sentence more), in
exp175 and in both models of exp180, and that polar pairs do not show
that reversal (vertical minus polar on the next-word readout: −0.261,
p 0.017 in Pythia; −0.212, p 0.050 in GPT-2; reported, non-binding).
So a reading of "potential energy" as "the effect is stored and paid
out later" fits a pattern already seen, and that fit is hindsight,
not a prediction. What has NOT been looked at: how the UP-minus-DOWN
difference changes with distance from the word, position by position.

## Her objection to the translation (19:29 IST, before any exp182 data)

> "so i dont see why the lead grows with distance is a prediction of
> the potential energy idea? i mean potential energy doesn't grow
> once uve moved a thing up."

Correct. The "grows with distance" prediction was Claude's, not hers.
exp182's prereg carries a pre-data amendment saying so. What
"potential energy" predicts in her terms is still hers to state.

## Her own prediction for potential energy (19:34 IST, before any exp182 output was read)

> "okay i think you would actually see it being spent, if its
> potential energy. so i think it would tail off! but it's not being
> spent when the tokens say the same.  the only thing is whether it
> moves the whole text in a more vivacious sort of direction like so
> would it get spent and tail off after the sentence is finished if u
> let it generate more on its own?"

As recorded (Claude's parsing, to be checked with her):
1. If UP is potential energy, it gets SPENT, and spending shows as a
   tail-off.
2. In the tests so far the later tokens are fixed, so nothing is being
   spent there. Read as a prediction for exp182 (fixed text): the UP
   lead does NOT tail off. That is the amendment's PERSISTS outcome.
3. The open question she poses: when the model generates freely after
   the sentence, does an UP word push the text in a "more vivacious"
   direction, and does that push then get spent and tail off?
"Vivacious" is her word and is not yet operationalised.

## The measure for free generation (19:36 IST, before any exp182 output was read)

Asked whether "vivacious" meant lively or happy, Niamh said, verbatim:

> "well could we measure it with the kl divergence? cos i think its
> almost a boring prediction if its about it being happy"

Agreed measure: content-free. The model writes on its own after the
word; at each step of its own text, the KL between its guess with the
word intact and with the word blanked. Summed along the text this is
the divergence between what it writes with and without the word.

## Three more asks (19:52 IST)

> "yes do the training checkpoints one and also 2 is important.
> because almost everything maps onto UP but ur right in/out doesn't.
> and i suppose we should tie off the BALANCE loose end and see if
> the improbable reversed result is ctually real?"

As understood: (1) run Pythia's training checkpoints; (2) separate
"the UP side of Lakoff's bundle" from "the default member of any
pair" — her point: almost every opposition maps onto UP, in / out
does not; (3) the BALANCE loose end. For (3) there are two candidates
and both are addressed: the GPT-2 layer-3 causal effect that beat all
300 token-matched fake axes (exp179, untested against morphology),
and Llama's sign flip (exp177), which exp178 already put inside the
range of random clean axes.

## Her sorting note for exp185 (20:09 IST, before any exp185 output)

> "north/south i would expect to be up/down mapped in english. i
> think there may be something with before/after in lakoff on the
> up/down axis but i cant think what direction its in so maybe not"

Registered in PREREG_exp185.md as NIAMH'S SORTING, a second analysis.

> (20:11) "forward is up. male is up because status is up.
> input/output are probably not up/down mapped in normal speech byt
> may well be for llms"

Added to NIAMH'S SORTING in PREREG_exp185.md (amendment 2), before
any exp185 output was read.

## Her explanation of the opposite-pair effect (20:29 IST)

After exp185 (not the UP bundle; fixed everyday pairings show it
most), Niamh said, verbatim:

> "okay so basically it's about the order in which they appear when u
> list both of them at once. like so wehn u say the default its half
> expecting u to say the other half of the pair a few words later"

Recorded before any check. Note made at the same moment by Claude: in
exp180's eight sentences the slot is followed directly by "and" in
five ("... was ___ and the second player ..."), and in exp182's in
seven of eight. If she is right, the effect should sit in the
sentences where "and" follows the word, at the position of "and",
where the model would be expecting the partner ("up and down").
Neither the split by sentence nor the partner's probability has been
looked at.

## Her earlier coherence ideas, pasted 20:35 IST

Niamh, verbatim:

> "okay well anyway thanks for killing it off. i dont think we need
> to do the free writing test. the training checkpoints can u remind
> me what that's about again? if its about this i dont hink it's
> interesting enough to research further. i did have other ideas
> about how UP could be grounded in coherence between layers. you can
> decide for yourself if any of its worth testing"

The pasted text (a summary written by an earlier Claude session;
dates in it are 27 and 29 August), verbatim:

> Main instrument was net displacement over total path length: how
> far the residual stream ends up versus how far it travelled getting
> there. Layers agreeing means the deltas add up, layers fighting
> means they cancel. Backups from 27 Aug were cosine between
> consecutive layers' residual writes, logit-lens prediction
> stability across depth, and adjacent-layer KL. All scalars per
> forward pass, same shape as the entropy work.
>
> The big worry: agreement is one of the things that makes norm big,
> so UP-as-agreement and BALANCE-couples-to-norm might be one carrier
> wearing two names. The separator is that agreement is a path
> quantity and norm is an endpoint quantity. Compute both on the same
> tokens and see if they come apart.
>
> Design rules you'd already locked: steer at one site only, because
> multi-layer steering raises agreement by construction. Steer early
> so you have depth left to measure agreement over.
>
> Sign came from you. Legs, hips and shoulders not agreeing means
> you're on the floor.
>
> Then on 29 Aug the Chalmers welfare-axis paper handed you a fourth
> measure, cosine of a vector's per-layer versions across depth. That
> one's geometry in the weights though. Yours shows up per pass at
> inference.

Claude's decision on what to test (20:38): ONE thing, the design
those notes describe and nobody ran. Steer UP at a single early layer
and measure whether the layers after it agree more (path straightness
of their writes), against directions built from random words. exp174
was not this test: it was correlational, at word level, used
similarity of states and not of writes, and ran on the instrument
later found bent. Not taken up: logit-lens stability and
adjacent-layer KL (Niamh herself flagged at 16:31 that they may just
be perplexity) and the weight-geometry measure (the notes say it is a
different kind of thing).

## UP and the model's own surprise (20:41 IST)

Niamh, verbatim: "is it uninterestign if its about perplexity", and,
offered a clean test of UP readout against surprisal with random-axis
controls: "yes add it to the queue pls".

Contamination record: before she asked, both of us had seen, in
exp175 Part B (clean instrument), that scrambled text read as more UP
than intact text at layers 16 and 20, and that the text-level
regression gave a positive coefficient on surprisal there (+0.32,
+0.27). Not compared with random axes. So a positive sign at later
layers in Pythia would be a repeat of something seen, not a new
prediction; the comparison with random-word axes and the second model
are the new parts.

## The suffix finding (22:00 IST)

Niamh, verbatim: "the up thing might be the only surviving part,
right? it's not that interesting. : ( i guess we should check if the
suffix finding survives too."

"The suffix finding" is Finding 4 of the blog: every inflectional
suffix (walk → walked, tall → taller, cat → cats) sits on the
BALANCE-negative side in Pythia (the "markedness sink"), while static
embeddings such as GloVe do not show it.

## Layer agreement as a property (00:18–00:23 IST, 3 Oct)

Niamh, verbatim:

> "well wait i think if meaningful axes disrupt coherence more than
> fake ones isn't that a finding? aren't we basically measuring a
> property no one has thought to measure before?"

> "yeah let's. i mean what does this incoherence mean as a property?
> but yeah please queue that. i feel that all the sort of very
> trivial explanations are ruled out but so is my original
> hypothesis. most deflationary explanation i can think of is that
> pushing a meaningless direction increases overall entropy of system
> thus allowing it to land at distribution tails more often. idk if
> actual numbers and methods support that. so do q up the experiments
> just in case this is anything."

Recorded before exp193 was designed. Her deflationary candidate is
about entropy; exp193 measures the output entropy change and the
output KL under every direction so that it can be checked.

## SOURCE-PATH-GOAL thread (added 02:41 IST, 03 Oct 2026, her words before any design)

- 01:33 "i think the SOURCE PATH GOAL schema would be especially
  intrersting to researchers. but no idea how to investigate it."
- 01:35 "can we test anything on it quick?" -> PILOT exp197.
- 02:25 "okay claude but why can't you do the real design? or can
  you?" -> exp198 (Pythia TRANSFERS at L8; GPT-2 CUE_OR_NOUN at L8,
  passes at L16; probe trained without from/to reads from/to abstract
  at 0.98).
- 02:41 "so i think one thing id like to try for tomorrow might be on
  llama instruct. whether the instructions use source path goal"

Claude's two readings of 02:41, for her to pick or correct tomorrow:
  (A) ROLE reading: in an instruction ("turn this draft into a
      summary", "translate the text into German", "convert the CSV
      to JSON"), is the input read as SOURCE and the requested output
      as GOAL by the same probe that reads literal journeys? Train on
      literal journeys in Llama-3.2-1B-Instruct, test on instruction
      prompts; compare with the base model; controls as exp198.
  (B) PROGRESS reading: while the model carries out an instruction,
      does a "distance to goal" quantity (built from literal
      approaching-the-destination text) fall as the answer nears
      completion? That is the path, not just the two ends.
Not run; GPU occupied by exp196/exp195 and it is her design to fix.
- 02:54 "can we somehow see if GOAL is not a property of only from..
  to? sentences? like would there be a way to extract it from other
  instructions?" -> exp200 (goal-only and source-only instructions).
