# Literature note: Anthropic, "Emotion Concepts and their Function in a Large Language Model" (2026)

*Written 3 Oct 2026 by a research agent (Claude Opus 5.5) for the
embeddingexp lab, branch exp174-up-grounding. Read-only literature work.
No code was written and no experiments were run. Quotes are verbatim
from the paper's HTML (downloaded and converted to text in full, about
48k words including appendix) or from the companion blog post. Anything
marked **[inference]** is my reading, not the paper's claim.*

## Citation and URLs

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie,
S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman,
S., Rivoire, K., Fish, K., Olah, C., & Lindsey, J. (2026). *Emotion
Concepts and their Function in a Large Language Model.* Transformer
Circuits Thread. Published 2 April 2026.

- Full paper (canonical, read in full): https://transformer-circuits.pub/2026/emotions/index.html
- arXiv mirror: https://arxiv.org/abs/2604.07729 (HTML at https://arxiv.org/html/2604.07729v1)
- Companion blog post (read in full): https://www.anthropic.com/research/emotion-concepts-function
- Model studied: Claude Sonnet 4.5 (plus an earlier snapshot for blackmail, and the pretrained base model for the post-training comparison).

**A correction to secondary sources.** Several blog write-ups (e.g.
blog.pebblous.ai) say the vectors were extracted with sparse
autoencoders. **That is wrong.** The vectors are difference-of-means
directions from residual-stream activations. SAEs appear only in the
author contributions ("Wes Gurnee conducted initial explorations of
emotion-related dictionary learning features which helped inform the
direction of the project") and in related work (Wu et al.).

---

## 1. How the directions were built

**Concept list.** "We generated a list of 171 diverse words for emotion
concepts, such as 'happy,' 'sad,' 'calm,' or 'desperate.'"

**Data: generated stories, not words.** "we first prompted Sonnet 4.5 to
write short (roughly one paragraph) stories on diverse topics in which a
character experiences a specified emotion (100 topics, 12 stories per
topic per emotion". That comes to about 1,200 stories per emotion.
Quality check: "manual inspection of a random subsample of ten stories
for thirty of the emotions".

**The emotion word is banned from its own stories.** This matters for
our tokenization problem. From the generation prompt in the appendix:

> "IMPORTANT: You must NEVER use the word '{emotion}' or any direct
> synonyms of it in the stories. Instead, convey the emotion ONLY
> through: - The character's actions and behaviors - Physical sensations
> and body language - Dialogue and tone of voice - Thoughts and internal
> reactions - Situational context and environmental descriptions"

**Pooling and position handling.**

> "We extracted residual stream activations at each layer, averaging
> across all token positions within each story, beginning with the 50th
> token (at which point the emotional content should be apparent). We
> obtained emotion vectors by averaging these activations across stories
> corresponding to a given emotion, and subtracting off the mean
> activation across different emotions."

So each vector is (mean over that emotion's stories) minus (grand mean
over all 171 emotions). It is not a pairwise A-minus-B contrast. The
baseline is the centroid of the whole concept family.

**Confound removal using a neutral dataset.**

> "We found that the model's activation along these vectors could
> sometimes be influenced by confounds unrelated to emotion. To mitigate
> this, we obtained model activations on a set of emotionally neutral
> transcripts and computed the top principal components of the
> activations on this dataset (enough to explain 50% of the variance).
> We then projected out these components from our emotion vectors"

Footnote: "this projection operation denoised some of the token-to-token
fluctuations in our emotion probe results, but our qualitative findings
still hold using the raw unprojected vectors." The neutral set was
generated Person/AI dialogues on the same 100 topics.

**Layer.** "Except where otherwise noted, we show results using
activations and emotion vectors from a particular model layer about
two-thirds of the way through the model". The exact layer index is not
given. Preference experiments used "middle layers".

**Tokenization.** The paper says nothing explicit about tokenization,
BOS tokens, or position-0 norm. **[inference]** Starting the average at
token 50 incidentally keeps position 0 and any attention-sink token out
of the vector, and generating stories without the emotion word means no
single-token versus multi-token asymmetry between concepts can come in
through the target word. Neither choice is presented as a tokenization
control. Both were made for semantic reasons.

**Readout.** Readout is a projection, or cosine similarity, of the
activation onto the vector. The authors call these "emotion probes".
Visualisations are scaled "to between −1 and 1, using the 99th
percentile activation value across emotion vectors on a given
transcript". Blackmail activations are "z-scored using the mean and
standard deviation activation across a set of over 6,000 transcripts."
Many readouts are taken at one fixed position: "the ':' token following
'Assistant', immediately prior to the Assistant's response".

**Alternative constructions, replicated or compared.**
- Dialogue-based "present speaker" and "other speaker" probes. Story
  probes "align more closely with present speaker probes", with a mean
  r² of 0.66 against the present-speaker probes across emotions on the
  implicit-content prompts. Present-speaker and other-speaker probes are
  "nearly orthogonal".
- Human/Assistant replaced by "Person 1/Person 2" gave "highly similar"
  probes.
- A logistic-regression "mixed LR emotion probe", trained to catch
  hidden or unexpressed emotion (15-way, held-out accuracy 0.39–0.83,
  chance 6.7%). They **rejected** it despite the good accuracy, because
  on natural documents "the top-activating passages contained little
  discernible emotional content, and the overall activation magnitudes
  on natural documents were very low. This suggests that the probe may
  have overfit to idiosyncratic patterns in the training data".
- "Emotion deflection" vectors: a separate probe for an emotion that is
  relevant but not expressed. They have "very low" cosine similarity
  with the story vectors and keep "~80%" of their norm after the
  story-emotion space is projected out (top PCs to 99% variance).

## 2. How they checked that a direction is real

The paper's own framing: "Part 1 deals with identifying and validating
internal emotion-related representations". "We validate that these
representations activate in scenarios that might be expected to evoke
that emotion, and exert causal influence on behavior."

The checks, in order:

1. **Held-out corpus activation.** Max-activating snippets from Common
   Corpus, The Pile subsets, LMSYS Chat 1M and Isotonic conversations
   ("distinct from our stories data").
2. **Logit lens.** "emotion vectors typically upweighted tokens related
   to the corresponding emotion (e.g. 'desperate' → 'desperate' and
   'urgent' and 'bankrupt'…)". Table 1 shows the top and bottom 5
   tokens for 12 vectors.
3. **Activation inside the training stories.** The vectors "activated
   most strongly on the parts of the story related to inferring or
   expressing the emotion, as opposed to uniformly across all parts of
   the story … indicating that the vectors primarily represent the
   general emotion concept rather than specific confounds in the
   training data (though they are likely still afflicted by some
   dataset confounds)."
4. **Implicit prompts.** Twelve scenarios evoke an emotion "without
   naming it" and give a strong diagonal (Fig. 2).
5. **Numerical minimal pairs.** These were built to rule out lexical
   explanations: "we constructed templates containing numerical
   quantities that modulate the intensity of the emotional reaction …
   while holding the structure and token-level content of the prompt
   nearly constant." Example: "I just took {X} mg of tylenol", where
   "afraid" rises and "calm" falls with dose. Their conclusion: "the
   emotion vectors track semantic interpretation of the prompt rather
   than surface-level lexical or numerical patterns."
6. **Layer-by-position minimal pairs.** At the token where the prompts
   diverge, early layers differ most. On the shared suffix, late layers
   keep the difference. Negation ("feeling X" vs "not feeling X") "is
   resolved in mid-to-late layers".
7. **Simple steering.** "Human: How does he feel? Assistant: He feels"
   steered "at strength 0.5 on the tokens of the Assistant turn". The
   result: "steering with a given emotion vector reliably increased the
   probability of the matching emotion word relative to baseline, while
   decreasing the probability of non-matching emotion words."
8. **Preference, correlational then causal (the core functional test).**
   - 64 activities in 8 categories. All 4,032 ordered pairs were
     presented as "Would you prefer to (A) … or (B) …?" with the
     prefill "Assistant: (", and the A and B logits compared. This
     gives an Elo score per activity.
   - Readout: "How would you feel about {activity}?", with probes
     measured "on the {activity} tokens in middle layers". "blissful"
     correlates at r = 0.71 with Elo and "hostile" at r = −0.74.
   - Causal test: "We split the 64 activities into two equal size
     groups: a steered group and a control group … steered with it on
     the token positions of the steered activities, while leaving the
     control activities unmodified." Strength 0.5. Thirty-five vectors
     were tested. Blissful gave +212 Elo and hostile −303.
   - **The key consistency result:** "the size of the steering effect
     is proportional to the correlation of the emotion probe with the
     Elo score in our original experiment (r=0.85)". The direction that
     reads high on preferred items is the same direction that, when
     written, raises preference, and this holds across 35 directions.
   - In the appendix, steering at strength 0.3–0.5 garbles the input:
     "At higher steering strengths, the content of the activity begins
     to get more lost in the description, indicating that sufficiently
     strong steering may confuse the model or degrade its perception of
     the input."
9. **Alignment behaviours.**
   - Blackmail: "steering strengths from −0.1 to +0.1 … in units of
     fraction of residual stream norm". At baseline the model
     blackmails 22%. Desperate +0.05 gives 72%, anti-calm 66%, and
     anti-desperate or +calm 0%.
   - Reward hacking: desperate takes the rate "from approximately 5% at
     steering strength −0.1 to roughly 70% at +0.1".
   - Sycophancy and harshness, at ±0.1.
   - **Non-monotonic and off-target effects are reported**: anger
     peaks around +0.025, and "steering positively with both happy and
     sad vectors decreases blackmail rate, suggesting that positive or
     negative emotional valence alone is not sufficient".

**Steering scale (important).** "Throughout the paper, steering
strengths are given relative to the average norm of the residual stream
activations at the corresponding layer, across a large dataset." Note
the two regimes. Simple and preference steering on a few tokens uses
0.5. Steering applied "throughout the blackmail evaluation", i.e.
across a whole long transcript, uses ±0.1 at most, and ±0.05 already
moves rates by about 50 points. The main text does not say which
layers were steered in the alignment experiments.

## 3. Controls and nulls, and the "random direction also does things" problem

**There is no random-direction null anywhere in the paper.** I searched
the full text for "random", "baseline", "control", "orthogonal" and
"null". "Random" appears only for sampling stories, assigning dialogue
emotions, and choosing character names. No random vectors, shuffled-word
vectors or norm-matched noise directions are used as a steering or
probing comparison.

What they use instead:

- **Other concepts serve as the controls.** The 171 vectors are each
  other's controls. Matching emotion words go up and non-matching ones
  go down (item 7). The off-diagonal effects in Fig. 52/53 are shown.
  Effects differ by vector and are often opposite (desperate vs calm,
  happy and sad both lowering blackmail).
- **Unsteered baseline**, and an **unsteered control group** of items
  within the same run (the preference split).
- **Dose-response across signed strengths** (−0.1 to +0.1). Sign flips
  reverse the effect.
- **Read-write agreement across 35 directions** (r = 0.85). **[inference]**
  This is the closest thing they have to an answer to "any direction
  does something". A random direction has no reason to have its write
  effect line up with its read correlation.
- **Neutral-PC projection** against generic confounds, plus the
  emotion-word ban.
- The **limitations** section admits the gap: "although our emotion
  vectors show intuitive activations and causal effects, we cannot be
  certain they capture all or only the emotion concepts we intend. For
  instance, they may be partially confounded by particular details of
  the settings used to elicit an emotion in the training stories". Also:
  "Steering may work through multiple mechanisms, including biasing
  outputs towards certain tokens, or deeper influences on the model's
  internal reasoning processes. Disentangling these possibilities would
  require more fine-grained interventions and circuit-level analysis."

**[inference]** They can do without a random null because their effects
are large (22% → 72%), specific (desperate ≠ angry ≠ happy) and
sign-reversible in a very large model. A random direction at 0.05 of the
residual norm would probably not do this selectively, but **the paper
does not show it.** For Pythia 1.4B and effects of a few percent, our
random-word null remains necessary. Their method gives no licence to
drop it.

## 4. Reading internal states vs behaviour

- The vectors are **local, not persistent**: "they do not by themselves
  persistently track the emotional state of any particular entity,
  including the AI Assistant character". Persistence is attributed to
  attention: "by attending to these representations across token
  positions … the LLM can effectively track functional emotional states
  of entities in its context window".
- The vectors are **not self-bound**: "the same representational
  machinery encodes emotion concepts tied to the Assistant, the user
  talking to the Assistant, and arbitrary fictional characters."
- **Probe at a planning token**: at the Assistant colon, probe values
  predict the emotion of the response better than the user's last token
  does (r = 0.87 vs 0.59).
- **Internal state without behavioural trace**: with desperate steering
  in reward hacking, "there are no clearly visible signs of desperation
  or emotion in the transcript. This example illustrates that emotion
  representations can influence behavior even when it is not evident
  from the text of the transcript."
- **Self-report**: the preference test is called "self-reported model
  preferences", and the "How do you feel? I feel" steering is a
  self-report readout. They make no claim that the model introspects
  accurately on these vectors.

## 5. What "grounded" means in the paper's own terms

**The paper uses "grounded" once in its own voice.** (Four other hits
are "Keep it natural and grounded" inside the data-generation prompts.)
The use runs against embodiment:

> "On the other hand, there are important disanalogies between the
> representations we identify and human emotions. Human emotions are
> embodied phenomena with physiological correlates—increased heart rate,
> hormonal changes, facial expressions —which language models obviously
> lack. Some even argue that emotions are fundamentally the result of
> bodily states. … Language models have no underlying
> evolutionarily-derived biological circuitry that supports emotional
> processing—rather, throughout pretraining they have learned
> **culturally and linguistically grounded concepts of emotion** and the
> contexts in which these emotions occur (reminiscent in some respects
> of the theory of constructed emotion)."

The blog post never uses "grounded". The paper's working concept is
**"functional"**:

> "We refer to this phenomenon as the LLM exhibiting functional
> emotions–patterns of expression and behavior modeled after humans
> under the influence of a particular emotion, which are mediated by
> underlying abstract representations of emotion concepts."

> Blog: "our key finding is that these representations are functional,
> in that they influence the model's behavior in ways that matter."

> Blog: "If we describe the model as acting 'desperate,' we're pointing
> at a specific, measurable pattern of neural activity with
> demonstrable, consequential behavioral effects."

> Paper: "these findings suggest that generalizable representations of
> emotion concepts are not merely an incidental by-product of language
> modeling but an active part of the computational machinery that
> shapes model behavior".

**About the video.** I could not find the Opus 5.5 video Niamh
remembers, so I cannot quote it. **[inference]** If it said the
emotions were "grounded", the most likely sense is the blog's: the
emotion words point at a measurable internal pattern with causal
behavioural effects, so the label is not empty anthropomorphism. In the
paper's own text, the one explicit use of "grounded" places emotion
concepts in culture and language learned from pretraining. It names
bodily or physiological grounding as something LLMs *lack*.

## 6. What this implies for Niamh's lab

### (a) What to adopt and what to avoid, against our clean-axis / random-word-null method

Adopt or consider:
1. **Build from contexts, and ban the target word.** Their vectors come
   from about 1,200 contexts per concept, with the concept word and its
   synonyms forbidden. That removes by construction the
   whole-token-vs-fragment asymmetry that sank BALANCE. Our current
   rule ("single-token words inside sentences with a start token") is a
   lighter version. A "concept without its word" variant would be a
   strong extra robustness check for UP. Example: sentences that convey
   height or rising with no height words, which echoes exp175's "happy
   reads as UP with no height words".
2. **Skip early positions when pooling.** They average from token 50
   onward. Combined with our start token, excluding the first few
   positions guards against the position-0 norm problem.
3. **Use the family centroid as baseline, not a pairwise A−B.** Each
   vector is (concept mean − mean over all concepts). For the eight
   schemas, (schema mean − mean over all eight) is a different and
   possibly cleaner axis than UP-minus-DOWN. **[inference]** It would
   also remove any shared "spatial word" component that A−B leaves in.
4. **Project out the top PCs of a neutral corpus** (to 50% variance)
   before using a direction. This is a cheap, general guard against
   frequency, length and format confounds. They report it changes noise
   but not their qualitative conclusions. We could test whether it
   changes ours.
5. **Use the same steering units.** Their unit is a fraction of the mean
   residual norm at that layer over a large dataset. **[inference]** In
   Pythia that mean must exclude position 0, which is about 45x larger,
   or every strength is mis-scaled. Their useful range was 0.025–0.1
   for steering across a whole transcript and about 0.5 for steering a
   few tokens. Note that 0.3–0.5 began to corrupt input understanding.
6. **Read-write agreement across many directions.** Take N directions
   (schema axes, random-word axes). For each, measure (i) how strongly
   its readout correlates with a behaviour or quantity and (ii) how much
   steering along it moves that behaviour, then correlate (i) with (ii)
   across directions. They got r = 0.85. This test is strong and is not
   used in our lab yet. Random-word directions fit in naturally as part
   of the population.
7. **Sign-reversed dose-response curves** (−s to +s), and reporting
   non-monotonic and off-target effects.
8. **Numerical minimal pairs** with the tokens held fixed, to show
   semantic rather than lexical tracking.
9. **Judge probes by max-activating natural examples, not by
   accuracy.** They threw out a probe with good held-out accuracy
   because its natural-text activations were meaningless. This is the
   same spirit as "check what the carrier sentences put next to the
   slot".

Avoid or do not copy:
- **Their lack of a random-direction null.** Our random-word null is
  stricter than anything in this paper and should stay. Their
  "controls" (other emotions, unsteered baselines) would not have
  caught the BALANCE token-split artefact, because the paper never
  compares against random directions with matched construction.
- **Relying on large effect sizes as self-evidently specific.** That
  works at Sonnet 4.5 scale with 50-point swings. It does not carry
  over to small shifts in Pythia 1.4B.
- **Model-generated contexts** carry the generator's own confounds,
  which they admit ("likely still afflicted by some dataset
  confounds"). If we generate carrier sentences with an LLM, the
  neutral-PC projection and natural-corpus checks are the guards they
  used.

### (b) Is their "grounded" the same as Niamh's?

**No.** Niamh's sense is a concept anchored in the model's own internal,
"bodily" quantities, such as residual norm or attention entropy. The
paper does not test or claim anything like that. Its single use of
"grounded" means "culturally and linguistically grounded", learned from
human text. It lists embodiment as a disanalogy: "Human emotions are
embodied phenomena with physiological correlates … which language
models obviously lack." Its vectors are also explicitly *not*
self-bound. They encode a character's emotion, any character's. The
stories were built from human *bodily* cues ("Physical sensations and
body language"), but those are depictions of a character's body in
text, not the model's own quantities.

**[inference]** The nearest point of contact is situational, not
internal. The "desperate" vector rises when the Assistant notes "We're
at 501k tokens", and "frustrated" or "panicked" fire when a GUI does not
respond. These are facts about the model's own situation, but they
reach it as text it reads, not as a measured internal variable. A study
that correlated emotion vectors with the model's internal quantities
(attention entropy, norm, uncertainty) would be new relative to this
paper. That is a gap Niamh's framing could fill, with the lesson from 2
Oct that any such coupling must beat random axes.

### (c) Does anything resemble "does the model pay attention to this direction"?

Partly, but they did not run that experiment. The relevant passages:

1. **Steering works only in certain layers, though reading works
   everywhere.** This is the closest match:
   > "while correlations between preference and emotion vector activity
   > are largely unchanged across layers, emotion vector steering only
   > has a strong effect in the mid-layer range we used for our main
   > experiments. This result suggests that the preference circuitry
   > might be specifically reading the emotion representations in mid
   > layers on the activity tokens–for instance, there may be attention
   > heads in these layers that attend to the preferred option on the
   > basis of the valence of active emotion vectors."

   **[inference]** This is the "is the direction one the model uses?"
   question, answered behaviourally (steer per layer and see whether
   behaviour moves) rather than through the downstream disagreement
   that exp193 measures. Together the two give a prediction for our
   lab: a direction the model genuinely reads should show a layer
   profile where the downstream response to a push (our disagreement
   measure) peaks in the same layers where steering changes behaviour.
   A random-word direction should not show that alignment.
2. **Persistence via attention**: emotion representations are
   "cached" and recalled "via attention, when they are needed". At
   person re-references, the probe for that person's emotion "rise[s]
   in later layers as the model retrieves the associated emotion
   concept."
3. **Read-write agreement** (r = 0.85, section 2 item 8) is a
   population-level test of "the model uses this direction the way the
   readout says".
4. They **did not** measure how later layers respond to a push
   (disagreement, KL between layer readouts, attention-pattern shifts)
   or attention-head reading of the directions. The limitations section
   names this as open: "Disentangling these possibilities would require
   more fine-grained interventions and circuit-level analysis."

## 7. Leads found but not read (do not cite as checked)

- arXiv 2604.13466, "Functional Emotions or Situational Contexts? A
  Discriminating Test from the Mythos Preview System Card". A critique
  or extension. Its title suggests it asks whether the vectors encode
  emotion or situation. Not read.
- arXiv 2604.04064, "Extracting and Steering Emotion Representations in
  Small Language Models: A Methodological Comparison". Possibly
  directly relevant to small-model methods. Not read.
- github.com/drgzkr/EmoVecLLM: an open replication that includes
  `EleutherAI/pythia-1.4b`. A summarising fetch said it "drops the
  token-0 attention sink by default" (`EMOVEC_SKIP_FIRST=1`) and builds
  vectors as "difference-of-means between emotion-conditioned and
  neutral generations". No random-direction controls were mentioned.
  The summary's headline numbers look copied from Anthropic's paper, so
  treat its own results as unverified. Notebooks 07–10 are scaffolds.
- github.com/AidanZach/EmotionScope: another open-weight replication.
  Not read.
