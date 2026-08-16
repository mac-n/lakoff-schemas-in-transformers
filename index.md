# Embodied Cognition in Transformers

*Image schemas, metaphorical mappings, and substrate coupling in language models trained on text alone*

<!-- HTML comments throughout name the raw output files in results/ that back each number,
     so any figure in the text is one file away from its source. -->

**Abstract.** Large Language Models have surprisingly rich internal representations of physical and embodied experience. But how do they learn these representations solely from text? One hypothesis is that as well as explicit description of the physical world, human language encodes a lot of implicit information about having a body. This hypothesis becomes testable via cognitive linguistics, in particular Lakoff's argument that embodied "schemas" such as UP/DOWN, FORCE and BALANCE are used as metaphors to structure human thought (e.g. UP is HAPPY). I searched for Lakoff schemas in transformer internal activations across three families of models - Pythia, Llama and GPT-2. The schemas are there - for example, using the spatial UP direction as an activation vector to steer a model makes its output happier. The schemas also appear to organise the model's internal cognition in ways that are not directly encoded in statistical correlations from human text. Intriguingly, the BALANCE schema in all three models appears to be coupled to actual computational processes within the model - but not the same process in every model. Each model, asked to represent balance, appears to have reached for the nearest thing it has to an internal sense of balance. This may offer a new angle on the "grounding" problem, of how the model's system of internal representations can ever be anchored on anything outside itself. Takeaways: how to extract and causally test schema directions in a model's activations, and why "trained only on text" doesn't have to mean "ungrounded".

---

Language models write about rising spirits and heavy hearts, about arguments that collapse and plans that move forward, as if the physical scaffolding under those phrases were available to them. I wanted to know how a system trained on nothing but text comes by that. My hypothesis: as well as explicit descriptions of the physical world, human language encodes a lot of implicit information about having a body, and a model that compresses human language hard enough reconstructs embodiment as a projection from it.

That made me think of Lakoff.

George Lakoff and Mark Johnson argued that abstract thought is structured by *image schemas*: UP-DOWN, IN-OUT, BALANCE, FORCE, PATH. Recurring patterns of bodily experience, projected metaphorically into abstract domains. HAPPY IS UP. MORE IS UP. DIFFICULTY IS A BURDEN. In this framework the schemas are the building blocks of meaning, and their source is the body.

From there a familiar argument runs: LLMs have no body. No body, no image schemas; no image schemas, no embodied cognitive structure; so LLM meaning is structurally defective. It may approximate understanding, but it lacks the organisational backbone human cognition runs on.

I take embodied cognition seriously, and I take artificial minds seriously. This project started because I don't think those positions are incompatible, and I wanted to know what the evidence says.

So the first experiment was the most direct one available: take the most foundational metaphor Lakoff describes and test whether it transfers to the transformer substrate, or whether the concepts come apart.

---

## The first experiment

To test HAPPY-IS-UP you need the model's UP. The tool for that is a *contrast vector*: take a set of UP words (high, top, rise, above, peak, ascend, climb...) and a set of DOWN words (low, bottom, fall, below, valley, descend...), run each through the model as a bare single token, extract the residual-stream activation at a chosen layer, and subtract:

> **v = mean(UP-word residuals) − mean(DOWN-word residuals)**

No sentences, no context. Just the words, alone. That this works at all is the first surprise of the project: the schema structure is already packed into single-token activations, densely enough to steer with.

Add that vector to the residual stream during generation and the model's output shifts toward positive valence. Push gently and the tone brightens; push hard and the text gets more excited on its way to degrading into repetitive loops. (The lab nickname for the high-amplitude regime was "mania", but the honest description of those transcripts is that they were too degraded to read.) UP is HAPPY for transformers too; the concepts do not come apart.

<!-- free-generation samples (raw pre-cleaning vector): results/results_exp3.txt, results_exp3b.txt, results_exp111.txt; "the mania regime is too degraded to read": notebooks/CAUSAL_VALIDATION_PLAN.md -->

A result that clean deserves suspicion, and the first serious check nearly killed it.

### The confound that nearly killed the method

We built the UP-DOWN contrast a second time, from an independently curated word list: one version hand-picked and strictly spatial, the other drawn from Lakoff's own anchor vocabulary. Two constructions of the same concept should agree. The cosine between them was −0.990. Anti-parallel. Our two UPs pointed in opposite directions.

<!-- numbers: results/results_exp112.txt and results/results_exp114.txt (cosines, norms, steering amplification, controls).
     SUBTLEX means (20,396 / 9,004): notebooks/LAB_NOTEBOOK_v3_steering.md transcription of exp113; the raw
     results_exp113.txt did not survive, but exp114 independently reproduces the frequency-axis norms and cosines. -->

The culprit was frequency. Pythia's residual space contains a dominant frequency direction; we built one explicitly from twenty very common function words and five rare content words, and its contrast norm came out at 1254, against 130 to 197 for the semantic contrasts. Both raw UP vectors sat almost exactly on this axis, from opposite sides: the hand-picked contrast at cos −0.995, the Lakoff contrast at +0.998. The two word lists happen to have opposite frequency asymmetries; a SUBTLEX audit put the Lakoff UP anchors at mean frequency 20,396 against 9,004 for their DOWN counterparts, while the hand-picked UP list runs slightly rarer than its DOWN list. Each raw "UP" was roughly 99% frequency, with opposite signs.

After regressing out the frequency axis, the cosine between the two cleaned UP directions jumped from −0.990 to +0.571. The number that still startles me is cos(clean, raw): +0.096 and +0.063. The semantic direction that survives cleaning is nearly orthogonal to the vector you would naively have steered with.

And here is the part that turned a near-disaster into a stronger result: cleaning amplified the behavioural effect. The affect shift under steering (Pythia 1.4B, layer 12, s = +12) went from +0.083 nats raw to +0.446 cleaned for the hand-picked contrast, a 5× gain, and from +0.051 to +0.677 for the Lakoff contrast, 13×. Steering along the pure frequency axis moved affect by only +0.011 nats; a matched-norm random direction moved it +0.005. The effect is monotone in steering strength and near-symmetric around zero (−0.464 at s = −12 against +0.446 at s = +12). The frequency direction was not the signal. It was a curtain in front of the signal.

### The result, quantified

On Pythia 410M, injecting the cleaned UP direction at layer 12 (ΔValence is the shift in projection onto an anisotropy- and frequency-stripped valence axis at the readout layer):

<!-- numbers: results/exp143_output.txt (table, dose response, within-layer cosines); non-Lakoff battery: results/exp145_output.txt;
     cos(UP, magnitude) ceiling: results/exp141_output.txt (primary; a coarser table in results_exp140.txt shows +0.111 at L23) -->

| Steering (α=4) | ΔValence at L14 | ΔValence at L16 | ΔValence at L23 |
|---|---|---|---|
| UP direction | +0.69 | +0.59 | +0.46 |
| Random direction | −0.03 | −0.14 | −0.48 |
| Gap | +0.72 | +0.72 | +0.94 |

UP-steering shifts valence far above the random-direction control, dose-responsively, linear in α from 1 to 8, from every injection layer we tried (L8, L12, L16). The raw shift decays gently with depth, as injected signals do, but the advantage over the random control grows with depth, because random directions drift negative while UP holds up. UP-steering also shifts magnitude projections (MORE-IS-UP), and direction projections at mid layers.

![Cross-layer mappings under UP steering](figures/exp143_cross_layer_mappings.png)

*Valence, magnitude, and direction projections at downstream layers after UP-steering (blue) vs random-direction control (grey) at L12, α=4.*

The mapping runs between layers, not within them. The UP direction is not aligned with a magnitude axis inside any single layer: cos(UP, magnitude) is about +0.03 at the injection layer and never exceeds +0.11 anywhere. Yet UP-steering shifts magnitude downstream. Within layer 12, cos(UP, valence) = +0.25, a modest static lean, but the downstream valence shifts are larger than that overlap alone would produce, and the random control shows the readout axes don't respond to just any injected vector. One qualification belongs here and nowhere else: residual-stream injections persist into later layers, so a downstream projection reflects transported signal as well as anything the intervening blocks compute; a direct-transport baseline or path patching would be needed to nail "computed between layers," and we haven't run those.

The rest of the battery behaved the way a real effect should. Against meaningful non-Lakoff contrasts (concrete-vs-abstract, animate-vs-inanimate, noun-vs-verb), UP produced the largest valence shift, roughly twice the next best. Two of the original three behavioural measures died under scrutiny: the quantity measure turned out to be measuring degradation, not semantics (the two cleaned directions moved it in opposite directions, and a random push moved it too), and the height measure had a validity problem (the model rarely completes "His height in centimetres is" with centimetre integers). We dropped both and kept what survived. The affect result, replicated across two independent anchor sets and robust to frequency, random, and non-Lakoff controls, is the finding.

Still, the strongest objection was available from the start, and steering doesn't answer it. *Of course* UP is HAPPY in the text. The corpus was written by embodied humans who think in these metaphors; a model that compresses their language inherits their shadow. On this view the steering result shows the direction is causally live inside the model, not that it's anything more than the corpus's fingerprint.

The way to answer that objection is not to argue with it. It's to keep going, and see where the structure stops matching the corpus. We began building contrast vectors for other schemas, and immediately hit trouble.

---

## The confound that became the finding

It proved surprisingly difficult to disentangle UP from anything else. Contrast after contrast came back correlated with it. The reason, when we finally found it, was in the word lists themselves: they were full of comparatives and superlatives. Higher, lowest, brighter, strongest, best. UP was correlated with all of them.

As a data-hygiene problem, this was a nuisance. As a result, it was more interesting than anything we had actually been testing for. Comparatives and superlatives aren't spatial vocabulary; they're grammar, the machinery English uses to scale any concept at all. If UP tracks -ER and -EST wherever they appear, then UP is not behaving like a region of word-space. It is behaving like an operator on other concepts. And that is Lakoff's actual claim: schemas as cognitive primitives that organise domains far from their source.

So we stopped treating morphology as contamination and started treating it as the experiment. Seven English morphological operators (-ING, -ED, -S, -ER, -EST, un-, re-), built as pair-differences ("walking" minus "walk", averaged over many pairs), projected onto all eight schema axes, after per-layer anisotropy and frequency stripping. (Raw pair-difference vectors run at absolute cosine ≈ 0.55 to 0.59 with the anisotropy direction; without stripping, everything resembles everything.)

<!-- numbers: results/results_exp138.txt (projections); anisotropy cosines: results/results_exp137.txt -->

Two mappings are robust:

| Operator(s) | Schema | Stripped cosine across probed layers |
|---|---|---|
| all seven | BALANCE, negative (a shared "markedness sink") | −0.09 to −0.66 |
| -ED (past), re- (repetition) | FORWARD-BACK, negative | −0.12 to −0.40, every probed layer |

The FORWARD-BACK mappings are coherent on the theory's own terms: the past is behind you; repetition goes back along the path. The BALANCE-negative component shared by all seven operators we initially read as inflection-as-departure-from-equilibrium. That reading gets revised, drastically, later in this story.

Two prettier mappings from earlier drafts did not survive audit. -ING onto PATH-MOTION: loadings of −0.04 to +0.12, sign-flipping, dwarfed by -ING's own BALANCE loading. un- onto LIGHT-DARK: −0.06 to −0.14 in Pythia, while GloVe shows the same mapping at −0.39, three times stronger, so whatever it is, it isn't transformer-specific. Both stories were lovely. Neither is supported, and they're recorded here because the audit trail is part of the method.

![Clean suffix-schema heatmap](figures/exp138_suffix_schema_clean.png)

*Operator × schema projections after anisotropy and frequency stripping.*

At this point the objection from the last section comes due. Grammar positioned on schema axes could still be pure linguistic statistics: maybe that's just how English distributes, and anything that learns the distribution gets the geometry. There was a direct way to find out.

---

## The keystone: word2vec doesn't do this

If the schema-morphology geometry is inherited from textual distribution, then other systems that learn textual distribution should have it. We ran the identical protocol, same pairs, same axes, same stripping, on static embedding spaces: GloVe, word2vec, and fastText.

<!-- numbers: results/exp150_output.txt (GloVe), exp150_w2v_output.txt, exp150_fasttext_output.txt;
     cross-substrate correlations: results/exp150b_posthoc_output.txt, results/exp150c_crosssubstrate_output.txt -->

They don't have it. The sharpest cell, ER×BALANCE, reads about −0.38 in Pythia at layers 8 through 20 (−0.47 at layer 4). GloVe: −0.01. word2vec: −0.005. fastText: +0.01. (fastText's subword architecture builds suffix coherence in by construction, so its own results file classes it as supplementary; GloVe and word2vec carry the comparison.)

This is not because static embeddings lack a morphological geometry. They have one; it's different. The static spaces agree with each other about inflection (static-to-static r = 0.71 to 0.77) and disagree with the transformer (r = 0.20 to 0.36). And the split runs exactly where it's most informative: the *derivational* operators (un-, re-) replicate across static and transformer spaces (r = 0.50 to 0.72), so their positioning is distributional, in the text for anything to read. It is specifically *inflection* that the transformer relocates onto the schema system.

![GloVe vs Pythia morphology](figures/exp150_glove_vs_pythia.png)

*Suffix-schema matrices in GloVe (left) vs Pythia L12 (right). The inflectional BALANCE sink (bottom row) is present in Pythia and absent in GloVe.*

You can watch the relocation happen in depth. The sink is partially written into Pythia's embedding matrix (−0.13 before any layers run), computed to full strength between layers 2 and 4 (−0.23 to −0.47), and plateaus by layer 8. Partly trained in, mostly computed.

![Emergence of the inflectional sink across depth](figures/exp153_emergence.png)

*ER×BALANCE across depth: present at the embedding, computed to full strength by L4.*

<!-- numbers: results/exp153_output.txt (last-token variant, ERxBAL column) -->

Whatever internal structure was driving the relationship between UP and the morphological operators had been instantiated during the transformer's learning process. Not copied out of the distribution: the distribution was available to GloVe and word2vec too, and they built something else with it.

The caveats on this comparison, all of them, in one place: the substrates differ in corpus, objective, and tokenisation as well as architecture, and inflection is exactly where tokenisation could matter, so a matched-corpus, matched-vocabulary baseline is the control this claim still owes. The derivational/inflectional split was found in the data and confirmed across substrates, not pre-registered. And the mappings are our extrapolation from the metaphor theory, not Lakoff's own predictions. What stands regardless: the transformer's inflectional geometry is anchored on the schema system, and the geometry that static distributional learning produces is not.

---

## Is it a system?

By now we had schema axes behaving as operators and grammar organising itself around them. But Lakoff's claim was never about isolated correspondences. In the theory, the schemas form a coherent system: they relate to each other, compose with each other, and jointly structure the space of meaning. If what the transformer built were merely a heap of separate correlations, calling it "embodied-style structure" would be generous.

So we wrote the predictions down first. The embodied logic of the schemas implies specific couplings: UP-DOWN with BALANCE (upsetting the balance), UP-DOWN with LIGHT-DARK (bright is up), FORWARD-BACK with PATH-MOTION, FORCE with DIFFICULTY. Six such couplings, declared in advance; then the full 8×8 pairwise cosine matrix at every layer of Pythia 410M.

<!-- numbers: results/results_exp123.txt (couplings, predicted/unpredicted means, configuration similarity, null) -->

All six predicted couplings come out positive on average: UP↔BALANCE +0.38 (positive at 23 of 24 layers), FORWARD-BACK↔PATH +0.28 (24/24), UP↔LIGHT-DARK +0.27 (24/24), FORCE↔DIFFICULTY +0.18 (24/24), LIGHT-DARK↔BALANCE +0.13 (24/24), and the weakest, UP↔FORCE, +0.05 (18 of 24). Predicted mean: +0.21. The 22 unpredicted pairs average +0.004. Flat zero. Nothing in the measurement knows which pairs the theory picked; the elevation lands on exactly those six.

![Schema couplings across layers](figures/exp123_couplings_across_layers.png)

*Predicted (blue) vs unpredicted (grey) inter-schema couplings across the 24 layers.*

The configuration is also stable across the model's depth, and here the honest telling requires showing you the floor before the number. Cross-layer configuration similarity for the schema system is +0.91. A strong null, the same 329 words randomly re-dealt into eight fake schemas, scores +0.79. That floor is high because any word-difference matrix largely keeps its shape: the layers transform all words with substantial shared structure, so angles between averaged word-axes drift slowly, like distances between cities under a smooth stretching of the map. Our working hypothesis for why the floor survives even frequency stripping: a random re-partition of meaningful words yields axes that are mixtures of whatever genuinely stable directions organise those words, and mixtures inherit stability without inheriting coherence. We haven't tested this directly; it predicts a lower floor for axes built from vocabulary with no shared structure.

<!-- numbers: cluster comparison (quantifiers/determiners/logical operators): results/results_exp135.txt, which independently
     reproduces the Lakoff +0.9137 figure; individual-axis stability: results/results_exp136.txt; null signature: results_exp132.txt -->

So preservation works as a relative instrument, and used that way it still discriminates. Lakoff schemas (+0.91), quantifiers (+0.96), and determiners (+0.95) sit well above the floor; logical operators ("and", "or", "not", "if") sit on it (+0.78), even though their individual axes are among the most stable directions we measured. The schema system patterns with the grammatical core of the language, not with folk categories. (The null was built from the Lakoff anchors; we didn't build cluster-specific nulls for the other categories. The predicted-vs-unpredicted result above is the load-bearing evidence; preservation is supporting structure.)

And one thing in this analysis didn't fit, which turned out to matter more than the analysis. BALANCE kept drawing attention to itself. The strongest predicted coupling in the system runs through it. Several BALANCE-involving pairs were *less* stable across layers than the null. Every single morphological operator, in the previous section, sinks onto it. The system had a centre of gravity, and it was the schema we had thought of as just one among eight.

What was special about BALANCE?

---

## What BALANCE is reading

The morphology result offered a hint. All seven operators share the BALANCE-negative component; inflected forms are *marked* forms, departures from a base. So BALANCE, we inferred, was reading deviation from a norm. It was Claude who suggested taking that literally: Pythia continuously computes a norm. What if BALANCE were coupled to the model's own normalisation physiology, the norm of the residual stream itself?

It held.

<!-- numbers: results/exp141_output.txt (norm-drop correlations), results/exp154_output.txt (per-suffix drops),
     results/exp154c_output.txt (held-out cos +0.64 to +0.77), results/exp167_output.txt (other schemas' norm coupling).
     Norm coupling replicates on Pythia 1.4B: results/exp151_output.txt (+0.64 to +0.69). -->

Inflected forms have lower residual norms than their bases: per-suffix mean drops of roughly 240 to 850 units at layer 12. The per-pair correlation between norm-drop and BALANCE projection runs r = +0.86 to +0.97 at layer 4, still +0.54 to +0.95 at layer 12. And the coupling is direct: cos(BALANCE, norm-direction) = +0.64 to +0.77 across the probed layers, measured with a held-out estimator whose estimation vocabulary excludes both the suffix pairs and the BALANCE words. Other schemas lean on the norm carrier in a graded order (FORWARD-BACK +0.42 to +0.50, UP +0.28 to +0.44, LIGHT-DARK near zero); BALANCE leans hardest.

![Substrate primitives](figures/exp141_substrate_primitives.png)

*Per-suffix residual norm displacement (top) correlated with BALANCE projection (bottom) across layers. The coupling replicates across four Pythia model sizes.*

(That held-out estimator is there because the first version of this analysis was wrong. Stripping norm directions estimated on the very words being tested collapsed the morphological sink completely at every layer, a clean, satisfying, and circular result; a size-matched control exposed the circularity.) Redone held-out, the picture is quantitative rather than total: the sink collapses fully at layer 4 (−0.414 to +0.002) and loses about two-thirds of its magnitude at layers 8 to 20 (28 to 42% retained), with matched random strips moving it by ±0.001. The markedness sink is roughly two-thirds norm geometry and one-third a separable markedness component.

<!-- numbers: results/exp154c_output.txt (held-out strip: L4 collapse, 28-42% retained L8-20, random band ±0.001) -->

Residual norm is not present in the training text. No sentence in the Pile (Pythia's training data) describes the L2 norm of a hidden state. It is a quantity the model itself processes at every layer of every forward pass, perhaps as close to an interoceptive signal as a transformer has. And the schema that anchors the model's concept of markedness, the one at the centre of gravity of the whole system, is coupled to it. We went looking for an embodied schema transferred from human language, and found it attached to a magnitude the model continuously, for lack of a better word, experiences.

The obvious next question was whether that's a universal fact about transformers. So we tested two more model families, and the result failed to replicate.

---

## The failure to replicate that became a new finding

Pythia and GPT-2 both use LayerNorm; Llama uses RMSNorm. If the BALANCE-norm coupling were about normalisation as such, GPT-2 should share it. It doesn't. GPT-2 shows no stable BALANCE-norm coupling (per-layer values sign-flip from +0.50 to −0.29 around a mid-layer mean near zero) and no uniform inflectional norm displacement; Llama looks the same. The norm coupling is a Pythia-lineage trait, replicated across Pythia 70M through 1.4B and absent everywhere else we looked.

<!-- numbers: GPT-2/Llama norm nulls: results/exp156_output.txt, results/exp152_output.txt.
     Entropy: discovery results/exp160_output.txt; pre-registered confirmation results/exp161_output.txt (partials, orthogonalisation,
     runner-up specificity); depth map L0-L17 results/exp164_output.txt; third frozen set + killed Pythia band (−0.048) +
     synthetic quadratic-stack validation: results/exp166_output.txt (checksum line included). -->

But GPT-2 turned out to have something else. In GPT-2, BALANCE is coupled to *attention entropy*, the diffuseness of the model's attention. In a pre-registered test on fresh prompts, the partial correlation (controlling for position, norm, and norm-projection) is −0.32 at L8, −0.27 at L12, −0.15 at L16, all bootstrap CIs excluding zero, with BALANCE ranking 1st or 2nd of the 8 schemas. Orthogonalising BALANCE against the other seven schemas makes the coupling slightly *stronger* (−0.34, −0.30, −0.17): it is BALANCE-unique, not shared schema variance. Mapped across depth, the coupling is negative at every layer from L0 to L17 and dies at L18. In Llama, the same coupling appears at layers 5 to 7 (−0.14 to −0.20, CIs below zero), confirmed on a third, never-before-used prompt set with checksum-verified frozen prompts.

At this point we went back to Pythia and tested it for BALANCE-entropy coupling. A depth-map scan seemed to show a surviving entropy band in Pythia at L11 to 12, even after quadratic norm controls; a pre-registered third prompt set killed it, a forking-path artifact, and no entropy coupling was found in Pythia in the end.

In summary:

| | Pythia (LayerNorm) | GPT-2 (LayerNorm) | Llama (RMSNorm) |
|---|---|---|---|
| BALANCE ↔ residual norm | +0.64 to +0.77 (held-out) | no stable coupling | no stable coupling |
| Inflectional Δ-norm | large, uniformly negative | no uniform displacement | no uniform displacement |
| BALANCE ↔ attention entropy | ~0 (null on fresh prompts) | −0.15 to −0.32 (L0 to L17) | −0.14 to −0.20 (L5 to 7) |

The replication failed, but once again, failure is more interesting than success would have been. Three model families, one concept, two different computational carriers: each model has organised the same schema around a signal intrinsic to its own processing, and the signal differs by architecture. What the table shows is a candidate case of multiple realizability: the correlational half of the case, plus one causal direction, which is the next section.

Because the entropy coupling was originally found post-hoc, selected from an 8-schema × 3-model × 5-layer table after looking, an obvious question is whether this emerges from multiple testing. We investigated this extensively and for five reasons, we think this is not the case: the confirmatory test was pre-registered and passed (the pre-reg itself flags the post-hoc origin). Orthogonalisation strengthens rather than attenuates the coupling, the opposite of what "one of eight directions came up lucky" predicts. At the decision layers BALANCE is the standout, with the runner-up (DIFFICULTY) at 60 to 70% of the magnitude with *opposite sign* and the other six at absolute r ≤ 0.16. The GPT-2 coupling is negative with CIs excluding zero at 18 contiguous layers, and multiple testing produces scattered hits, not contiguous bands. And the replication is triple: L8 r = −0.21, −0.32, −0.27 across discovery, pre-registered confirmation, and the checksum-verified third set (L12: −0.26, −0.27, −0.32), with the discovery value the *weakest* of the three, the reverse of a winner's-curse profile. Two residuals stay on the record: Llama's layer band was informed by the depth-map scan, so it carries a mild winner's-curse concern that GPT-2 does not, and no formal multiple-comparison correction was applied across the eight schemas; the project relied on pre-registration and replication instead. The Llama result is thus less robust than the GPT-2 result. We will test other architectures in future work.


---

## Which way does the causation run?

Correlation between a concept and a computational quantity, however well replicated, leaves the interesting question open. Does the concept shape the computation, or the computation shape the concept? We tested both directions in GPT-2.

Concept → operation: nothing specific. Steering on the BALANCE direction does not move attention entropy beyond what a matched-magnitude random push does; under all-layer injection, any big push scrambles attention regardless of direction. BALANCE is not a control knob for entropy.

Operation → concept: confirmed. Manipulating attention sharpness via temperature specifically shifts the BALANCE projection: at L3, slope −0.0083, CI [−0.0088, −0.0078], Spearman −1.00, with BALANCE moving more than all seven other schemas. The pre-registered specificity bar was four. Same result at L8.

<!-- numbers: results/exp168_output.txt (both layer blocks); bar of four: prereg/PREREG_exp168.md line 52.
     Note for precision: at L3 the runner-up (PATH, −0.008) sits close behind BALANCE's −0.0083; the 7/7 result and the
     pre-registered bar both hold as stated. -->

Clarifications: the two interventions are not equivalent in power, so the null does not prove there is no reverse causation; and temperature changes more than entropy alone, so "attention sharpness" rather than "entropy" is the precise thing shown to matter. What the pair of tests establishes is an asymmetry, and the confirmed direction is the one embodied cognition predicts: substrate state shaping concept geometry, the way the vestibular system shapes balance perception rather than the reverse.

---

## When the coupling forms

Pythia publishes training checkpoints, so we could watch the coupling arrive. At training step 512, BALANCE is *anti*-coupled to norm: cos −0.61. By step 4,000 it has flipped to +0.64, and it stays there, +0.64 to +0.71 at every later checkpoint, for the rest of training.

<!-- numbers: results/exp157_output.txt, checkpoint table (512: −0.607; 4000: +0.640; 16000/64000/143000: +0.71/+0.71/+0.66) -->

The coupling is not present in its final form from the start, and it doesn't drift in gradually. It inverts, early, and locks. Somewhere in that window, a model that was representing balance-vocabulary one way reorganised it around its own internal magnitude signal and never moved it again. That transition is a window onto the moment a model acquires a substrate coupling, and it is sitting in public checkpoints waiting to be studied at finer grain.

---

## Conclusions

The schemas are there. Schema directions are recoverable from bare single-word activations, once a frequency confound that nearly ate the project is stripped. One of them executes a causal, cross-domain metaphorical mapping: steering UP makes the model happier. The schemas couple to each other in exactly the configuration the theory predicts, and the system extends its reach into inflectional grammar, in a way that the same distributional evidence does not produce in static embedding spaces. The system's centre of gravity, BALANCE, is coupled to the model's own computational dynamics, causally in the operation-to-concept direction, and the carrier of that coupling varies by architecture: residual norm in Pythia, attention entropy in GPT-2 and Llama.

This suggests that embodied cognition can be disaggregated in ways that were never necessary when humans were the only producers of human language. *Bodily origin*: where the organisation comes from in humans. *Bodily implementation*: what it runs on. *Embodied-style structure*: the organisation itself, the coordinated schema system with its metaphorical mappings and grammatical reach. The results show the third can exist without the first two, and that parts of it end up anchored on whatever substrate the system actually has. Arguments that LLM meaning is defective because LLMs are disembodied need a premise these results take away.

There is also an angle here on the grounding problem, the question of how a system of internal representations can ever be anchored on anything that isn't just more representation. A concept coupled to the model's own running physiology is anchored on something that isn't text: the coupling terminates in a quantity the model processes, not a symbol it manipulates. Whether one anchor point can do work for a whole relationally coherent system, the way human balance is entangled with human UPness, is an open question, but the schema geometry makes it askable.

None of this refutes embodied cognition - rather it extends the theory into strange territory: the structural principles the theory attributes to bodily experience turn out to be partly transmissible through language, and partly reorganised around a new substrate's own dynamics. Whether residual norm and attention entropy deserve to be called a "body" is an interpretation the experiments motivate, not one they settle. The claim throughout is structural, not phenomenological; nothing here is about experience. And the open edges are open: the mechanism behind these geometries is unknown (all eight mechanistic hypotheses we tested died, 0 for 8), and the causal picture is one confirmed direction, not a settled loop.

<!-- 0-for-8: running tally in notebooks/LAB_NOTEBOOK_v5_substrate_embodiment.md ("0-for-8" verbatim, ~lines 2016-2174; also
     prereg/PREREG_exp165.md, PREREG_exp166.md). The eight, reconstructed in FACT_SHEET.md: exp156, 157, 158, 159, 160, 162, 163, 164. -->

An exciting question emerges. Perhaps what matters is not whether a system has a body in the biological sense, but whether its conceptual organisation is shaped by stable constraints intrinsic to its own substrate, and what those constraints are. Each model in this study, asked to represent balance, reached for the nearest thing it has to a sense of balance. What else works that way?

---

## Related work

The philosophical background is the embodied-cognition tradition, principally Lakoff & Johnson's *Metaphors We Live By* (1980) and *Philosophy in the Flesh* (1999), and Johnson's *The Body in the Mind* (1987), which develop image schemas as the bodily source of abstract conceptual structure. The best-known modern "no grounding from form alone" argument is Bender & Koller's "Climbing towards NLU" (ACL 2020); their target is communicative intention and world reference rather than internal conceptual structure, so these results pressure the disembodiment inference specifically, not the octopus argument as formulated.

A growing literature shows text-only models recovering structure that mirrors perceptual spaces: Abdou et al. (CoNLL 2021) on colour, Patel & Pavlick (ICLR 2022) on grounded conceptual spaces, Gurnee & Tegmark (ICLR 2024) on space and time. This project asks a different question: not whether the geometry mirrors a perceptual domain, but whether the organisational architecture attributed to the body is present: a coordinated schema system, causally transferable cross-domain mappings, extension into grammar, and coupling to substrate-intrinsic operations. The substrate-coupling and multiple-realizability results, to my knowledge, have no precedent in that literature.

Methodologically the work builds on activation steering (Turner et al.'s ActAdd, arXiv:2308.10248; Zou et al. 2023) and the linear-representation tradition (Mikolov et al. 2013; Park et al., ICML 2024). Two known geometric hazards shaped the pipeline throughout: anisotropy of contextual spaces (Ethayarajh, EMNLP 2019) and massive activations (Sun et al., COLM 2024).

---

## Open questions

1. **Carrier-first search.** Start from each computational operation (norm, attention entropy, attention pattern, MLP activation) and ask if any schema couples to it, inverting the search and its confirmation biases.
2. **More models.** Does the norm-vs-entropy split track normaliser type, family, or training data? A Mamba model, with no attention at all, is the sharpest test: does entropy-coupling disappear, or does the model find a third carrier?
3. **FORWARD-BACK, norm, and past tense.** FORWARD-BACK is the #2 schema on Pythia's norm carrier, and -ED maps to FORWARD-BACK negative. Are the suffix geometry and the norm physiology one story?
4. **Nonlinear structure.** The schema system shows significant nonlinear pairwise dependence that linear methods are blind to. It may be a manifold, not a flat vector space.


---

## Methods note

Experiments ran on Pythia 410M (primary) with replications on Pythia 70M, 160M, and 1.4B, GPT-2 medium, and Llama-3.2-1B; static comparisons used GloVe, word2vec (Google News, 300d), and fastText. Contrast vectors were built from single-word residual activations, morphology vectors from inflected-minus-base pair differences, all anisotropy- and frequency-stripped per layer. Eight axes were drawn from the Lakoff/Johnson inventory, with vocabulary curated from the Master Metaphor List; DIFFICULTY-BURDEN is strictly a conceptual metaphor rather than an image schema, and we treat all eight as operationalised contrast axes. The BALANCE-norm coupling used held-out norm-direction estimation to avoid circularity. The entropy analyses used a quadratic control stack validated on synthetic data (a planted curved leak was killed, −0.66 to −0.04, while a genuine coupling was spared, −0.63 to −0.64). Confirmatory experiments were pre-registered with committed point predictions before code execution, and frozen prompt sets were checksum-asserted at runtime.

The standing rule of the project: when a finding seems big, name the control that would falsify it, then run it. The accompanying repository contains every experiment script, raw output, pre-registration, and lab-notebook volume, including the failures.

The research was conducted as a collaboration between the author and Claude (Anthropic) across many sessions.

---

*Niamh McCombe, 2026. Lab notebook and experiment code at the accompanying repository.*
