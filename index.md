# Embodied Cognition in Transformers

*Image schemas, metaphorical mappings, and substrate coupling in language models trained on text alone*

> **Status: v0.2, working draft, shared early.** This research was conducted
> collaboratively with Claude (Anthropic) across an extended series of
> experiments. Claude drafted much of the present exposition from our shared
> lab notebooks; I directed the research and am continuing to revise the
> interpretation and writing. Public here does not mean canonical.

---

**TL;DR.** Embodied cognition theory, principally Lakoff and Johnson, holds that abstract thought is organised by image schemas (UP-DOWN, BALANCE, PATH, FORCE) rooted in bodily experience. A common inference drawn from this: language models have no body, so their conceptual organisation must be structurally defective. We tested that inference directly, across three model families, with pre-registered confirmatory experiments. Text-only transformers turn out to exhibit a surprising amount of the conceptual organisation the theory associates with bodily experience:

- Image-schema-like directions exist in the residual stream, recoverable from single-word activations, and they form the specific relational system the theory predicts (predicted inter-schema couplings +0.21, unpredicted +0.00).
- Steering on the UP direction causes metaphor-congruent changes in valence (HAPPY-IS-UP), beating a random-direction control by +0.72 to +0.94, dose-responsively.
- The transformer reorganises inflectional morphology onto the schema axes. Off-the-shelf static embeddings (GloVe, word2vec) learned from textual distribution don't: the inflectional BALANCE sink is about −0.38 in Pythia and ~0.00 in static spaces.
- BALANCE is systematically coupled to the models' own computational dynamics, and the carrier varies by model: residual norm in Pythia, attention entropy in GPT-2 and Llama. A candidate case of multiple realizability.
- Manipulating the computational operation shifts the concept (specificity: 7 of 7 schemas beaten). Steering the concept does not detectably move the operation.

What this shows is that "no biological body" does not straightforwardly imply "no embodied-style conceptual organisation." Bodily origin, bodily implementation, and embodied-style structure are not the same thing, and these models force the distinction.

---

## The question

George Lakoff and Mark Johnson argued that abstract thought is structured by *image schemas*: UP-DOWN, IN-OUT, BALANCE, FORCE, PATH, patterns arising from bodily experience and projected metaphorically into abstract domains. HAPPY IS UP. MORE IS UP. DIFFICULTY IS A BURDEN. In this framework the schemas are the building blocks of meaning, and their source is the body.

From there, a familiar argument runs: LLMs have no body. No body, no image schemas. No image schemas, no embodied cognitive structure, so LLM meaning is structurally defective. It may approximate understanding, but it lacks the organisational backbone that human cognition runs on.

Where this project comes from: I take embodied cognition seriously, and I take artificial minds seriously. It started because I don't think those positions are incompatible, and I wanted to know what the evidence says.

There is an obvious objection, and it deserves to be stated before the evidence: *of course* the schemas are in the text statistics. The training corpus was written by embodied humans who think in these schemas; a model that compresses that corpus will inherit their shadow. On this view everything below is distributional semantics with extra steps. Two of the findings are aimed at that objection. Finding 3 shows that static embeddings, learning from the same broad source of evidence (textual distribution), end up with a *different* morphological geometry: the transformer doesn't inherit the schema-anchored organisation, it builds it. Finding 4 shows that one schema is coupled to the model's own computational operations, quantities like residual norm and attention entropy that do not exist in the training text at all. Whatever those two results are, they are not properties of the corpus.

Four findings, after a note on method:

1. **Steering on a spatial UP vector shifts the model's valence.** HAPPY-IS-UP behaves as a causal, cross-layer mapping.
2. **The schema axes form a stable relational system** across the depth of the model, with the specific couplings the theory predicts.
3. **Morphological operators position on the schema axes** in the transformer, but not in static embeddings.
4. **BALANCE is coupled to computational operations**, with causal evidence running from operation to concept.

---

## Method: contrast vectors from single words

The basic tool of this work is the *contrast vector*: take a set of UP words (high, top, rise, above, peak, ascend, climb...) and a set of DOWN words (low, bottom, fall, below, valley, descend...), run each word through the model as a bare single token, extract the residual stream activation at a chosen layer, and compute:

> **v = mean(UP-word residuals) − mean(DOWN-word residuals)**

This vector is the UP-DOWN schema direction in the model's internal geometry. The same procedure builds LIGHT-DARK, FORWARD-BACK, BALANCE, IN-OUT, PATH-MOTION, FORCE, and DIFFICULTY-BURDEN: eight axes drawn from the Lakoff/Johnson inventory, with vocabulary curated from the Master Metaphor List. (One taxonomic note: DIFFICULTY-BURDEN is strictly a conceptual metaphor rather than an image schema; we treat all eight as operationalised contrast axes and let the results speak to how schema-like they are.)

The surprise is that you don't need sentences. No context, no sentence frame, just the word alone: the schema structure is already packed into single-token activations, densely enough to steer with (Finding 1). Static embeddings have supported analogy arithmetic since word2vec, but a bare token's contextual activation carrying a *causally transferable* cross-domain direction, from spatial verticality to affective valence, is not something analogy arithmetic ever showed.

### The frequency confound

Getting there required surviving a confound that nearly killed the method.

We built the UP-DOWN contrast twice, from two independently curated word lists: a hand-picked strictly-spatial set, and a set drawn from Lakoff's anchors. Two constructions of the same concept should agree. Instead, the cosine between them was −0.990. Anti-parallel.

The culprit was frequency. Pythia's residual space contains a dominant frequency direction (we built one explicitly from twenty very common function words versus five rare content words; its contrast norm came out at 1254, against 130 to 197 for the semantic contrasts). Both raw UP vectors sat almost exactly on this axis, from opposite sides: the hand-picked contrast at cos −0.995, the Lakoff contrast at +0.998, because the two word lists happen to have opposite frequency asymmetries (a SUBTLEX audit put the Lakoff UP anchors at mean frequency 20,396 against 9,004 for their DOWN counterparts, while the hand-picked UP list runs slightly rarer than its DOWN list). Each raw "UP" vector was roughly 99% frequency, with opposite signs.

After regressing out the frequency axis, the cosine between the two cleaned UP directions jumped from −0.990 to +0.571. The number that still startles me is cos(clean, raw) = +0.096 and +0.063: the semantic direction that survives cleaning is nearly orthogonal to the vector you would naively have steered with.

And cleaning amplified the behavioural signal rather than destroying it. The affect shift under steering (Pythia 1.4B, layer 12, s = +12) went from +0.083 nats raw to +0.446 cleaned for the hand-picked contrast (5×), and from +0.051 to +0.677 for the Lakoff contrast (13×), roughly a 2× odds-ratio shift on positive-versus-negative affect tokens. Steering along the pure frequency axis moved affect by only +0.011 nats, so this is not a Pollyanna-principle frequency artifact; a matched-norm random direction moved it by +0.005. The effect is monotone in steering strength and near-symmetric around zero (−0.464 at s = −12 vs +0.446 at s = +12).

Two other behavioural measures in the original battery died under scrutiny. The quantity measure was unreliable (the two cleaned directions shifted it in opposite directions, and a random push moved it too: it was measuring degradation, not semantics). The height measure had a validity problem (the model rarely completes "His height in centimetres is" with centimetre integers). Both cleaned directions did shift expected height positively, consistent with MORE-IS-UP, but we don't rest weight on a DV with a known validity issue. The affect result, replicated across two independent anchor sets and robust to frequency and random controls, is the finding.

---

## Finding 1: Steering on UP makes the model happier

If the UP-DOWN direction really carries the HAPPY-IS-UP mapping, adding it to the residual stream should shift output toward positive valence. The result, on Pythia 410M, injecting at layer 12 (ΔValence is the shift in projection onto an anisotropy- and frequency-stripped valence axis at the readout layer):

| Steering (α=4) | ΔValence at L14 | ΔValence at L16 | ΔValence at L23 |
|---|---|---|---|
| UP direction | +0.69 | +0.59 | +0.46 |
| Random direction | −0.03 | −0.14 | −0.48 |
| Gap | +0.72 | +0.72 | +0.94 |

UP-steering shifts valence far above the random-direction control. It's dose-responsive, linear in α from 1 to 8, and persists across injection layers (L8, L12, L16). The raw shift decays gently with depth, as injected signals do, but the advantage over the random control grows (+0.72 at L14 to +0.94 at L23), because random directions drift negative while UP holds up. UP-steering also shifts magnitude projections (MORE-IS-UP), and direction projections at mid layers.

![Cross-layer mappings under UP steering](figures/exp143_cross_layer_mappings.png)

*Valence, magnitude, and direction projections at downstream layers after UP-steering (blue) vs random-direction control (grey) at L12, α=4.*

(Free-generation samples exist in the project record, but they were produced with the raw pre-cleaning vector and are noisy seed-to-seed, so we don't quote them.)

### Cross-layer, not within-layer

The UP direction is not aligned with a magnitude axis within any single layer: cos(UP, magnitude) is about +0.03 at the injection layer and never exceeds +0.11 anywhere. Yet UP-steering shifts magnitude projections downstream. Within layer 12, cos(UP, valence) = +0.25, a modest static lean, but the downstream valence shifts (+0.46 to +0.69) are larger than that overlap alone would produce, and the random control shows the readout axes aren't picking up just any injected vector. The mappings look computed between layers rather than stored as within-layer alignments.

One caveat on that phrasing: residual-stream injections naturally persist into later layers, so a downstream projection reflects transported signal as well as anything the intervening blocks compute. The low within-layer cosines are suggestive of computation, not proof of it; a direct-transport baseline or path patching would be needed to nail "executed by the circuitry," and we haven't run those.

The other controls: a "meaningful non-Lakoff" battery (concrete-vs-abstract, animate-vs-inanimate, noun-vs-verb) showed UP produces the largest valence shift, roughly 2× the next best. It also revealed a structural late-layer pull toward valence for orthogonal meaningful directions generally (about +0.20 to +0.30), with one unexplained exception (animate-vs-inanimate went negative). The specific UP→valence mapping is real and largest, but it is not the only thing pulling toward valence at late layers.

Could the mapping itself just be inherited from text statistics? Human writers do encode HAPPY-IS-UP in their word choices. Distinguishing "learned the metaphor from text" from "architecture reconstructs the metaphor" needs a comparison substrate. That is what Finding 3 provides.

---

## Finding 2: A stable schema relational system

Eight recoverable directions would be interesting but not surprising; any clustering method finds directions. The real question is whether they form a *system*: do the schemas relate to each other the way the theory says they should, and does that relational structure persist across the model's depth?

The embodied logic of the schemas predicts specific couplings: UP-DOWN with BALANCE (upsetting the balance) and with LIGHT-DARK (bright is up), FORWARD-BACK with PATH-MOTION, FORCE with DIFFICULTY. We declared six such couplings as predictions, then measured the 8×8 pairwise cosine matrix at every layer of Pythia 410M.

The configuration holds. Cross-layer configuration similarity is +0.91, against a strong null (random anchor partitions, same words) of +0.79. All six predicted couplings come out positive on average: UP↔BALANCE +0.38 (positive at 23 of 24 layers), FORWARD-BACK↔PATH +0.28 (24/24), UP↔LIGHT-DARK +0.27 (24/24), FORCE↔DIFFICULTY +0.18 (24/24), LIGHT-DARK↔BALANCE +0.13 (24/24), and the weakest, UP↔FORCE, +0.05 (positive at 18 of 24). The predicted mean of +0.21 is over all six, weak one included. The 22 unpredicted couplings average +0.004. What's present is specifically the predicted system, not a generic clustering.

![Schema couplings across layers](figures/exp123_couplings_across_layers.png)

*Predicted (blue) vs unpredicted (grey) inter-schema couplings across the 24 layers.*

Depth caveat up front: the configuration is established early and persists through the stack, but the final layer partially reorganises toward the output (L23's similarity to earlier layers drops to +0.46 to +0.86, against +0.9 mid-stream). Each individual axis is also cross-layer stable (mean cosines +0.74 to +0.83 across L4 to L22). A per-pair stability measure is more mixed: several BALANCE-involving pairs are individually less stable than the null. The configuration-level result is the one that holds.

### Not every word cluster does this

We ran the same cross-layer matrix-preservation measure on other linguistic categories. Lakoff schemas (+0.91), quantifiers (+0.96), and determiners (+0.95) all preserve their inter-axis structure well above the null; logical operators sit at it (+0.78 vs +0.79). Not because the operator axes are unstable; individually they're among the most stable directions we measured. What logical operators lack is preserved *relational* structure. Schemas pattern with the grammatical core of the language, not with folk categories.

Two caveats. Generic word-difference matrices also preserve shape across layers, so this test works as a relative comparison between clusters, not as standalone proof of "coordinated system." And the null is the random-partition null built from the Lakoff anchors; we didn't build cluster-specific nulls. The predicted-vs-unpredicted coupling test (+0.21 vs +0.00) is the stronger evidence.

---

## Finding 3: Morphology on the schema axes, and the word2vec keystone

If the schema system is the model's organising geometry for meaning, grammatical operators should live in the same geometry. We built contrast vectors for seven English morphological operators (-ING, -ED, -S, -ER, -EST, un-, re-) as pair-differences ("walking" minus "walk", averaged over many pairs) and projected them onto the schema axes, after per-layer anisotropy and frequency stripping (raw pair-difference vectors run |cos| ≈ 0.55 to 0.59 with the anisotropy direction, which collapses everything into superficial similarity).

Two mappings are robust:

| Operator(s) | Schema | Stripped cosine across probed layers |
|---|---|---|
| all seven | BALANCE, negative (the shared "markedness sink") | −0.15 to −0.50 |
| -ED (past), re- (repetition) | FORWARD-BACK, negative | −0.12 to −0.40, every probed layer |

The FORWARD-BACK mappings are conceptually coherent on the theory's own terms: the past is behind you, repetition goes back along the path. And every operator shares the BALANCE-negative component, which we initially read as inflection-as-departure-from-equilibrium (Finding 4 revises what that component actually is).

Two prettier mappings from earlier drafts did not survive audit. -ING onto PATH-MOTION: loadings of −0.04 to +0.12, sign-flipping, dwarfed by -ING's own BALANCE loading. un- onto LIGHT-DARK: −0.06 to −0.14 in Pythia, while GloVe shows the same mapping at −0.39, three times stronger, so it is not transformer-specific. Both stories were lovely. Neither is supported.

![Clean suffix-schema heatmap](figures/exp138_suffix_schema_clean.png)

*Operator × schema projections after anisotropy and frequency stripping.*

### The keystone: static embeddings don't do this

We ran the identical protocol (same pairs, same axes, same stripping) on static embeddings: GloVe, word2vec, and fastText. If the schema-morphology mappings were simply inherited from textual distribution, static spaces should show them too.

They don't. The sharpest cell, ER×BALANCE, reads about −0.38 in Pythia at layers 8 through 20 (−0.47 at layer 4). GloVe: −0.01. word2vec: −0.005. fastText: +0.01. (fastText's subword architecture builds suffix coherence in, so its own results file classes it as supplementary; GloVe and word2vec carry the comparison.)

And this isn't because static embeddings lack morphology geometry. They have one; it's just different. The static spaces agree with each other on inflectional geometry (static-to-static r = 0.71 to 0.77) but correlate only weakly with the transformer's (r = 0.20 to 0.36). Meanwhile the *derivational* operators (un-, re-) do replicate across static and transformer spaces (r = 0.50 to 0.72): their positioning is distributional. It is specifically *inflection* that the transformer relocates onto the schema system.

![GloVe vs Pythia morphology](figures/exp150_glove_vs_pythia.png)

*Suffix-schema matrices in GloVe (left) vs Pythia L12 (right). The inflectional BALANCE sink (bottom row) is present in Pythia and absent in GloVe.*

In depth, the sink is partially written into Pythia's embedding matrix (−0.13 before any layers run), computed to full strength between layers 2 and 4 (−0.23 to −0.47), and plateaus by layer 8. Partly trained in, mostly computed. The dissociation replicates fully on Pythia 1.4B.

![Emergence of the inflectional sink across depth](figures/exp153_emergence.png)

*ER×BALANCE across depth: present at the embedding, computed to full strength by L4.*

### Caveats

The static-vs-transformer comparison is a dissociation between off-the-shelf spaces, not an architecture-isolating experiment: the substrates differ in corpus, objective, and tokenisation as well as architecture, and inflection is exactly where tokenisation differences could matter. A matched-corpus, matched-vocabulary baseline is the control this claim still owes. The derivational/inflectional split is post-hoc: found in the data, confirmed across substrates, not pre-registered. And the mappings are our extrapolation from the metaphor theory, not Lakoff's direct predictions. What the finding establishes as it stands: the transformer's inflectional geometry is anchored on the schema system, and the geometry static distributional learning produces is not.

---

## Finding 4: BALANCE is coupled to computational operations

The BALANCE schema is special. Every morphological operator shares a BALANCE-negative component. Why? What is BALANCE reading?

The answer, across three model families, is that BALANCE reads a computational operation. Which operation depends on the model.

### In Pythia: residual norm

Inflected forms ("walked", "running") have lower residual norms than their bases: per-suffix mean drops of roughly 240 to 850 units at layer 12. The per-pair correlation between norm-drop and BALANCE projection is r = +0.86 to +0.97 at layer 4, still +0.54 to +0.95 at layer 12. And the coupling is direct: cos(BALANCE, norm-direction) = +0.64 to +0.77 across the probed layers, using a held-out estimator whose estimation vocabulary excludes the suffix pairs and the BALANCE words. Other schemas lean on the norm carrier in a graded order (FORWARD-BACK +0.42 to +0.50, UP +0.28 to +0.44, LIGHT-DARK near zero); BALANCE leans hardest.

![Substrate primitives](figures/exp141_substrate_primitives.png)

*Per-suffix residual norm displacement (top) correlated with BALANCE projection (bottom) across layers. The coupling replicates across four Pythia model sizes.*

How much of Finding 3's sink is this physiological signal? Stripping the held-out norm directions collapses the sink entirely at layer 4 (−0.414 to +0.002) and removes about two-thirds of it at layers 8 to 20 (28 to 42% retained); matched random strips move it by ±0.001. So the sink is roughly two-thirds norm geometry, one-third a separable markedness component. (An earlier version of this analysis, with norm directions estimated on the very words being tested, collapsed the sink completely at every layer; a size-matched control exposed that as circular, and the held-out numbers replaced it.)

### In GPT-2 and Llama: attention entropy

Pythia and GPT-2 both use LayerNorm; Llama uses RMSNorm. If the BALANCE-norm coupling were about normalisation type, GPT-2 should share it. It doesn't: GPT-2 shows no stable BALANCE-norm coupling (per-layer values sign-flip from +0.50 to −0.29 around a mid-layer mean near zero) and no uniform inflectional norm displacement. Llama looks the same. The norm coupling is a Pythia-lineage trait, replicated across Pythia 70M through 1.4B.

But GPT-2 has something else. BALANCE is coupled to attention entropy, the diffuseness of the model's attention. The partial correlation, controlling for position, norm, and norm-projection, is −0.32 at L8, −0.27 at L12, −0.15 at L16, all bootstrap CIs excluding zero, BALANCE ranking 1st or 2nd of 8 schemas. Orthogonalising BALANCE against all seven other schemas makes the coupling slightly *stronger* (−0.34, −0.30, −0.17): it is BALANCE-unique, not shared schema variance. Mapped across depth, the coupling is negative at every layer from L0 to L17 and dies by L18. In Llama, the same coupling appears at layers 5 to 7 (−0.14 to −0.20, CIs below zero), confirmed on a third, never-before-used prompt set with checksum-verified frozen prompts.

| | Pythia (LayerNorm) | GPT-2 (LayerNorm) | Llama (RMSNorm) |
|---|---|---|---|
| BALANCE ↔ residual norm | +0.64 to +0.77 (held-out) | no stable coupling | no stable coupling |
| Inflectional Δ-norm | large, uniformly negative | no uniform displacement | no uniform displacement |
| BALANCE ↔ attention entropy | ~0 (null on fresh prompts) | −0.15 to −0.32 (L0 to L17) | −0.14 to −0.20 (L5 to 7) |

Three models, one concept, two different computational carriers. Read as embodiment, this is a candidate case of multiple realizability: each model organising the same concept around a signal intrinsic to its own processing. Demonstrating full multiple realizability would require showing both carriers serve the same function under matched causal interventions; what the table shows so far is the correlational half of that case, plus a causal result in GPT-2, below.

### Which way does the causation run?

Does the concept shape the computation, or the computation shape the concept? We tested both directions in GPT-2.

Concept → operation: no specific effect found. Steering on the BALANCE direction does not move attention entropy beyond what a matched-magnitude random push does; under all-layer injection, any big push scrambles attention regardless of direction. BALANCE is not a control knob for entropy.

Operation → concept: confirmed. Manipulating attention sharpness via temperature specifically shifts the BALANCE projection: at L3, slope −0.0083, CI [−0.0088, −0.0078], Spearman −1.00, with BALANCE moving more than all seven other schemas (the pre-registered specificity bar was four). Same result at L8.

So the causal evidence runs from operation to concept, and the reverse test found nothing. Two honesty notes on that asymmetry: the two interventions are not equivalent in power, so the null does not prove the absence of reverse causation; and temperature changes more than entropy alone, so "attention sharpness" rather than "entropy" is the precise thing shown to matter. Still, the direction we could confirm is the direction embodied cognition predicts: substrate state shaping concept geometry, the way the vestibular system shapes balance perception rather than the reverse.

### What didn't hold

The original hope was that entropy coupling would be universal across all three models. A depth-map scan seemed to show a surviving entropy band in Pythia at L11 to 12 even after quadratic norm controls. A third prompt set, pre-registered and checksum-verified, killed it: the band collapsed to −0.048 with a CI straddling zero. It was forking-path noise, the winner's curse of scanning 24 layers by eye. If we'd built the steering experiment on that band, the tempting move, we'd have steered noise.

The partial result is more interesting than universality would have been. Two of three models share the entropy carrier; one uses norm. The concept is conserved; the implementation varies by model. That is the shape multiple realizability would take, if the causal side fills in.

### Could this be multiple testing?

The BALANCE↔entropy coupling was originally found post-hoc, selected from an 8-schema × 3-model × 5-layer table after looking, and the pre-registration says so. So the question is fair. Five answers:

1. The confirmatory test was pre-registered with a specific hypothesis before any model ran (GPT-2, layers 8/12/16, negative sign, magnitude 0.10 to 0.30, fresh prompts). One pre-specified test, not a search. It passed.
2. Orthogonalising against the other seven schemas strengthens rather than attenuates the coupling. "One of eight random directions came up lucky" predicts the opposite.
3. At the decision layers, BALANCE is the clear standout; the runner-up (DIFFICULTY) is at 60 to 70% of the magnitude with *opposite sign*, and the other six sit at |r| ≤ 0.16.
4. The GPT-2 coupling is negative with CIs excluding zero at 18 contiguous layers. Multiple testing produces scattered hits, not contiguous bands.
5. Triple replication on independent prompt sets: L8 r = −0.21, −0.32, −0.27 across discovery, pre-registered confirmation, and a checksum-verified third set (L12: −0.26, −0.27, −0.32). The discovery value was the weakest, the opposite of a winner's-curse profile.

And the same machinery that confirmed GPT-2 killed Pythia's entropy band, which really was a multiple-testing artifact. The system discriminates.

Residuals: Llama's layer band was informed by the depth-map scan, so it carries a mild winner's-curse concern that GPT-2 does not; and no formal multiple-comparison correction was applied across the eight schemas (the project relied on pre-registration and replication instead). Weight the GPT-2 result; treat Llama as supporting.

---

## What this adds up to

Schema-like directions are recoverable from bare single words, and they form the specific relational system the theory predicts (Finding 2). One of them behaves causally, executing a cross-domain metaphorical mapping (Finding 1). The system extends into inflectional grammar in a way static distributional learning doesn't reproduce (Finding 3). And one schema is coupled to the model's own computational dynamics, with causal evidence running from operation to concept, and different models using different carriers (Finding 4).

### What this changes

The inference this challenges is: no body, therefore no image schemas, therefore structurally defective meaning. That inference treats three different things as one. *Bodily origin*: where the organisation comes from in humans. *Bodily implementation*: what it runs on. *Embodied-style structure*: the organisation itself, the coordinated schema system with its metaphorical mappings and grammatical reach. The results show the third can exist without the first two, and that parts of it end up anchored on whatever substrate the system actually has.

This is not a refutation of embodied cognition, and it is certainly not a claim that Lakoff is wrong about humans. If anything it extends the theory into strange territory: the structural principles the theory attributes to bodily experience turn out to be partly transmissible through language, and partly reorganised around a new substrate's own dynamics. Whether norm and entropy deserve to be called a "body" is an interpretation the experiments motivate, not one they settle. The distinction they do force is the one above, and it matters beyond this paper: arguments that LLM meaning is defective *because* disembodied need a premise these results take away.

The claim throughout is structural, not a claim about experience. Two open edges: the mechanism is unknown (every mechanistic hypothesis we tested died, 0 for 8), and the causal picture is one confirmed direction, not a settled story.

The question this opens is the one I find genuinely exciting: perhaps what matters is not whether a system has a body in the biological sense, but whether its conceptual organisation is shaped by stable constraints intrinsic to its own substrate, and what those constraints are. Each model in this study, asked to represent balance, seems to have reached for the nearest thing it has to a sense of balance. What else works that way?

---

## Related work

The philosophical background is the embodied-cognition tradition, principally Lakoff & Johnson's *Metaphors We Live By* (1980) and *Philosophy in the Flesh* (1999), and Johnson's *The Body in the Mind* (1987), which develop image schemas as the bodily source of abstract conceptual structure. The best-known modern "no grounding from form alone" argument is Bender & Koller's "Climbing towards NLU" (ACL 2020); their target is communicative intention and world reference rather than internal conceptual structure, so these results pressure the disembodiment inference specifically, not the octopus argument as formulated.

A growing literature shows text-only models recovering structure that mirrors perceptual spaces: Abdou et al. (CoNLL 2021) on colour, Patel & Pavlick (ICLR 2022) on grounded conceptual spaces, Gurnee & Tegmark (ICLR 2024) on space and time. This project asks a different question: not whether the geometry mirrors a perceptual domain, but whether the organisational architecture attributed to the body is present: a coordinated schema system, causally transferable cross-domain mappings, extension into grammar, and coupling to substrate-intrinsic operations. Findings 4 and 5, to my knowledge, have no precedent in that literature.

Methodologically the work builds on activation steering (Turner et al.'s ActAdd, arXiv:2308.10248; Zou et al. 2023) and the linear-representation tradition (Mikolov et al. 2013; Park et al., ICML 2024). Two known geometric hazards shaped the pipeline throughout: anisotropy of contextual spaces (Ethayarajh, EMNLP 2019) and massive activations (Sun et al., COLM 2024).

---

## Open questions

1. **Carrier-first search.** Start from each computational operation (norm, attention entropy, attention pattern, MLP activation) and ask which schema couples to it, inverting the search and its confirmation biases.
2. **More models.** Does the norm-vs-entropy split track normaliser type, family, or training data? A Mamba model, with no attention at all, is the sharpest test: does entropy-coupling disappear, or does the model find a third carrier?
3. **FORWARD-BACK, norm, and past tense.** FORWARD-BACK is the #2 schema on Pythia's norm carrier, and -ED maps to FORWARD-BACK negative. Are the suffix geometry and the norm physiology one story?
4. **Nonlinear structure.** The schema system shows significant nonlinear pairwise dependence that linear methods are blind to. It may be a manifold, not a flat vector space.
5. **Ontogeny.** At training step 512, BALANCE is anti-coupled to norm (cos −0.61); by step 4,000 it has flipped to +0.64, where it freezes for the rest of training. The sign inversion is a window into the moment a model acquires its substrate coupling.
6. **Structure replicates; mechanisms don't.** Structural findings replicate across prompts, sizes, and architectures; every mechanistic upgrade failed. That meta-pattern is itself a finding about the current limits of interpretability.

---

## Methods note

Experiments ran on Pythia 410M (primary) with replications on Pythia 70M, 160M, and 1.4B, GPT-2 medium, and Llama-3.2-1B; static comparisons used GloVe, word2vec (Google News, 300d), and fastText. Contrast vectors were built from single-word residual activations, morphology vectors from inflected-minus-base pair differences, all anisotropy- and frequency-stripped per layer. The BALANCE-norm coupling used held-out norm-direction estimation to avoid circularity. The entropy analyses used a quadratic control stack validated on synthetic data (a planted curved leak was killed, −0.66 to −0.04, while a genuine coupling was spared, −0.63 to −0.64). Confirmatory experiments were pre-registered with committed point predictions before code execution, and frozen prompt sets were checksum-asserted at runtime.

The standing rule of the project: when a finding seems big, name the control that would falsify it, then run it. The accompanying repository contains every experiment script, raw output, pre-registration, and lab-notebook volume, including the failures.

The research was conducted as a collaboration between the author and Claude (Anthropic) across many sessions, with continuity maintained through the lab notebooks and per-session handoff documents.

---

*Niamh McAllister, 2026. Lab notebook and experiment code at the accompanying repository.*
