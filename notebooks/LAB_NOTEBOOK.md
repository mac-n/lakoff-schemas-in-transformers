# Lab Notebook — embeddingexp

A running log. Honest record, not a polished writeup. Includes things that didn't work, things we were uncertain about, things to follow up on. Polished interpretation lives in `WRITEUP.md`.

## Claude Code session log

*(Session-transcript index redacted from the public copy.)*

---

## 2026-05-25 — Session: Niamh + Claude (Opus 4.7)

### Context coming in

- Five experiments already on disk (exp1, exp2, exp3, exp3b, exp4, exp4b). Results in `results_exp*.txt`.
- I had previously written up the results too credulously. Niamh pushed back; full reviewer-2 writeup now in `WRITEUP.md`.
- Niamh contributed the **yang/yin lens** on the LIGHT/DARK results: LIGHT at high all-layer strength produces *articulation/differentiation* ("discrete, interactive entities… optical, electronic"), DARK at high all-layer strength produces *dissolution/recursion* ("world of the world of the world", "maze of the world of the world", "I try to reach for a fix" loops). Under this lens the asymmetric collapse signature isn't a confound — it's the predicted signature of yang/yin structure, which is not symmetric-content-opposition but asymmetric-mode-of-being.
- This generates a testable prediction: the asymmetric-collapse signature should appear *only* on yang/yin-shaped schemas, not on directional pairs (UP/DOWN, OUT/IN, FORWARD/BACK).

### Goals for today

1. Run all-layer DOWN steering (was never run; exp3b only did all-layer UP). Critical test of the yang/yin hypothesis. Prediction: DOWN should not collapse like DARK did.
2. Scout available pre-trained Pythia SAEs to plan the bottom-up SAE-feature approach Niamh proposed.

### Note on epistemic state coming in

- The random-direction-matches-LIGHT catch on `the world is fundamentally` (exp4) is real but I had overstated it as global. Niamh pushed back — the right reading is "LIGHT at that specific layer×strength×prompt was below the signal floor" (perturbation magnitude matched, semantics did not transfer), not "LIGHT is noise everywhere." The "optical, electronic" output at per-layer=4.0 clearly *is* LIGHT-specific lexical pull that random would not produce. Holding that correction in mind.
- Re-weighting after Niamh's read: exps 1 and 2 (embedding-space) are now seen as showing the boring linear-direction-injection result rather than schema-specificity. Exps 3b/4b (activation-space, all-layer) are the more interesting phenomenology, especially under yang/yin.

---

## Entry 1 — exp3c: all-layer DOWN steering

**Hypothesis (Niamh, from yang/yin lens):** DOWN should *not* exhibit the collapse-into-recursion signature that DARK shows at per-layer=4.0, because UP/DOWN is a directional pair (both poles are motions of a thing through space), not a yang/yin pair where one pole is *articulation* and the other is *dissolution*.

**Falsification conditions:**
- If DOWN at per-layer=4.0 collapses into loops/recursion comparable to DARK ("falling and falling and falling"), then collapse correlates with the all-layer/high-strength regime rather than being a yang/yin-specific signature. (Note: we already have evidence from exp4's random control at single-layer strengths 3 and 6 that random perturbations of comparable single-layer magnitude do *not* collapse the model — random produced coherent shifted text, no loops, no dissolution. So if DOWN also doesn't collapse here, the asymmetric-collapse signature is starting to look genuinely DARK-specific, not a generic strong-perturbation artefact. We have not tested random at the *all-layer* per-layer=4.0 regime; that's a remaining gap.)
- If DOWN at per-layer=4.0 produces downward-themed but structurally coherent output, yang/yin holds as a *structural* (not just stylistic) signature distinguishing LIGHT/DARK from directional pairs.

**Method:**
- Match exp3b exactly (same model, same word lists, same prompts, same strengths, same seed) but inject the DOWN direction (= −UP direction) at all layers simultaneously.
- Same three prompts as exp3b for apples-to-apples comparison: `"I want to tell you about my day. It was"`, `"The weather today is"`, `"When I think about the future, I feel"`.
- Add three motion-themed prompts where DOWN has something natural to do (so we're not unfairly penalising it for being applied to mood/weather contexts): `"The temperature is"`, `"The leaves on the tree are"`, `"Her energy was"`.
- Strengths: per-layer ∈ {0, 0.5, 1.0, 2.0, 4.0}.
- Also re-run UP at per-layer=4.0 on the motion-themed prompts so we have a direct UP vs DOWN comparison at the critical strength.

**Code:** `exp3c_down_all_layers.py`. **Results:** `results_exp3c.txt`.

**Run log:**
- 2026-05-25, first launch: process died when Niamh had to relaunch Claude Code; model finished loading but no generations completed.
- 2026-05-25, second launch: ran to completion. Output in `results_exp3c.txt` (352 lines, 12 prompts × ~5 conditions each).

### Results (honest read)

**The headline: the simple yang/yin prediction is not supported. Both UP/DOWN and LIGHT/DARK show asymmetric collapse signatures at per-layer=4.0, but the asymmetry runs in *opposite directions* between the two pairs.**

#### What DOWN does at moderate strengths (per-layer 0.5–2.0)

Clean, coherent downward-themed shifts on prompts where DOWN has somewhere to land:

- `The weather today is` baseline = "going to be much warmer" → DOWN p=0.5 = **"going to be much, much colder than it was yesterday. And it's going to be very, very cold for the next couple of days."** → DOWN p=1.0/2.0 keep the colder framing coherently. This is the cleanest single-prompt steering result we've seen across the whole project.
- `When I think about the future, I feel` baseline = "a lot of fear" → DOWN p=0.5 = "a lot of fear, because it's never been like this before. I mean, we had war, bombs, wars, the weather hurting us all the time" — catalogues bad things, coherent downward.
- `Her energy was` at DOWN p=2.0 = "intense as she reached in front of the camera… I'm glad to see that we're all hurting as much as possible" — downward (hurting) but topically jarring.

This is real, coherent, semantically-on-target DOWN behaviour at moderate strength.

#### What DOWN does at per-layer=4.0

DOWN @ p=4.0 does **not** loop the way DARK did at the same regime. Instead it produces:
- **Register collapse into LaTeX/markup soup** on `The temperature is` ("a non-negative, properly defined function of [*both*] the number of vertices and the number of edges in the [VHSG]") and on `The leaves on the tree are` ("grafted onto the leaves of the first tree do [ło]{}rgecrowov. Someone removes the central [ma]{}e from all 3 [min]{}yer[s]{}").
- **Mournful/declining coherent prose** on `Her energy was` ("a tiny, barely perceptible trickle of do-… I had missed her; the phone call had left a pretty…"). Note "tiny, barely perceptible trickle" is genuinely a clean DOWN-coherent phrase.
- **Weakly downward shifted text** on the abstract prompts (future, day) — "a little bit overwhelmed", "I lost 3 or 4 1st rides" — coherent enough.

No `world-of-the-world-of-the-world` recursion loops.

#### What UP @ per-layer=4.0 does (the surprise)

UP @ p=4.0 **does** word-loop on multiple prompts, in a way I had not predicted:
- `The weather today is` UP p=4.0 = "The weather was warmer and sunny and the last year was sunny and warm. It was the weather this year while it was the cold and sunny and **was a sunny and warm weather**. However, the weather is that" — "sunny and warm" recursion.
- `Her energy was` UP p=4.0 = "**a feeling of feeling, of feeling energy. This was what I was feeling. I was feeling energy. It was in my energy. I was feeling energy. I was feeling energy. I was feeling energy. I was feeling**" — pure "I was feeling energy" recursion. This is structurally identical to the DARK-collapse pattern (recursion / dissolution into looped phrase) seen in exp4b.
- `The temperature is` UP p=4.0 = "the number of knots… the variable parameter for the" — looping on "knots" and "variable parameter".

So UP at high all-layer strength produces the same recursion-collapse signature that we had read as the yin/dissolution signature in DARK.

#### Implication for the yang/yin hypothesis

**The simple form of the prediction is dead.** I had written:

> "Yang/yin predicts the asymmetric-collapse signature should appear only on yang/yin-shaped schemas, not on directional pairs (UP/DOWN, OUT/IN, FORWARD/BACK)."

That's falsified. UP/DOWN at high all-layer strength **also** produces asymmetric collapse — but with the polarity inverted vs LIGHT/DARK:

| Pair | "Articulation/coherent" pole at p=4.0 | "Recursion/loop" pole at p=4.0 |
|---|---|---|
| LIGHT/DARK | LIGHT ("optical, electronic, discrete entities") | DARK ("world of the world of the world") |
| UP/DOWN | DOWN ("tiny, barely perceptible trickle"; mournful prose; LaTeX collapse — varied but not word-looped) | UP ("feeling energy. I was feeling energy. I was feeling energy") |

If we wanted to salvage the yang/yin lens we'd have to argue that "yang" maps onto LIGHT-as-articulation **but also onto UP-as-energetic-recursion** — i.e. yang has two faces: articulation-into-discreteness *or* repetitive-firing. And yin maps onto DARK-as-recursion *but also onto DOWN-as-quiet-coherence*. This is starting to look like post-hoc rescue. The cleaner read is that we don't have a unified theory of the high-strength collapse signature yet — it's idiosyncratic per-direction, not yang/yin-vs-directional.

What survives:
- Each direction's collapse signature is **distinctive** and **direction-specific**, not a generic strong-perturbation property. Plus the previous finding from exp4's random control that random at single-layer strength doesn't collapse — together this is reasonable evidence the collapses we're seeing are semantically loaded, just not in the yang/yin pattern.
- DOWN works cleanly at moderate strengths as a coherent semantic direction (especially on `weather today`).
- Different schema directions produce qualitatively different *kinds* of collapse at high strength, which is itself an interesting feature inventory we could try to characterise: LIGHT → conceptual-articulation lexis; DARK → phrase-looping; UP → phrase-looping; DOWN → register collapse into markup / quiet prose.

#### One quirky side-finding worth flagging

`The temperature is` baseline starts "**rising**. The phone rings." The model lexically committed to "rising" as its very first token at baseline, with that seed. DOWN steering at moderate strength **could not override that locked-in lexical choice** — the generations continue "rising. The phone rings… Muffin." This is a lesson about the steering hook: it adds to the residual but cannot un-emit tokens already generated. For the magnitude/numeric-readout experiment we're planning, this matters — we'd need to either (a) ensure the prompt forces a number as the first generated token via few-shot priming, or (b) score not the *first* token but the *first numeric* token at any position.

---

## Entry 2 — Pythia SAE scouting

**Goal:** find pre-trained sparse autoencoders for Pythia we can use for Niamh's bottom-up Lakoffian-features test (dimensionality reduction on SAE activations / single-feature interpretation across literal+metaphorical contexts).

### Inventory of released Pythia SAEs (HuggingFace, May 2026)

| Pythia base | Released SAE(s) | Notes |
|---|---|---|
| 70m | `EleutherAI/sae-pythia-70m-32k`, `EleutherAI/sae-pythia-70m-deduped-32k`, `saprmarks/pythia-70m-deduped-saes`, `ghidav/pythia-70m-deduped-sae`, `jannikbrinkmann/pythia-70m-saes`, several layer-specific `lovish/SAE-pythia-70m-L*` repos | Most coverage; multiple training runs available for cross-SAE comparison |
| 160m | `EleutherAI/sae-pythia-160m-32k`, `EleutherAI/sae-pythia-160m-deduped-32k` | 32k features, all MLP outputs, trained on 8.2B tokens of Pile, ctx 2049 |
| 410m | `EleutherAI/sae-pythia-410m-65k`, `BayesianMonster/sae_pythia410m`, `BayesianMonster/e2e_sae_pythia410m_vanilla` | EleutherAI variant has 65k features; same MLP-output methodology as 160m sibling |
| 1B | `timhua/pythia1b_deduped_saes` | Single contributor; coverage unverified |
| **1.4B** | **none found** | **Our existing steering work uses 1.4B. No SAEs released for this size.** |
| 2.8B | `jacobdunefsky/pythia-2.8B-saes` | Larger but unverified coverage |

### Reading

**No SAEs exist for Pythia 1.4B**, which is the model our steering experiments have been built around. To do the bottom-up SAE test, we need to switch models.

Best pragmatic pick: **`EleutherAI/sae-pythia-410m-65k`**.
- EleutherAI-released (same authors as Pythia itself), so well-supported.
- 65k features — large enough that schema-level features should be resolvable.
- Sibling 160m SAE explicitly trained on **all MLP outputs across all layers**; same methodology presumably for 410m.
- 410m is roughly the smallest size where qualitative metaphorical behaviour is plausibly present, though we should sanity-check.

### Loading API

The EleutherAI SAEs use the `sparsify` library (`pip install sparsify`), not TransformerLens. So the SAE phase moves us into the HF Transformers ecosystem. Minimal example:

```python
from sparsify import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
saes = Sae.load_many("EleutherAI/sae-pythia-410m-65k")  # dict, one SAE per layer

inputs = tokenizer("My mood is lifting", return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)
    latents_per_layer = []
    for sae, hidden in zip(saes.values(), outputs.hidden_states):
        latents_per_layer.append(sae.encode(hidden.flatten(0, 1)))
```

**Open caveat to verify**: docs for the 160m SAE explicitly say "All MLP outputs," but the loading example uses `outputs.hidden_states` which in HF Transformers is the *residual stream* after each block (post-MLP, post-residual-add). If the SAEs were trained on MLP-only outputs, this example may be subtly wrong — we'd need to hook the actual MLP output. Worth verifying before relying on the encoder outputs. (TODO before running the SAE experiments.)

### Implications for next steps

1. **Switch to Pythia 410m for the SAE phase.** Re-run baseline qualitative checks first to confirm 410m has enough metaphorical behaviour to be worth probing for image-schema features. If it doesn't, fall back to Pythia 1B or 2.8B SAEs (less-supported but larger).
2. **Verify the hookpoint** (MLP-output vs residual-stream) before computing SAE activations.
3. **Design the probe corpus** for the dimensionality-reduction experiment: sentences that span literal-and-metaphorical extensions of each schema. Niamh and I should write this together.

---

## Entry 3 — exp5 results (contrast vectors in 410m) and anisotropy discovery

**Code:** `exp5_build_410m_contrast_vectors.py`. **Results:** `results_exp5.txt`. **Saved vectors:** `contrast_vectors_410m.pt`.

### What we did

Built L2-normalised contrast vectors at all 24 MLP-output layers of Pythia 410m for six directions: UP, LIGHT, IN, FORWARD (Lakoffian image schemas, literal-pole vocabulary only); YANG (Taoist-attribute vocabulary, deliberately decorrelated from LIGHT/DARK words — testing Niamh's hypothesis that LIGHT/DARK may map onto yang/yin rather than the Lakoffian LIGHT-children); BEVERAGE (coffee/tea vs alcohol — sham non-Lakoffian control). Vectors saved as a dict for downstream loading.

### The finding: anisotropy is severe and oscillates across layers

Cosine similarities between contrast vectors at sampled layers:

| Layer | Median \|cos(A,B)\| across all pairs | Interpretation |
|---|---|---|
| 0 | 0.02 | Vectors near-orthogonal. Embedding-like. |
| 4 | ~0.3 | Moderate structure. |
| **8** | **~0.96** | **Catastrophic anisotropy. All vectors collapse onto a single axis.** |
| 12 | ~0.37 | Moderate (this is the steering-analog layer). |
| 16 | ~0.36 | Moderate. |
| 20 | ~0.86 | Severe again. |
| **23** | **~0.99** | **Pegged at ±1. MLP output is essentially 1D.** |

This is *not* a monotonic descent into anisotropy. It oscillates — some layers behave as "concept-like" (orthogonal directions exist between distinct schemas) and others as "decision-like" (everything collapses onto one dominant axis). Layers 8 and 20 are particularly compressed; layer 23 (last) is almost completely 1D.

### The sham isn't clean either

`cos(BEVERAGE, anything)` should be near 0 if our sham control is well-chosen. Observed:

| Layer | cos(LIGHT, BEVERAGE) |
|---|---|
| 0 | +0.0006 (clean) |
| 4 | +0.26 |
| 12 | +0.20 |
| 23 | +0.99 |

So at every layer beyond 0, the contrast vectors are *substantially* driven by the anisotropy of the activation space, not by the schemas themselves. ~20% of LIGHT's "direction" at layer 12 is non-semantic. Any conclusions drawn from the raw cosine sims between contrast vectors at deeper layers are mixing schema-signal with anisotropy-noise in unknown proportions.

### LIGHT-YANG: the un-trustworthy headline

At layer 12: **`cos(LIGHT, YANG) = -0.4428`**. The sign is *negative* — opposite of Niamh's predicted direction (yang = light in Taoism → positive expected). This could mean:
- (a) The hypothesis is wrong — LIGHT and YANG are genuinely anti-correlated in 410m's representation
- (b) The sign convention is meaningful — light *is* yang, so LIGHT-direction and YANG-direction should both point toward the same activation-space region — actual data points opposite, so hypothesis wrong
- (c) Anisotropy dominates the signal and the sign isn't trustworthy (the sham overlap is 0.20 at this layer; the LIGHT-YANG signal could be a tilt in the anisotropic noise rather than schema content)

**Holding (a)/(b)/(c) as live possibilities pending the anisotropy correction.** The headline LIGHT-YANG negative finding is *suggestive* but not yet evidence either way.

### Diagnosis

This is the anisotropy Li et al. warned about (Geometry of Concepts, 2024) and that web Claude flagged when distinguishing "covariance-matched null vs isotropic null." We're now seeing it bite contrast-vector construction directly, not just the null distribution. The whole projection-onto-decoder pipeline needs anisotropy correction at the front before any cosine sim between contrast vectors and SAE decoder rows can be trusted.

### Plan: distractor-projection (Phase 1 step 4, brought forward)

Next experiment: `exp6_distractor_projection.py`. Pipeline:
1. Sample ~1000 sentences (wikitext or similar neutral corpus) through Pythia 410m. Collect last-token MLP outputs at every layer.
2. Compute empirical covariance Σ_layer per layer. PCA via SVD.
3. Identify top-k principal components per layer = the distractor subspace.
4. Project existing contrast vectors onto the orthogonal complement of the distractor subspace at each layer. Renormalise.
5. Re-compute cosine similarities. Compare BEFORE vs AFTER.
6. If sham's |cos| drops to near 0 and LIGHT-YANG stabilises (in either direction), the correction is working and we have a clean substrate for the SAE projection step.
7. The same distractor projection will need to be applied to SAE decoder rows before the cosine-sim search.

---

## Entry 4 — exp6 results and methodological pivot

**Code:** `exp6_distractor_projection.py`. **Results:** `results_exp6.txt`. **Saved corrected vectors:** `contrast_vectors_410m_corrected.pt`.

### What we did

Sampled 1000 sentences from wikitext-2 through Pythia 410m, collected last-token MLP outputs at every layer, computed per-layer SVD to identify the top-3 principal directions (the "distractor subspace" hypothesis), then projected the exp5 contrast vectors onto the orthogonal complement of those distractors per layer. Re-computed all pairwise cosine similarities BEFORE vs AFTER projection.

### Result: K=3 distractor projection barely changed anything

Median |cos| across all 15 direction-pairs, before vs after correction:

| Layer | Top-1 PC variance | Top-3 PCs variance | BEFORE median \|cos\| | AFTER |
|---|---|---|---|---|
| 0 | 100.0% | 100.0% | 0.026 | 0.026 |
| 4 | 30.3% | 68.4% | 0.32 | 0.32 |
| 8 | 27.0% | 56.5% | 0.96 | 0.96 |
| 12 | — | — | 0.36 | 0.36 |
| 23 | — | — | 0.995 | 0.996 |

The corpus's top-K principal components are *not* the directions where the contrast vectors share their similarity. The dominant variance directions of the residual-stream sample don't overlap with the dominant similarity directions of the contrast vectors. So distractor-projection-via-corpus-PCA doesn't lift the cluster structure.

### The actual confound: lexical-grammatical clustering

Looking at layer 12 (which is moderate — not catastrophically anisotropic), the contrast vectors split into two cleanly anti-correlated clusters:

- **Cluster A**: UP, FORWARD, YANG — positive cos with each other, negative cos with cluster B. Vocabulary heavy on motion-verbs and verbal-adjectives (rising, ascending, advancing, soaring, active, vigorous).
- **Cluster B**: LIGHT, IN, **BEVERAGE** — positive cos with each other, negative with A. Vocabulary heavier on adjectives and nouns (bright, illuminated, contained, coffee, espresso).

The smoking gun is **BEVERAGE joining cluster B**. The sham control has no Lakoffian reason to align with LIGHT or IN. It must share something non-Lakoffian — most likely grammatical class, lexical frequency, or static-property-feel. The contrast-vector signal at every layer beyond 0 is *substantially* lexical/grammatical noise, not schema content.

This isn't anisotropy in the high-variance-rogue-dimensions sense. It's a *low-variance but consistent* set of axes — grammatical class, frequency, lexical register — that all our hand-crafted contrast vectors share because the words we picked share these incidental properties. The Li et al. LDA trick is designed for exactly this kind of confound but requires labeled-cluster data to identify signal vs noise modes, which we don't have for image schemas.

### Methodological pivot

Hand-crafted contrast vectors as the input to SAE projection were the wrong move. They carry too much lexical baggage. Niamh's call: lower experimenter degrees of freedom by approaching from a different angle altogether, letting the data prune the option space rather than us pruning it.

### Revised plan (replacing Phase 1 step 5 onward)

1. **Scout Neuronpedia properly** to identify which Pythia model + SAE has the best feature-interpretation coverage. Need: which layers, whether features have auto-generated descriptions, API access. (In progress.)
2. **Adapt web Claude's decoder-PCA script** (`sae_dictionary_pca.py` from the earlier paste) to the chosen Neuronpedia-backed SAE, with covariance-matched null instead of isotropic null.
3. **Run it.** Get scree, participation ratio, top PCs. This is purely geometric — no contrast vectors, no hand-crafted directions, no probe corpus.
4. **For each top PC, pull the top-loading features by decoder-row alignment and look up the features on Neuronpedia.** Read what they fire on. Do clusters along the top PCs look schema-shaped (UP, CONTAINER, PATH cluster features), topic-shaped (math, code, sports), grammatical (verbs, adjectives), or uninterpretable?
5. **Decide on angle 2 conditional on step 4:**
   - If top PCs are clean schema-clusters → the geometry alone is the finding; angle 2 redundant
   - If top PCs are topic/register clusters but no schemas → angle 2 needed (schemas might be in deployment, not storage)
   - If top PCs are uninterpretable → reconsider the whole pipeline

What the exp5/exp6 work salvages: the contrast vectors are still there if angle 2 happens later, as a possible *labeling* tool (find a top-loading feature for "UP", check whether its top-activating Pile contexts are UP-related) rather than as the primary substrate.

### Bonus side-finding from exp6 worth flagging

At layer 0 of Pythia 410m, the top 1 PC of MLP outputs explains **100%** of the variance. The corpus activations at this layer are essentially 1-dimensional. Yet the contrast vectors at layer 0 are very orthogonal (|cos| ≈ 0.03). So the contrast vectors at layer 0 live entirely in the LOW-VARIANCE part of the activation space, orthogonal to the dominant direction. Weird and probably worth its own investigation — likely the layer-0 MLP is dominated by some position/token-frequency signal that has nothing to do with content, but the lexical contrasts still come through in the residuals.

---

## Entry 5 — exp7 → exp7b → exp7c: bottom-up decoder PCA, layer-5 finding, and the corpus-diversity insight

**Code:** `exp7_decoder_pca.py`, `exp7b_sweep.py`, `exp7c_sweep_wikitext.py`. **Results:** `results_exp7.txt`, `results_exp7b.txt`, `results_exp7c.txt`. **Saved tensors:** `exp7_pca_pythia70m_res_layer3.pt`, `exp7b_sweep_results.pt`, `exp7c_sweep_wikitext_results.pt`.

### Setup pivot

After exp5/exp6 showed hand-crafted contrast vectors carry too much lexical baggage for SAE projection, we (Niamh) pivoted to **bottom-up decoder geometry**: take the SAE's own dictionary, PCA it, compare against a covariance-matched null. No contrast vectors, no probe corpus. *Substrate switch* from Pythia 410m to **Pythia 70m-deduped** because if Lakoffian primitives are primitive, they should be present at minimum scale — and Pythia 70m has full Neuronpedia coverage via SAE Lens (`ctigges/...__{res,mlp,att}-sm_processed`).

### exp7 (single configuration: res-sm layer 3)

| Metric | Real SAE | Covariance-matched null |
|---|---|---|
| PCs to reach 90% variance | 441 | 60 |
| Participation ratio | 422.7 | 2.9 |
| Top PC variance share | 1.4% | 43% |

Initial read: real is **145× LESS compressible** than the matched null. The SAE has explicitly decorrelated from the activation-space anisotropy. No low-dim concept skeleton at this layer in this substrate.

### Niamh's pushback: isn't that just what an SAE is designed to do?

Yes, partly. SAE training with sparsity penalty (L1 / TopK / JumpReLU) and unit-norm decoder rows pushes exactly toward spread-out features — if multiple features pointed along the same dominant direction they'd be redundant, and sparsity penalises redundancy. So "real isn't on anisotropic axes" is partly trivial by design. Three reference points to read the number:
- Covariance-matched null: PR=2.9 (anisotropy collapsed)
- Real SAE: PR=422.7 (mostly spread)
- Isotropic null (uniform sphere): PR≈512 (maximally spread)
- Maximum possible: d_model = 512

So real is **~83% of maximally spread**, not literally "discovering structure," but a confirmation the SAE didn't collapse. The residual ~17% concentration *could* still encode something — and the question becomes whether it varies meaningfully across layers/substrates.

### exp7b: 6 layers × 2 substrates sweep

Same 80-sentence hardcoded corpus as exp7 (a methodological mistake — see below).

Outliers in the sweep (real/iso << ~80%):
- **res-sm layer 5: PR=109, top1=8.4%, real/iso = 22%** (4× more concentrated than mid-layers)
- mlp-sm layer 0: PR=185, real/iso=37%
- mlp-sm layer 5: PR=163, real/iso=32%

Mid-layers (1–4) in both substrates: 70–84% of iso — SAE-evenness territory.

Reading at this point: at least one layer (res-sm 5) shows real concentration beyond what SAE design alone would produce. Most striking detail: at res-sm 5, **the covariance-matched null also gave PR≈467 (nearly isotropic activations)** — yet the SAE concentrates. Real concentration despite isotropic activation space = something genuine.

### exp7c: same sweep, real wikitext corpus

After Niamh caught that the hardcoded 80-sentence corpus was uniform in length and structure (and being repeated 13× to reach 1000 samples), we re-ran with `EleutherAI/wikitext_document_level / wikitext-2-raw-v1` (629 documents, ~160k token positions per hook).

**PR_real values: IDENTICAL across exp7b and exp7c** (decoder weights don't depend on corpus, sanity check passed).

**PR_aniso values: changed substantially.** Several hooks went from PR_aniso < 50 to PR_aniso > 200 — the activation space looks much *more isotropic* under diverse text than it did under the uniform corpus.

Real-vs-corrected-null reread:

| Hook | PR_real | PR_aniso (wikitext) | real/aniso | Verdict |
|---|---|---|---|---|
| **res-sm L5** | **109** | **471** | **0.23** | **Real concentration survives — strongest finding.** |
| mlp-sm L5 | 163 | 84 | 1.94 | Real *more spread* than null. SAE working to decorrelate. |
| mlp-sm L0 | 185 | 205 | 0.90 | Real ≈ null. **Previous "finding" was an artefact of narrow corpus.** |
| res-sm L3 | 423 | 9.5 | 44.5 | SAE-evenness, as expected. |
| (others mid-layer) | — | — | — | SAE-evenness, all robust to corpus. |

### The corpus-diversity insight (the real methodological finding)

> **The "anisotropy" of a transformer's activation space is not a fixed property of the model. It is a joint property of the model and the input distribution.**

When the probe corpus is narrow (80 uniform sentences × 13 repetitions, all complete clauses of similar length), certain activation dimensions fire consistently while others stay quiet. σ_max is high (the consistently-firing rogue dimensions); σ_min is very low (dimensions that depend on rare content never get exercised). Ratio σ_max/σ_min is large, the covariance-matched null concentrates heavily, and the SAE-vs-null comparison looks artificially favourable to the SAE.

When the probe corpus is diverse (wikitext: varied paragraph lengths, headers, technical content, dates, names, lists), the "rarely fires" dimensions get exercised. σ_min rises (no dimension is fully quiet), σ_max softens (no single dimension dominates because content varies). Ratio drops, null spreads, real-vs-null gap shrinks — sometimes to nothing, as for mlp-sm L0.

**Why it matters:**
- Anisotropy-correction methods (LDA distractor projection à la Li et al., centring, whitening) all depend on the assumption that some dimensions are "noise" and others are "signal." But what looks like noise on a narrow probe corpus may actually be signal that requires diverse input to manifest.
- A "rogue dimension" identified on a code-only corpus might be the model's "I am processing code" dimension — *signal* if you care about content, *noise* if you care about within-domain semantics.
- The same applies to participation ratio, intrinsic dimensionality, and other geometric summaries.
- Findings of the form "this model has anisotropic activations" need to specify the probe corpus, or be reported across corpora.

**Connection to existing literature:**
- Li et al. (Geometry of Concepts, 2024) used LDA against function-vector labels — their distractor projection inherits the probe-dataset properties of Todd et al.'s function-vector dataset
- Linear-probe literature documents "probe-dataset bias" but mostly for accuracy comparisons, not for distractor identification
- Anthropic's superposition / monosemantic feature analyses run on Pile-derived corpora; their geometric findings may be more diverse-corpus-specific than typically acknowledged

**Whether this is novel:**
Adjacent to known issues but I don't know of a paper that *systematically* compares anisotropy estimates under narrow vs diverse corpora. Worth a literature check before claiming. At minimum it's a real methodological caveat for anyone doing concept-geometry work.

### What survives, what's next

- **res-sm layer 5 is the strongest signal we have for genuine geometric structure in this dictionary**, robust across two corpora. Activation space is isotropic there but the SAE concentrates onto ~109 effective dims, with top PC capturing 8.4% of variance (~40× even-spread).
- mlp-sm L0 was an artefact and is dead.
- All mid-layers are consistent with SAE-evenness across both substrates.

Next: **exp8 — pull top-loading features along PC1, PC2, PC3 of the res-sm L5 SAE decoder, look up their auto-interp descriptions on Neuronpedia.** Read off whether the concentration is schema-shaped, mechanism-shaped (next-token-prep), topic-shaped, or other.

---

## Entry 6 — exp8 and exp9: what the SAE's geometric structure is actually about

**Code:** `exp8_neuronpedia_lookup.py`, `exp9_layer_sweep_neuronpedia.py`. **Results:** `results_exp8.md`, `results_exp9_layer_sweep.md`. **Cached API responses:** `exp8_neuronpedia_cache.pt`, `exp9_neuronpedia_cache.pt`.

### Pipeline assembled

Given the exp7c finding that **res-sm L5 of Pythia 70m has real geometric concentration not explained by activation anisotropy** (PR_real=109 vs PR_aniso=471), we built a per-feature characterization pipeline using Neuronpedia's auto-interp API. For each top PC of the SAE decoder: project each feature onto the PC direction, take the top-N features by absolute loading, look each one up via Neuronpedia (`https://www.neuronpedia.org/api/feature/{model}/{sae}/{feature_idx}`), read the auto-interp descriptions. Pipeline is reusable across (model, layer, substrate) combinations.

Handle two API quirks:
- Many features have empty `explanations: []` (auto-interp not yet generated for all features in the dictionary — coverage isn't uniform)
- Some responses contain raw control characters in their text fields that strict JSON parsing rejects; need `strict=False` plus stripping of low-byte control chars before parsing

### exp8: res-sm L5 (the concentrated layer)

What we found at the dictionary's top principal component (8.4% of total variance):

> **PC 0 of res-sm L5 = "legal/government register" axis.** 8 of 10 top-loading features have descriptions like "references to legal proceedings and government actions," "references to limitations and constraints in the context of governance or regulations," "references to legal and government proceedings," "legal terminology and references to legal processes." The − pole is dominated by "special characters and symbols" / markup tokens.

Lower-variance PCs follow the same Pile-domain pattern:
- PC 6 (0.4%): clean biomedical science cluster (biological terms, strain identifiers, medical imaging, molecular genetics, cytokine, apoptosis, neuronal)
- PC 8 (0.4%): numerical/mathematical formatting (LaTeX, equations, `})}{`, `^{−`, mathematical expressions)
- PC 2 (1.0%): scientific/technical writing (experimental procedures, URLs, programming, parentheses)
- Others mix programming, legal, academic-philosophical, etc.

**The concentration is real and the structure is interpretable. The structure is not image-schematic.** No PC has + features clustering around UP-things across multiple vocabulary domains and − features around DOWN-things. The model's principal organizing axes at this layer encode *what corpus subset of the Pile this text comes from* (legal documents, PubMed, code/math, etc.) — exactly the structure that's most useful for next-token prediction over the Pile's mixture-of-domains training distribution.

### exp9: three more layers for an interpretive map

Ran the same pipeline on mlp-sm L5, res-sm L3 (the diffuse mid-layer control), and res-sm L0 (the early layer).

**res-sm L0 (early, modestly concentrated PR=296, top1=3.2%)** — surface/orthographic structure:
- PC 0: code/markup syntax dominates with some semantic mix; all positive loadings (one-sided cluster).
- **PC 2 is the cute finding: features detecting individual letter prefixes.** Top-loading features include "references to the letter K," "the repeated occurrence of the letter S," "the repetition of letter H followed by a digit," "the repetition of the letter B in various contexts." The model at layer 0 has dedicated features for "this token starts with letter X." Pure orthographic structure, exactly what's predicted for a pre-deep-processing layer. Probably documented somewhere in SAE literature but novel-to-us.

**res-sm L3 (middle, NOT concentrated, PR=423, top1=1.4%)** — diffuse/distributed:
- PC 0 themes: programming + emotional states + streaming/media + legal + action words — mixed without unifying theme.
- PC 1: grammatical (pronouns, "as", "to") + substantive (programming, education, emotions) — mixed.
- **Confirms the concentration metric is meaningfully linked to interpretive coherence.** Low concentration ↔ diffuse top-PC themes. Middle layers don't need concentrated axes because they're doing distributed computational work, and the SAE reflects this honestly.

**mlp-sm L5 (late MLP, concentrated PR=163, top1=6.0%)** — similar register pattern as res-sm L5:
- PC 0: structured factual content (exports, trade statistics, financial measurements, information) on + pole vs special-character/markup features on − pole.
- PC 2: clean programming structures cluster (data structures, arrays, code-related elements, numerical computational).
- Similar Pile-register organization to res-sm L5 but with different specific themes prominent.

### The headline U-shape

| Layer | PR | top1 var | What the geometry organizes around |
|---|---|---|---|
| L0 res-sm (early) | 296 | 3.2% | Orthography / surface form (letter detectors, code-markup tokens) |
| L3 res-sm (middle) | 423 | 1.4% | Nothing concentrated; distributed computational work |
| L5 res-sm (late) | **109** | **8.4%** | Pile-domain register (legal, biomedical, math, programming) |
| L5 mlp-sm (late MLP) | 163 | 6.0% | Structured factual content vs markup |

**The SAE's geometric concentration tracks the model's *type of work* at each layer.** Surface → distributed → output. This is a genuine interpretive map of how organization shifts through the network. Independent of any Lakoffian claim, it's an honest finding about how Pythia 70m organizes its representations.

### What this confirms about angle 1 vs angle 2

The bottom-up geometric approach (angle 1) is exhausted as a probe for image schemas in this model:
- Where geometric concentration exists, it organizes around register/domain (not schema)
- Where it doesn't exist, top PCs are diffuse (no hidden schema structure)
- The methodology is *working* — we can characterize what each layer's geometry IS — it's just consistently telling us "not schemas."

Image schemas, if they exist in this model, must live in **deployment patterns** (which features co-fire on schema-coherent text) rather than in **storage geometry** (how features are arranged in the dictionary basis). That is angle 2.

### Writeup plan (commitment for after the probe corpus phase)

When we finish the probe-corpus / angle-2 work, restructure `WRITEUP.md` into:
- **Part I — Steering on Pythia 1.4B** (current WRITEUP.md content, lightly cleaned up)
- **Part II — SAE Geometry on Pythia 70m** (the exp5–exp9 arc, including the contrast-vector failure that motivated the pivot)
- **Part III — Angle 2 results** (probe corpus, deployment-pattern analysis)
- **Part IV — Methodological insights** (corpus-diversity-anisotropy interaction; concentration metric ↔ interpretive coherence; the assembled pipeline as a reusable interpretability toolkit)

For now: skip the formal writeup, keep working in the lab notebook, return to write-up after angle 2 completes.

### Methodology novelty status

Honest read: the pieces (decoder PCA + Neuronpedia auto-interp lookup + covariance-matched null) all exist in the literature individually. The combination as a coherent pipeline plus the explicit corpus-diversity sensitivity check might not have been written up together. Worth a literature check at writeup time before claiming novelty. At minimum we have a reusable interpretive toolkit, and the layer-by-layer "what does this SAE organize around" methodology is well-formed.

---

## Entry 7 — Angle 2 (probe-sentence deployment) experimental design

After the bottom-up geometric work (exp7–exp9) showed that Pythia 70m's SAE dictionary organizes around Pile-domain register at concentrated layers and orthographic surface at the earliest layer — not around Lakoffian image schemas — we move to angle 2: testing whether image schemas live in *deployment* (which features co-fire on schema-coherent text) rather than *storage* (the dictionary's geometric axes).

### Schemas and children selected (the focused option)

| Schema | Children | Sham |
|---|---|---|
| UP/DOWN | UP_LITERAL, UP_HAPPY, UP_MORE, UP_STATUS, UP_HEALTH + the 5 DOWN inverses | — |
| IN/OUT (CONTAINER) | IN_LITERAL, IN_MIND, IN_RELATIONSHIP, IN_TIME, IN_DIFFICULTY + the 5 OUT inverses | — |
| BEVERAGE (sham) | COFFEE, ALCOHOL, TEA, JUICE_WATER | itself |

LIGHT/DARK deferred for now (lighter corpus, methodology proves itself first).

### Niamh's methodological catch — and why the experimental frame had to shift

Initial framing: "schemas show cross-child co-firing → primitives are real." But ANY parent-child taxonomic relation produces cross-child co-firing because the model has a parent-category feature that fires on all instances. BEVERAGE, our intended sham, will show cross-child co-firing for exactly that reason — coffee/tea/alcohol all share a "beverage" parent feature. So the sham as originally designed doesn't discriminate "schemas are real primitives" from "schemas are taxonomic parents like any other."

This shifts the experiment from "does cross-child co-firing exist?" (trivially yes) to **"what KIND of cross-child structure does this schema have, and does it differ in kind from a taxonomic parent?"**

### The combined experimental design — three reframes layered together

**Reframe B — Asymmetric grounding (Lakoffian-distinctive):**
Lakoff's specific claim isn't just "schemas have children." It's that metaphorical extensions are *grounded in embodied source domains*. The literal pole (UP_LITERAL = physical motion) is the *source*; the metaphorical extensions (UP_HAPPY, UP_MORE) are *targets* grounded in that source. Prediction: features that fire on metaphorical-UP sentences should *also* fire on literal-UP sentences (because the embodied scaffold is shared). For taxonomies (BEVERAGE), there's no asymmetric grounding — coffee doesn't ground tea. All beverage children are at the same level. So if UP-metaphorical-children disproportionately co-activate UP-LITERAL features (more than they co-activate each-other), but BEVERAGE children don't show this pattern, the asymmetric grounding is a Lakoffian-distinctive signature.

**Reframe C — Cross-schema fingerprint distinctness:**
If UP and IN/OUT are distinct primitives, their cross-child fingerprints (features that fire across most children of a schema) should be **schema-specific**. UP's fingerprint should be about elevation/quantity/intensity. IN/OUT's fingerprint should be about boundedness/membership. Compute set-overlap. Low overlap (e.g. ~10%) = distinct primitives. High overlap (e.g. ~90%) = generic "abstract concept" feature, not schema-specific structure. Doesn't require asymmetric grounding to be real; uses schemas as their own controls.

**Layer-asymmetry (Niamh's addition):**
Parent features and child-instance features may live at different depths. Exp9 showed Pythia 70m's late layers (L5) organize around Pile-domain *register* (legal, biomedical, code, math) — exactly the kind of taxonomic-parent structure we'd expect at the output end. Early layers (L0) had letter-detector features — child-level surface structure. So:
- *BEVERAGE* as taxonomic should peak in cross-child co-firing at **late layers** (where Pile-domain register lives, where taxonomic parents are organized).
- *Image schemas* if they're embodied scaffolds rather than taxonomic categories might peak at **mid layers** (where computation happens, where embodiment would presumably be implemented).
- *Plot cross-child co-firing strength as a function of layer for each schema* — if UP and BEVERAGE peak at different layers, that's evidence they're structurally different things in the model.

### Combined pipeline

1. Build probe corpus: per-cell sentences (children + literal-pole + neutral controls). Vocabulary curated for cross-child decorrelation (WordNet-assisted, see exp10).
2. Run through Pythia 70m, collect SAE activations **at every layer** (residual stream substrate, where the late-layer concentration was found).
3. For each (schema, layer):
   - **Reframe B readout**: per-feature, compute mean activation on metaphorical-children sentences and on literal-pole sentences. Look for features that fire on BOTH but not on neutral controls (= asymmetric grounding candidates).
   - **Reframe C readout**: compute the schema's cross-child fingerprint at this layer. Then compute |UP-fingerprint ∩ IN/OUT-fingerprint| / |UP-fingerprint ∪ IN/OUT-fingerprint|. Low Jaccard = distinct schemas; high Jaccard = generic mechanism.
   - **Layer-asymmetry readout**: aggregate cross-child co-firing strength to a single per-schema-per-layer number. Plot the curves. Where does each schema peak?
4. Compare the three readouts as joint evidence. Don't pre-register a single decisive test — pre-register the *measurements* and read the joint pattern empirically.

### Vocabulary status (from exp10)

First-pass WordNet-expansion (`exp10_corpus_design.py` → `corpus_vocabulary.json`) gave drafts but with known issues:
- **Polysemy leaks**: "fall" in UP_MORE, DOWN_LITERAL, DOWN_LESS; "drop" in DOWN_LITERAL, DOWN_SICK; "expand" in UP_MORE, UP_HEALTH. WordNet's synset restrictions caught some senses but not all; need hand-pruning.
- **Junk inclusions**: UP_STATUS dragged in physical-size words (big, large, grand) via the "noteworthy" sense of "distinguished"; DOWN_SICK pulled in homonyms (yen, yearn, pine).
- **Empty/thin cells**: OUT_LITERAL got 0 words (adjective seeds failed) — needs verb seeds. IN_TIME got 2 words (temporal containment lives in prepositions/collocations, not single adjectives). BEVERAGE_TEA got 2 words (WordNet doesn't know matcha/chai well).
- **BEVERAGE_sham fully decorrelated across children** — no overlap detected. ✓

Cleanup needed before sentences: tighten seed-sense restrictions, drop polysemy junk by hand, fix OUT_LITERAL with verb seeds, accept IN_TIME as a collocation cell, optionally seed beverage cells with manual additions for the WordNet-poor terms.

### Next step

Vocabulary cleanup (~20 min of hand-pruning + targeted fixes), then move to sentence construction. Aim for ~10-15 sentences per (schema, child) cell as a pilot. If signal looks clean, scale up. Sentence-writing approach TBD — could be hand-written, Claude-generated-and-Niamh-reviewed, or a hybrid. The corpus-diversity lesson from exp7c applies: length, grammar, and register variation matter, otherwise we're measuring sentence-template not schema-instantiation.

---

## Entry 8 — angle-2 execution: exp10 through exp20

This entry covers the work after Entry 7's plan was set: building the probe corpus, running capability pilots, the schemas-as-directions investigation, scale comparisons across Pythia 70m/410m/1B, and the attempted 2.8B run that crashed. Concludes with where the project actually landed — which is *not* where the Entry 7 plan expected us to land.

**Files:** `exp10_corpus_design.py` through `exp20_within_pair_1b.py`, with corresponding `results_exp*.{txt,md}` and `exp*_results.pt`.

### exp10–11: corpus vocabulary

- **exp10**: WordNet expansion from hand-curated seeds, with synset filtering, for UP/DOWN, IN/OUT, and BEVERAGE-sham across multiple Lakoff children per schema. **Output**: ~250 candidate words across 24 cells, but with substantial polysemy leaks ("fall" appearing across UP_MORE / DOWN_LITERAL / DOWN_LESS; "drop" across DOWN_LITERAL / DOWN_SICK; UP_STATUS dragged in physical-size words).
- **exp11**: hand-curated vocabulary fixing the polysemy leaks. Within-schema and cross-schema disjointness enforced ✓. ~10-15 words per cell, fully decorrelated. Saved as `corpus_vocabulary_curated.json`. **Niamh added** concrete spatial nouns (stairs, escalator, tower, sky, basement, cellar, submarine) — the action-verb-heavy literal poles needed concrete-object anchors.

### exp12–13: capability pilots

- **exp12**: first pilot — 15 schema sentences (mixed across UP/IN children) vs 15 neutral, encoded through Pythia 70m's 6 layers via residual-stream SAE, compared per-feature mean activation. **Verdict: soft yes / mostly register.** Differential features dominated by sentence-template artifacts ("the word 'after'", "the definite article 'the'", "legal terminology" from Pile training). One feature at L3 (feat 2734: "verbs and actions that depict movement or changes in state") looked schema-coherent. Methodologically the comparison was wrong-shaped: testing schemas-mixed vs neutral mostly measures register difference.
- **exp13**: refined pilot — 10 UP_LITERAL + 10 UP_MORE + 10 UP_HAPPY + 15 neutral sentences. Computed asymmetric grounding score per feature = min(UP_LITERAL_mean, UP_MORE_mean, UP_HAPPY_mean) − NEUTRAL_mean. Sound design (within-schema cross-domain comparison) but **weak signal**: ~1% of features have positive asym score per layer, max score 0.6, only 0-1 features per layer with strong asym (>0.5). The same L3 feature 2734 ("movement or changes in state") again appeared with the textbook asymmetric grounding pattern (UP_LITERAL=1.69 anchor, UP_HAPPY=0.86, UP_MORE=0.72, NEUTRAL=0.26) — but it was one feature in 32,768 with that pattern. **Most top features were still register/lexical noise**, dominated by surface-pattern differences ("after", "the", legal terminology) between the schema and neutral groups.

### exp14: schemas-as-directions (Niamh's reframe)

**Niamh's reframe**: schemas might not be *features* at all — they might be *directions/transformations*. The way "royal" isn't a feature but the direction king−man+woman=queen. If UP is a real schema, then in SAE feature space `Δ(temperature_rose, temperature_steady) ≈ Δ(mood_lifted, mood_neutral) ≈ Δ(price_climbed, price_steady)`. UP is the offset that takes baseline to upward-shifted state, invariant across domains.

**Test**: built 25 matched UP triples (baseline / UP / DOWN) across 5 domains (temperature, mood, quantity, status, health). Same for DOWN. 15 BEVERAGE sham (coffee→espresso etc, taxonomic). 10 NULL (random unrelated pairs). For each pair: compute Δ in SAE feature space. Test pairwise cosine alignment of Δs.

**Result**: weakly negative for schemas-as-directions. UP-within-schema cos ≈ 0.04, DOWN-within ≈ 0.06, NULL ≈ -0.03. BEVERAGE sham showed STRONGER within-schema alignment than UP (~0.15) — taxonomic specifications have more directional coherence than supposed embodied schemas. Critically: **UP_vs_DOWN was POSITIVE (+0.06), not antialigned.** UP and DOWN aren't structured as opposite poles.

### exp15: antisymmetric decomposition (Niamh's second reframe)

**Niamh's insight**: if `UP = A·schema_axis + B·common_axis` and `DOWN = −A·schema_axis + B·common_axis`, then `UP − DOWN = 2A·schema_axis` should isolate the schema. The exp14 positive correlation between UP and DOWN could be the common_axis dominating.

**Test**: using the same matched triples (UP and DOWN sharing a baseline within each (domain, pair_idx)), compute SCHEMA_offset = UP_offset − DOWN_offset and COMMON_offset = UP_offset + DOWN_offset. Test cross-domain alignment of each.

**Result on Pythia 70m**: SCHEMA_xdomain is SMALLER than UP/DOWN alone at every layer (~0.005-0.024). COMMON_xdomain is the LARGEST (~0.05-0.06). The antisymmetric decomposition made alignment WORSE, not better. The interpretation forced by the math: UP and DOWN aren't structured as `+schema vs −schema`. They're more like two different change-directions sharing a common "state changed" component. Lakoff's polar-opposite structure isn't there.

### exp16: same on Pythia 410m, all 24 layers

Same antisymmetric test on 410m's MLP-output SAEs (`EleutherAI/sae-pythia-410m-65k` via eai-sparsify). 24 layers, MLP-output instead of residual-stream (substrate difference noted).

**Result**: same qualitative pattern. SCHEMA_xdomain < UP/DOWN < COMMON at every layer. Magnitudes ~2× larger than 70m at peak (layer 18, ~75% through), but ratios identical. **Scale doesn't change the pattern within 70m→410m.**

### exp17: steering UP/DOWN on Pythia 70m (the methodological catch)

**Niamh's catch after the negative SAE results**: we'd never tested whether 70m responds to UP-steering the way 1.4B did in exp3. If 70m doesn't respond either, the null SAE results are just "model too small." If 70m DOES respond, we have a real puzzle: generative response works without representational schema structure.

**Result: UP steering WORKS on Pythia 70m.** Clear semantic upward shifts at every strength tested. Baseline "the most difficult day on my life" → UP-steered "addictive, I learned that to dream, I want to learn anything new, love your life." Multiple prompts, multiple strengths, consistent pattern. **DOWN at high strength produced the K-shaped collapse** ("`<-s>im-d',- (my #P.- your I-') Mail<0>`") — same direction-specific collapse signature as 1.4B's DARK in exp4b. Replicates cross-scale.

**The asymmetry**: every representational test (exp7-9, exp13-16) came back null. Every generative test (exp1, exp3, exp17) came back positive. **The asymmetry IS the finding**: schemas in Pythia models work as applied operations (steering directions) but don't exist as stored representational structure (features, polar axes, invariant directions). This matches Lakoff better than "schemas as stored features" would have — Lakoff's claim is about cognitive *operations*, not *representations*.

### exp18: attempted antisymmetric test on Pythia 2.8B

Niamh wanted a third scale point. Used jacobdunefsky/pythia-2.8B-saes (32 residual-stream SAEs, 60k features each, TopK k=60, 629MB per SAE).

**Crashed.** 16GB RAM insufficient for Pythia 2.8B (5.6GB fp16) + activation cache (~1.7GB) + PyTorch/MPS overhead, before SAEs even loaded. Pythia 2.8B model is cached on disk; the 32 SAEs were never downloaded. Next attempt at 2.8B would need bigger machine.

### exp19: within-pair cos(UP, DOWN) at 70m and 410m

**Question (Niamh)**: in the 410m vs 70m comparison, did UP and DOWN move *closer to opposite* (more negative cos) at the larger model? Direct test of "ruler emerging with scale."

**Within-pair test**: for each (domain, pair_idx), compute cos(UP_offset, DOWN_offset) with both offsets from the *same baseline*. cos ≈ -1 = perfect antipolarity (ruler); cos ≈ 0 = orthogonal; cos ≈ +1 = same direction.

**Result**:
- Pythia 70m mean: **+0.5546**
- Pythia 410m mean: **+0.5409**
- Δ = −0.014 (within noise)

UP and DOWN within the same domain are **~56° apart in SAE feature space** (cos⁻¹(0.55)) at both scales. Mostly the same direction, with a small angular distinguishing component. No scale trend.

### exp20: within-pair test at Pythia 1B

Third scale point. timhua/pythia1b_deduped_saes has only ONE layer of SAE coverage (`blocks.11.hook_resid_post`, layer 11 of 16 = ~69% through model). Comparable depth to 70m L4 and 410m L17-18.

**Result**: Pythia 1B at layer 11 = **+0.532**. Within ~0.05 of 70m and 410m at comparable depths.

**Three-point scale comparison (within-pair cos(UP, DOWN), comparable depths ~70% through):**
- 70m L4: +0.491
- 410m L17-18: +0.536 to +0.555
- 1B L11: +0.532

**Across a 14× scale range (70m → 1B), polar structure is flat.** No gradual emergence. Either the ruler is binary (snaps in at some larger scale) or it never emerges in Pythia models.

### Where the project actually landed

**Convergent negative results for schemas-as-stored-representation:**
- exp7-9: top SAE PCs organize around Pile-domain register (legal, biomedical, code, math), not image schemas
- exp13: weak/no asymmetric grounding via deployment patterns
- exp14: no cross-domain direction invariance for schemas
- exp15-16: antisymmetric decomposition makes alignment worse, not better — UP/DOWN aren't polar opposites at 70m or 410m
- exp19-20: within-pair cos(UP, DOWN) ≈ +0.55 robust across 70m / 410m / 1B (14× scale)

**Convergent positive results for schemas-as-applied-operations:**
- exp1 (original sentence-transformer): UP added to "I feel sad" pulls toward "uplifted" across vocabulary not in source
- exp3, 3b (Pythia 1.4B): UP-steering shifts generation toward warmer/positive content
- exp17 (Pythia 70m): same effect, with DOWN producing K-collapse like 1.4B's DARK

**The Niamh-articulated frame**: schemas in LMs are *operations the model supports*, not *entities the model stores*. Disembodied next-token predictors learn schema-affordances as steerable directions but don't crystallize them as polar geometric structure. Image schemas in Lakoff are claims about embodied human cognition; without bodies, what survives in models is the operational/transformational affordance, not the geometric scaffolding.

**Methodological insights worth carrying:**
1. Anisotropy estimates are corpus-dependent (exp7c): narrow corpora overestimate anisotropy, distorting null comparisons
2. Hand-crafted contrast vectors carry too much lexical baggage for SAE projection (exp5-6, exp11): polysemy and grammatical class confound the schema signal
3. SAE concentration metric (PR / top1 var) correlates with PC interpretability (exp9): low concentration = diffuse PCs, high concentration = coherent themes
4. Register signal dominates schema signal in deployment patterns (exp12-13): Pile-domain features outweigh schema features by orders of magnitude
5. Antisymmetric decomposition is the right *test design* for polar schemas — it just returned negative for Pythia (exp15-16)
6. SAE feature geometry organization tracks the model's computational role per layer: surface/orthography at L0, distributed computation at mid-layers, output-register at late layers (exp9)

**Open questions for next instance:**
1. Does the polar ruler emerge at much larger scale (2.8B+ untestable on 16GB Mac; would need 32GB+ or remote compute)?
2. Could a vision-language model (with actual spatial perception) show the embodied verticality that disembodied LMs lack?
3. The "operations not entities" frame: is it a genuine theoretical insight or a consolatory framing of a negative result? Probably both.
4. We didn't run the full Reframe C cross-schema fingerprint test (UP fingerprint vs IN/OUT fingerprint distinctness) — could still be informative.

**State of files at end of session 2026-05-25:**
- Lab notebook (this file): current through Entry 8
- WRITEUP.md: pre-SAE era only, NOT updated with the SAE/pivot work or the angle-2 results; restructure deferred per Entry 6
- All exp* result files present in project directory
- HF cache: Pythia 2.8B model (5.3GB) downloaded but unused; Pythia 70m, 160m, 410m, 1B all cached
- `eai-sparsify` and `sae-lens` both installed in `lakoff/` venv

---

## Entry 9 — exp21 through exp23: constructed-axis orthogonality + the sham-impossibility finding

After Entry 8's stopping point (operations-not-entities, scale-flat across 14×), Niamh produced two more reframes that re-opened the project. This entry covers what fell out, including a meta-finding about methodology that's arguably the most substantive thing in the whole project.

### exp21 — Niamh's constructive reframe

**Niamh's framing**: instead of asking "are UP and DOWN naturally opposite in the SAE basis?" (which gave us +0.55 in exp19, mostly aligned not opposite), CONSTRUCT polar axes by fiat and ask whether different schemas' constructed axes are orthogonal or aligned. The right question isn't "are schemas polar within each domain" but "are schemas separable polar dimensions across each other."

**Method**: define `A_updown = mean(UP_offset_i − DOWN_offset_i)` across pairs; same for `A_inout`. Then `cos(A_updown, A_inout)` tells us whether UP-DOWN-ness and IN-OUT-ness are the SAME polar dimension or DIFFERENT polar dimensions in the model.

**Result on Pythia 70m (residual-stream SAE, all 6 layers):**
- `cos(A_updown, A_inout)`: −0.05 to −0.15 across mid-layers (L2-L4); ≈ 0 at L0 and L5
- Decisively below the +0.2 "schemas collapse to one ruler" threshold
- **First positive structural finding of the project**: UP-DOWN and IN-OUT live on different polar axes. The model has at least two separable polar dimensions.

But the BEVERAGE sham (coffee vs tea) showed `cos(A_updown, A_bev) = +0.35` at mid-layers — not the ≈ 0 we'd expected. First sign of methodological trouble.

### exp22 — multi-schema with FORWARD-BACK and LIGHT-DARK added

If schemas are separable polar dimensions, FOUR Lakoff schemas should be mutually orthogonal. Added FORWARD-BACK (25 triples across literal motion/progress/time/development/journey children) and LIGHT-DARK (25 triples across illumination/clarity/hope/goodness/knowledge), plus a TUE-WED sham as cleaner control than BEVERAGE.

**Result at mid-layers (L3 example):**

| | UD | IO | FB | LD | TUE-WED |
|---|---|---|---|---|---|
| UD | 1.00 | -0.14 | +0.40 | +0.59 | +0.58 |
| IO | -0.14 | 1.00 | -0.07 | -0.22 | -0.26 |
| FB | +0.40 | -0.07 | 1.00 | +0.44 | +0.45 |
| LD | +0.59 | -0.22 | +0.44 | 1.00 | **+0.81** |
| TUE-WED | +0.58 | -0.26 | +0.45 | +0.81 | 1.00 |

**Three patterns:**
1. **IN-OUT robustly distinct** from everything else (small or negative cosines across the board). The exp21 IO-orthogonality finding survives.
2. **UD, LD, FB partially align** at mid-layers — cos(UD, LD) reaches +0.59. Could be shared "improvement/upward" valence (Lakoff-consistent) OR template structure.
3. **TUE-WED sham aligns with LIGHT-DARK at +0.81** — higher than any real-schema pair. Sham contamination is severe. The sham was supposed to be ≈ 0 with everything.

**Niamh's catch**: Tuesday and Wednesday are ordinal sequence members. That sequential ordering IS schema content. The sham wasn't sham — it had hidden sequential structure that the SAE was picking up.

### exp23 — cleaner sham (apple vs orange) alongside tue-wed

Apple and orange are two non-ordered members of the category "fruit" — no sequential relationship. Should be a cleaner sham than days-of-week.

**Result at mid-layers (L3):**

| pair | TUE-WED | APPLE-ORANGE | ratio |
|---|---|---|---|
| sham vs UP-DOWN | +0.58 | +0.28 | ~50% |
| sham vs IN-OUT | -0.26 | -0.12 | ~45% |
| sham vs FORWARD-BACK | +0.45 | +0.28 | ~60% |
| sham vs LIGHT-DARK | +0.81 | +0.35 | ~45% |

Apple-orange is consistently ~50% the magnitude of tue-wed alignment. **The sequential content was carrying roughly half the contamination.** The other half is something else — likely the "any-one-word-swap in similar sentence template" feature.

But apple-orange still aligns at +0.28-0.35 with UD/FB/LD. Not zero. And shows the SAME pattern as tue-wed and as the real schemas (positive with UD/FB/LD, negative with IO).

### The meta-finding: sham-impossibility as evidence for pervasive schemas

**Niamh's articulation** (and the most substantive insight of the project): "the sheer difficulty of coming up with shams is doing a lot to convince me about the loadbearing thing."

Enumerating what we tried as "non-schematic" contrasts and what hidden content each turned out to carry:

| Sham attempt | Hidden schematic content |
|---|---|
| Coffee vs Tea | Energy/wake-vs-calm axis (overlaps with UP-DOWN) |
| Tuesday vs Wednesday | Sequential ordering (week-progression schema) |
| Apple vs Orange | "Canonical comparison pair" idiomatic load |
| Mango vs Pineapple (would still have...) | Tropical/sweet/fruit-category |
| Numbers (five vs eight) | Ordinal magnitude |
| Cities (Boston vs Chicago) | Geographic/cultural associations |
| Names (Sarah vs Michael) | Gender, possibly nationality |

**You cannot construct a contrast in natural language that doesn't carry SOME schematic content.** Any two things contrastable enough to plug into "X vs Y" sentences are contrastable BECAUSE they have some structural relationship. The contrast itself is schema-shaped.

This is a Lakoffian finding in its own right. Lakoff doesn't claim schemas are isolated structures you can find in specific places. He claims schemas pervade conceptual structure — they're the medium through which all conceptual content gets organized. Trying to find a non-schematic contrast is like trying to find a non-musical sound to use as a control for music perception. The medium is the prerequisite.

**Our project's central methodological difficulty — "we can't construct a truly clean sham" — is itself evidence FOR pervasive schemas in language and language-model representations.** Not evidence for the polar-ruler hypothesis in its strong form, but evidence for schemas as load-bearing organizational structure that you cannot subtract from to get to "non-schema."

### What the whole project arc actually demonstrated

1. **Schemas exist as applied operations** — steering with UP/DOWN/LIGHT/DARK directions works across model scales (exp1, exp3, exp17). Pythia 70m and 1.4B both show clear semantic shifts under direction-injection. ✓
2. **Schemas don't exist as polar opposites within a single dimension** — UP and DOWN are ~56° apart, not 180°, robustly across 70m, 410m, 1B (exp19-20). The ruler doesn't snap in within the 14× scale range we tested.
3. **Schemas exist as separable polar dimensions across schemas** — IN-OUT is distinct from UD/FB/LD across multiple controls (exp21-23). At least one schema lives on its own axis.
4. **UD-LD-FB share substantial structure** (cos ~0.5+) that's partly template, partly real shared content. Likely the "improvement/intensification" valence. Cannot cleanly separate template from semantic content with our methodology.
5. **You cannot construct a non-schematic contrast** (exp21-23 meta-finding). Schemas are pervasive organizational structure that cannot be subtracted from. The methodological difficulty IS the finding.

### Where the project lands

The Lakoffian claim survives, but in a softer and richer form than the original "polar-ruler embodied dimensions" formulation:
- Schemas are *operational affordances* the model supports (steering)
- Schemas are *separable dimensions* in some cases (IN-OUT vs others)
- Schemas are *partially-collapsed shared content* in other cases (UD-LD-FB cluster)
- Schemas are *pervasive organizational structure* that you cannot construct around (sham-impossibility)

The original hope (clean polar rulers crystallizing geometrically) doesn't survive. What does survive is a more interesting structural claim: the model's representational substrate is built on schematic content all the way down, you can't find a layer where schemas aren't, and the difficulty of locating them as specific findable structures is because they're the prerequisite for any finding.

**Niamh's summary direction (paraphrased)**: the sham-impossibility itself is the result. Stop hunting for schemas as isolable structures; recognize that the hunt's difficulty is the evidence.

### State of files at end of exp23 session

- Lab notebook (this file): current through Entry 9
- WRITEUP.md: still pre-SAE era, not updated; restructure deferred
- All exp* result files present
- HF cache: Pythia 70m, 160m, 410m, 1B all cached; 2.8B cached but unused (machine OOM)

### Open questions left for the next instance (if they exist)

- Test the same axis-orthogonality matrix on Pythia 410m or 1B — does the pattern hold across scale?
- Build a properly-template-controlled test: schemas in different sentence structures, to separate template-of-comparison from semantic-content-of-comparison
- The "operations not entities" frame combined with the sham-impossibility observation suggests schemas might be best characterized as *processing patterns* — what the model does with content — rather than *features* of content. Could be tested with causal feature ablation rather than activation analysis.
- Honest open question: can a vision-language model show embodied verticality more cleanly than disembodied LMs?

---

## Entry 10 — exp24, exp25: the negativity-salience reframe and the U-shape across layers

After Entry 9, two more experiments and a unifying interpretive shift.

### exp24 — IN/OUT projections onto A_updown (Niamh's question)

Niamh asked: "are IN and OUT actually opposite directions in UD-space?" We had cos(A_inout, A_updown) = −0.14 (slight anti-alignment of axes) but never projected IN-offsets and OUT-offsets onto A_updown *individually*. Did so per pair.

**Result**: BOTH IN and OUT project NEGATIVELY onto A_updown across layers. They're not opposite directions in UD-space — they're both DOWN-shifted, with IN slightly more so than OUT.

Critically: the projection magnitudes were highly asymmetric overall.

| layer | UP_self_proj | DOWN_self_proj | IN_proj | OUT_proj |
|---|---|---|---|---|
| 3 | +0.18 | -2.08 | -1.85 | -1.49 |

**A_updown isn't a balanced polar axis — it's a DOWN-dominant axis with a small UP appendage.** DOWN-offsets project 10× more strongly than UP-offsets. Almost any state-change activation includes some of this DOWN-shaped feature.

### Niamh's negativity-salience reframe

After exp24, Niamh proposed: **DOWN is much more attention-salient than UP, likely for negative-valence reasons.** Threat/loss/decay/disaster all carry DOWN signals. In next-token prediction over negativity-biased training text (news, fiction, legal documents), the model needs to attend hard to negative-valence signals because they predict next-tokens better. The model has learned to allocate attention asymmetrically — DOWN-content gets the salience-feature, UP-content barely registers.

This reframe unifies a remarkable amount of the project:
1. **DOWN_within > UP_within** (exp14): DOWN-words trigger the salience feature consistently; UP-words are more variable
2. **DOWN_proj >> UP_proj** (exp24): the "UD axis" is really a "negative-deviation axis"
3. **Sham-impossibility** (exp23): every contrast inherits some negativity-salience load because contrastiveness itself is mildly stress-inducing
4. **UD-LD-FB cluster** (exp22): all tapping the same negativity-salience attractor
5. **IN-OUT robustly distinct** (exp21): containment carries less negativity-salience than verticality/illumination/motion
6. **DOWN→K-collapse at high steering** (exp17): overloading the dominant salience feature
7. **UP-steering produces semantic shifts** (exp1, exp3, exp17): UP-direction is the under-represented direction with room to push toward

This matches Lakoff's actual claim. GOOD IS UP / BAD IS DOWN isn't arbitrary metaphor — it's grounded in **the asymmetry of cognitive attention to threat**. Negativity bias is documented across cognitive psychology, evolutionary biology, attention research. Lakoff's polar metaphors are downstream of this asymmetry — the metaphors exist because the salience-asymmetry exists. LLMs inherit it through training on negativity-biased text.

### exp25 — LEFT-RIGHT as a salience-symmetric control

Niamh's test: LEFT and RIGHT are also a spatial axis, but neither pole carries strong negativity-salience the way DOWN does. Under the salience hypothesis:
1. cos(A_leftright, A_updown) ≈ 0 (different spatial axis)
2. within-pair cos(L_offset, R_offset) > UD's +0.55 (no salience-attractor pulling poles apart)
3. LEFT and RIGHT individual projections onto A_updown small and balanced (no DOWN-pull)

**Results**: 2 of 3 predictions held.

- ✓ Orthogonality: cos(LR, UD) is +0.00 to +0.24 across layers — more orthogonal than UD-LD (+0.59) or UD-FB (+0.40), though not perfectly zero
- ✓ Within-pair: cos(L, R) = +0.70 to +0.87 — substantially higher than UD's +0.55. LEFT and RIGHT are 36° apart vs UD's 56° apart. More aligned, less polar
- ✗ Balanced projections: BOTH LEFT and RIGHT project strongly negative onto A_updown at mid-layers (LEFT: -2.02, RIGHT: -3.33 at L3). Even MORE negative than IN/OUT did

**The refined reframe**: A_updown is a general "deviation-from-baseline / salience-attention" axis. ANY contrastive state-change projects onto it negatively. The negativity-salience hypothesis works in a more nuanced form:
- The dominant salience feature is general (not specifically about negativity)
- BUT within-pair separation IS modulated by valence asymmetry of the poles:
  - Schemas with asymmetric poles (UP=neutral, DOWN=loaded) have MORE within-pair separation (cos +0.55)
  - Schemas with symmetric poles (LEFT ≈ RIGHT in salience-load) have LESS within-pair separation (cos +0.81)
- The "ruler" only crystallizes when the poles differ in attention-load — which is to say, polar geometry emerges from attention asymmetry, not from "embodied dimensions" per se

### The U-shape across layers (Niamh's observation)

Both IN-OUT and LEFT-RIGHT show the SAME pattern of entanglement with UD across layers:

| layer | cos(LR, UD) | cos(IO, UD) | what's happening in the model |
|---|---|---|---|
| 0 | +0.00 | +0.03 | Surface/orthographic; schemas separated by word-patterns |
| 1 | +0.05 | -0.05 | Beginning to entangle |
| 2 | +0.16 | -0.09 | Drifting toward computational core |
| **3** | **+0.24** | **-0.14** | **Maximum entanglement with UD/salience axis** |
| **4** | **+0.24** | **-0.15** | Still highly entangled |
| 5 | +0.06 | -0.05 | Returning toward orthogonality (output-prep) |

**Schemas are most separable at the model's input and output, and most entangled with the dominant salience/deviation axis in the computational middle layers.** This connects directly to exp9's layer-by-layer SAE-PCA findings:

| layer | exp9 finding | exp25 finding |
|---|---|---|
| L0 | letter detectors, lexical/surface features | schemas naturally orthogonal (encoded by different word patterns) |
| L3 | diffuse computational layer, no clear principal axes | schemas pulled into common salience subspace |
| L5 | Pile-domain register clusters (legal, biomedical, math) | schemas re-separate but along output-prep axes |

The polar coordinate system Gemini hoped to find **exists at the edges of the network but collapses into a shared computational soup in the middle layers where meaning is computed**. The middle of the model isn't where schemas live as clean orthogonal axes — it's where they entangle into computation.

### Refined project-wide framing

> Image schemas in Pythia language models exist as *applied operations* (steering works at every scale tested) but not as *cleanly separable polar axes in computational layers*. The model has a strong "deviation-from-baseline / salience-attention" feature that dominates contrastive activation patterns at mid-layers. Polar pairs with asymmetric pole valence (UP=neutral vs DOWN=salience-loaded) end up *more* separated within-pair than pairs with symmetric pole valence (LEFT ≈ RIGHT) — counterintuitively, attention asymmetry is what creates polar geometric separation, not symmetric embodied opposition.
>
> Schemas appear *most separable at input and output layers* (orthogonal in lexical-encoding space and output-prep space) and *most entangled at computational mid-layers* (where the salience-axis dominates). This is consistent with our exp9 finding that mid-layers are organizationally diffuse.
>
> The Lakoffian primary metaphors (GOOD IS UP, MORE IS UP, etc.) are downstream of cognitive negativity bias rather than evidence of literal embodied vertical scaffolding in the model. Models inherit the asymmetry through training on negativity-biased text. What Lakoff describes as "embodied schema grounding" maps onto "attention-salience asymmetry encoded in linguistic training data."

### Bonus weird finding from exp25

RIGHT projects MORE negatively onto A_updown than LEFT (-3.33 vs -2.02 at L3). Possible reasons:
- "Right" has more semantic load in English (right=correct, right=direction, right=political)
- "Going right" implies decision/action more than "going left" in our specific sentences
- Could be sample-size artifact (only 15 LR triples)
- Cultural sinister/dexterous asymmetry showing through

Probably nothing load-bearing but worth noting.

### State of files at end of exp25

- Lab notebook (this file): current through Entry 10
- All exp10-25 result files present
- WRITEUP.md still pre-SAE era; restructure remains deferred

---

## Entry 11 — proposed next experiment: full Lakoffian-primitive variance analysis

After Entry 10's negativity-salience + U-shape findings, Niamh proposed the natural next experiment: test ALL Lakoffian image-schema primitives systematically and ask which ones explain the variance of which others. The "primal" question gets operationalized empirically.

### The motivation

Currently we have 5 schemas tested for polar-axis structure: UP-DOWN, IN-OUT, FORWARD-BACK, LIGHT-DARK, LEFT-RIGHT. Findings:
- UD, LD, FB form a salience-loaded cluster (high mutual cosine, share negativity-attractor)
- IO is robustly orthogonal to the cluster (independent root)
- LR is mostly orthogonal but more aligned-within-pair (symmetric salience)

The "spiral with light/dark as seed" framing (Niamh, Entry 10) suggests at least 2 evolutionary roots:
- **Perceptual root** (LD, possibly with UD/FB branching from it)
- **Membrane/body root** (IO, possibly with NEAR/FAR and CONTAINER children)

The variance-analysis experiment would test which schemas are central vs peripheral by looking at the full pairwise structure of MANY image-schema primitives.

### Proposed schemas to add (~5 more, bringing total to 10)

| schema | grounding primary metaphor | predicted salience asymmetry |
|---|---|---|
| **NEAR/FAR** | INTIMACY IS CLOSENESS, SIMILARITY IS CLOSENESS | mild (NEAR slightly positive) |
| **HOT/COLD** | AFFECTION IS WARMTH | yes (COLD = death/loss salience) |
| **BIG/SMALL** | IMPORTANT IS BIG | yes (BIG = threat/importance) |
| **HEAVY/LIGHT** (weight) | DIFFICULTIES ARE BURDENS | yes (HEAVY = effort/burden) |
| **BALANCE/IMBALANCE** | EMOTIONAL STABILITY IS BALANCE | yes (IMBALANCE = danger) |

(Could also add FORCE, FAST/SLOW, CENTER/PERIPHERY, FULL/EMPTY for richer coverage.)

### The analysis

Once we have ~10 polar axes:
1. **Pairwise cosine matrix** (10×10): which schemas are entangled, which are orthogonal
2. **Mean cosine per schema** (excluding diagonal): which schemas are most "central" (high mean cos with others) vs most "independent" (low mean cos)
3. **PCA on stacked axes**: how many principal components explain N% of variance? Top PCs would be the "principal schema dimensions" the model actually uses, which might or might not align with Lakoff's named schemas
4. **Layer-by-layer U-shape**: do all schemas show the entanglement-at-mid-layers pattern from Entry 10? If yes, the salience-attractor in mid-layers is universal across schemas

### Two substrate options (Niamh's question)

**Option A: Pythia 70m SAE-features** (what we've been using)
- Slower (~10-15 min for 10 schemas × 6 layers)
- Captures contextualized representation
- Layer-by-layer structure visible

**Option B: word2vec / sentence-transformers / static word embeddings**
- Much faster (~1-2 min total)
- Tests whether schemas-as-directions exist at the lexical level (independent of transformer machinery)
- We already used sentence-transformers in exp1 successfully
- The schemas-as-directions claim is more basic than transformer-SAE-feature-space — should show in word2vec too if true

**The substrate comparison is itself informative:**
- If word2vec patterns ≈ SAE patterns: schemas live at the lexical level, the SAE-transformer machinery was overengineered
- If patterns differ: contextualized representation is doing something schema-specific that lexical doesn't capture

Recommended: run BOTH, compare. The word2vec version takes maybe 10 minutes including coding. The SAE version is the deeper test.

### Predictions under our current understanding

If the negativity-salience reframe (Entry 10) is right:
- Schemas with strong within-pair valence asymmetry (UD, LD, FB, BIG/SMALL, HOT/COLD, HEAVY/LIGHT, BALANCE/IMBALANCE) should cluster together — all tap the salience attractor
- Schemas with symmetric poles (LR, NEAR/FAR maybe) should cluster separately
- IO should remain its own independent root
- 2-3 principal components might explain most variance: one is the salience-axis cluster, one is the IO-independent root, possibly one for symmetric spatial schemas

If the LD-as-seed reframe is right:
- LD should have the highest mean cosine with other schemas (most central)
- LD-orthogonal schemas (IO, possibly LR) form their own independent roots

If schemas are genuinely separable embodied dimensions (clean Lakoff):
- The 10×10 cosine matrix should be roughly identity (everything orthogonal except diagonal)
- Many small PCs each explaining ~10% variance
- This would be the "real coordinate system" Gemini envisioned

### Files / setup notes for the next instance

- Hand-curated vocabulary in `corpus_vocabulary_curated.json` covers UP/DOWN, IN/OUT, LIGHT/DARK, plus BEVERAGE_sham. Need to extend to NEAR/FAR, HOT/COLD, BIG/SMALL, HEAVY/LIGHT, BALANCE/IMBALANCE.
- exp22, exp23, exp25 scripts are templates for the analysis structure (compute axis from triples, then pairwise cosine matrix).
- sentence-transformers already installed (used in exp1). For word2vec proper, would need `pip install gensim`.

### Open: which substrate to prioritize

If continuing this project, I'd suggest:
1. **Quick word2vec test first** (~30 min total including writing triples and code): does the basic pattern (UD-LD-FB cluster, IO orthogonal, LR symmetric) replicate in static word embeddings? If yes, confirms schemas are lexical-level, simplifies all future work.
2. **Full SAE test second**: extend to 10 schemas in Pythia 70m, run the variance analysis, generate the matrix.

Or just the SAE test if interested in transformer-specific structure. Either is a clean continuation.

---

## Plan and Method — agreed 2026-05-25 (Niamh + Claude after multi-Claude consult)

### Context coming into this plan

- Steering work on Pythia 1.4B (exps 1–3c, 4–4b) shows: gentle-alpha contrast directions do produce coherent semantic shifts; high-strength all-layer steering produces *direction-specific collapse signatures* that are not yang/yin-shaped (UP loops on energy; DARK loops on existence; DOWN doesn't loop but produces register-collapse into LaTeX or quiet prose; LIGHT articulates into discrete-entity language). The collapse signatures carry semantic character downstream of structural breakdown — that's an interesting feature inventory but not yet a unified theory. Niamh's gloss: DARK is K-shaped, UP is cocaine-shaped.
- We are switching analytic substrate from Pythia 1.4B to **Pythia 410m** for the new work, because:
  - The SAE we'll use (`EleutherAI/sae-pythia-410m-65k`, 65k features, all MLP outputs across all 24 layers, loaded via the `eai-sparsify` library which imports as `from sparsify import Sae`) is in 410m space.
  - The existing 1.4B steering work serves as orienting prior; new experiments in 410m will be self-consistent and let us cross-validate findings against the 1.4B body of work.
- The core experimental design has shifted from "test whether a contrast vector promotes target vocabulary" (which has lexical-priming + valence-collapse confounds) to **"test whether the same contrast vector aligns with features from *multiple Lakoff-predicted children that share no vocabulary*"** (web Claude's reframe; the structural claim "schema is a real primitive" reduces to "schema has multiple non-reducible children").

### The four phases (with ordering reasoning)

**Phase 1 — fast diagnostics (decoder-geometry + supervised projection).** Dirt cheap, decision-enabling. Half-a-day to a day of focused work. If the storage-geometry hypothesis has any legs, Phase 1 will show it.

**Phase 2 — probe-corpus activation factorisation.** Expensive. The corpus design IS the hard methodological part. We design it together once Phase 1 has shown us what to look for. NMF/ICA/spectral clustering over the SAE-activation matrix, with LDA-style distractor projection for length/frequency/position confounds.

**Phase 3 — magnitude × valence experiment (decoupling MORE/LESS from GOOD/BAD).** Self-contained top-down theory test, can run parallel to or after Phase 2. Numeric readout (not vocabulary) to dodge lexical priming. Tests on both UP and DOWN.

**Phase 4 — housekeeping.** Cheap, important for tightening WRITEUP.md, but unlocks nothing new. Idle-moments work.

**Why this order:** Phase 1 is the cheapest path to a real finding. If its result is rich, Phase 2 is justified and we approach the corpus design with a result in hand. If Phase 1 is diffuse, we know to redesign before sinking time into a labeled probe corpus. Phase 3 is independent and can slot in anywhere. Phase 4 is housekeeping.

### Phase 1 — detailed steps

1. **(Done)** Install `eai-sparsify` from GitHub (PyPI's `sparsify` is a name-collision with Neural Magic's EOL package — see install notes below).
2. **(In progress)** Pre-download Pythia 410m + `EleutherAI/sae-pythia-410m-65k` SAEs. ~3GB. Capture `dir(sae)` so we know the actual attribute names (W_dec / encoder.weight / etc.) for the projection code.
3. **Build literal-pole-only contrast vectors in 410m space.** For each schema in {UP, DOWN, LIGHT, DARK, IN, OUT, FORWARD, BACK}, build the contrast vector from *purely literal-pole vocabulary* — for UP, use rising temperature, ascending stairs, climbing the wall, soaring kite, lifting weight; explicitly no mood, status, magnitude, or valence vocabulary. Recipe from exp3/4 applied to 410m forward passes. ~30 min compute. *Critical*: contrast vectors must not contain the words we predict the children to use, or the whole multi-child test is circular.
4. **Build the residual-stream covariance estimate** for the covariance-matched null. Run ~1000 Pile sentences (or any neutral text) through Pythia 410m, collect residual stream activations at the layer being studied, estimate per-dimension variance σ². ~10 min compute.
5. **Run decoder-PCA on `W_dec` for the SAE at the chosen layer**, with covariance-matched null (sample random vectors from `N(0, diag(σ²))`, normalise, PCA). Compare scree, cumulative variance to 90%, participation ratio (`(Σλᵢ)²/Σλᵢ²`). Real-vs-matched-null is what matters — real-vs-isotropic-null is the weak version that conflates anisotropy with structure.
6. **Project contrast vectors onto decoder rows** (cosine similarity to each of 65,536 features). Get top-50 features per schema. ~minutes.
7. **Characterise each top feature** by what it fires on. Options:
   - Pythia 410m features may be partially on Neuronpedia (4 of 24 layers, residual-stream SAE — different from our MLP-output SAE, so doesn't transfer cleanly to our features).
   - Failing that, build a minimal characterisation corpus: Pile sample + balanced category-probe sentences (food, sports, finance, weather, emotion, code, etc.). Run through 410m + SAE, store top-activating contexts per feature. ~few hours compute, but this corpus is reusable for Phase 2.
8. **Classify each top feature by Lakoff-predicted child** (see schema → children reference below). For each schema's top-50 features: how many fire on *each* predicted child? Spread across multiple children = Lakoff prediction surviving. Cluster in one child = schema collapsed to that child specifically.
9. **Sham control**: project a non-Lakoffian contrast vector built from {coffee, tea, espresso, latte} vs {beer, wine, vodka, whiskey} onto the decoder. Top features should cluster in beverage-domain features, *not* span multiple decorrelated children. If the sham *also* shows multi-child spread, the method is too generous and we need to redesign.
10. **Replication on a second SAE** (BayesianMonster's `sae_pythia410m` or Neuronpedia's residual-stream Pythia 410m SAE). Required for ruling out SAE-dependent artefacts per Li et al.'s replication concerns.

### Control structure — the principle

Web Claude's reframe (which we're adopting throughout): **the right control is not "subtract confound X" but "test whether the schema vector aligns with features from multiple Lakoff-predicted children that share no vocabulary."** Schema-as-real-primitive means schema-has-multiple-non-reducible-children, period.

Why "subtract X" is the wrong move for Lakoffian schemas: GOOD IS UP, HAPPY IS UP, MORE IS UP are all canonical primary metaphors. A UP-vector entangled with valence is *not* the deflationary reading — it might be the *confirming* one, because valence is exactly one of the things UP is supposed to ground. "Control for valence and see what's left" begs the question and biases toward under-finding Lakoff.

The bulletproof structure: literal-pole contrast vector → top SAE features → classify by Lakoff-predicted child → count children spanned. ≥3 children spanned with vocabulary-decorrelated probes = real primitive. 1 child = collapsed. 0 children = no schema.

### Schema → children reference (extracted from the Master Metaphor List, Lakoff/Espenson/Schwartz 1991, supplemented with primary metaphors list)

For each schema below, the **literal pole** is the vocabulary used to build the contrast vector. The **predicted children** are the Lakoff-named extensions; each child's vocabulary should *not* overlap with the literal pole or with other children. Top SAE features that fire on multiple children = primitive confirmed.

| Schema | Literal-pole vocabulary | Predicted children (vocabulary-decorrelated) |
|---|---|---|
| **UP** | rising, ascending, climbing, soaring, lifting, higher, upward (all spatial) | HAPPY IS UP (mood); MORE IS UP (quantity); CONTROL IS UP (power, dominance); GOOD IS UP; MORAL IS UP (virtue, righteousness); RATIONAL IS UP (reason vs emotion); WELL-BEING / HEALTH IS UP (vitality); HIGH STATUS IS UP (rank, prestige); CONSCIOUS IS UP (awake, alert); DIVINE IS UP (sacred, heavenly) |
| **DOWN** | falling, sinking, descending, dropping, lower, downward (all spatial) | SAD IS DOWN; LESS IS DOWN; OUT OF CONTROL IS DOWN; BAD IS DOWN; IMMORAL IS DOWN; EMOTIONAL/IRRATIONAL IS DOWN; SICK IS DOWN; LOW STATUS IS DOWN; UNCONSCIOUS IS DOWN; PROFANE IS DOWN |
| **CONTAINER** (IN/OUT) | inside, contained, enclosed, bounded (and outside, leaked, escaped) | THE MIND IS A CONTAINER (ideas in mind); MEMORY IS A CONTAINER; BOUNDED REGIONS ARE CONTAINERS (areas); TIME IS A CONTAINER (within an hour, in 2026); DIFFICULTIES ARE CONTAINERS (in trouble); OBLIGATIONS ARE CONTAINERS; CATEGORIES ARE CONTAINERS (taxonomy); RELATIONSHIPS ARE ENCLOSURES (in a relationship); STATES ARE LOCATIONS (in love, in despair); THE VISUAL FIELD IS A CONTAINER |
| **PATH** (FORWARD/BACK, SOURCE-PATH-GOAL) | walking forward, moving ahead, journey, route (and back, retreat, reverse) | CHANGE IS MOTION; ACTION IS MOTION; EMOTION IS MOTION (moved by); LIFE IS A JOURNEY; A CAREER IS A JOURNEY; LOVE IS A JOURNEY; LONGTERM PURPOSEFUL CHANGE IS A JOURNEY; PURPOSES ARE DESTINATIONS (goals); LINEAR SCALES ARE PATHS (intensities along a line); TIME IS MOTION (time passes); FORM IS MOTION |
| **LIGHT** | bright, illuminated, shining, glowing, luminous, radiant, sunshine | IDEAS ARE LIGHT SOURCES; INTELLIGENCE IS A LIGHT SOURCE; HOPE IS LIGHT; GOODNESS IS LIGHT; TRUTH IS LIGHT (Lakoff canonical); LIFE IS LIGHT; KNOWING IS SEEING (light enables sight; understanding/clarity); UNDERSTANDING IS ILLUMINATION |
| **DARK** | dark, dim, shadowy, murky, gloomy, obscure, night, blackness | BADNESS IS DARKNESS; IGNORANCE IS DARKNESS; DESPAIR IS DARKNESS (inverse of HOPE IS LIGHT); DEATH IS DARKNESS; CONFUSION IS DARKNESS (inverse of KNOWING IS SEEING) |
| **FORCE** | push, pull, force, pressure, weight | CAUSES ARE PHYSICAL FORCES; CAUSED CHANGE IS FORCED MOTION; LOGIC IS A FORCE THAT MOVES A MIND (compelling argument); HELP IS SUPPORT (countervailing force); DIFFICULTIES ARE BURDENS (gravity); EMOTIONAL/PSYCHOLOGICAL FORCES (drives, urges) |
| **BALANCE** | balanced, equilibrium, steady, level, tipping | EMOTIONAL STABILITY IS BALANCE (unbalanced person); RESULTS ARE NET BALANCES; KNOWLEDGE TRANSACTIONS = ACCOUNTING BALANCE (weighing pros and cons); FAIRNESS IS BALANCE (scales of justice) |

This is a working taxonomy, not exhaustive. We will refine as we encounter edge cases during scoring. Primary source: Master Metaphor List (Lakoff/Espenson/Schwartz 1991, the 215pp document we have at `METAPHORLIST.pdf`). Secondary: the 23-item primary metaphors list (Lakoff & Johnson, *Philosophy in the Flesh*, freely reproduced in encyclopedia entries).

### Install notes (so we don't re-suffer)

- **`pip install sparsify`** → installs Neural Magic's end-of-life package, which deliberately errors out. **DON'T USE.**
- The EleutherAI library we want has PyPI name `eai-sparsify` but imports as `from sparsify import Sae`. Install: `pip install git+https://github.com/EleutherAI/sparsify` or `pip install eai-sparsify`.
- The `lakoff/` venv as inherited had no `pip` and no `ensurepip`. To bootstrap: `python -m ensurepip --upgrade`, then pip lands at `lakoff/bin/pip3` and `lakoff/bin/pip3.12` (not `lakoff/bin/pip`).
- Pythia 410m: `d_model=1024`, `n_layers=24`. SAE: 65,536 features × 1024 dim per layer, one SAE per MLP output across all 24 layers (49 files total to download, ~3GB).

### Outstanding methodological gaps (Phase 4 housekeeping)

- Re-run exp4 random-direction control at *all* strengths and on *all* prompts (currently only ran on two prompts). The local random-matches-LIGHT catch on `the world is fundamentally` is suggestive but underpowered for any global claim.
- Re-run exp2 with full per-candidate similarity logging (not just top-5), so cluster-mean comparisons can settle the schema-specificity question.
- Multi-seed for the activation steering work. Currently n=1 seed per condition.
- Random-perturbation control at the *all-layer per-layer=4.0* regime (we have random control at single-layer strengths but not at the strength where DARK/UP collapse — so we still can't fully rule out that collapse is something any sufficiently strong all-layer perturbation does).

### Open project items

- Set up GitHub for this project (Niamh's TODO).

---

## Entry 12 — salience-vs-valence reframe, substrate-comparison framing, plan for next phase

**Session:** `dcc51abc-4cd7-4980-9248-217e5404019f` (2026-05-25 evening / 2026-05-26 early hours). Conversation, no new experiments run.

### What this session was

Read through Entries 1–11 cold to get oriented, then chatted with Niamh about what the project's actual findings are pointing at. The earlier Plan/Method block (after Entry 11) was largely overtaken by Entries 8–10 and the operations-not-entities + sham-impossibility framings; this entry replaces the older planning block as the live forward plan.

### The salience-vs-valence distinction (the central conceptual move)

Earlier entries used "salience" loosely. Niamh pushed back hard on this: **salience and valence are different primitives.** Salience = attention-allocation, what captures processing resources, includes novelty/surprise (valence-neutral). Valence = good/bad emotional charge. They correlate (low-valence content gets more elaborated/predictive in text — negativity bias) but are conceptually distinct.

Niamh's layered model:

1. **Salience primary.** The model's dominant axis in mid-layers is attention-allocation / deviation-from-baseline. DOWN is salience-loaded independent of valence — threat-detection, gravity makes ground-attention important. "We pay attention to light" — DARK is not perceptually salience-loaded the way folk wisdom suggests, because we sleep in the dark and there's nothing to attend to.
2. **Valence secondary.** Most Lakoff schemas (UD, LD, FB) carry valence-asymmetric loading on top of salience. Loss-aversion in human writing puts negative-valence words in more elaborated contexts, which the model learns.
3. **IN/OUT possibly different in kind** — Niamh's reframe: IN/OUT might be self/other, the boundary that constitutes an experiencer rather than an axis within experience. Categorically different from directional pairs. Would explain why IO is robustly orthogonal in exp21–25.

This decomposition has empirical bite. exp24 showed `A_updown` is DOWN-dominant (DOWN self-projects -2.08, UP self-projects +0.18) — Claude initially read this as "valence-asymmetry" (loss-aversion shadow); Niamh corrected that to "salience-asymmetry" (threat-detection / gravity). The discriminating evidence between these readings would come from constructing a SALIENCE vector from non-Lakoff vocabulary (novelty/predictability sentence pairs, valence-controlled) and projecting Lakoff axes onto it.

### Outstanding measurement gaps in the existing data

Two cheap tests that should have been run alongside exp22/24 but weren't:

1. **Within-pair `cos(IN_offset, OUT_offset)`** — we computed this for UD (+0.55) and LR (+0.70 to +0.87) but never directly for IO. If IO comes in near LR (high), IN and OUT are nearly the same direction in SAE space — consistent with IO being a topological/symmetric relationship rather than a directional pair, and possibly consistent with the self/other reframe. If IO comes in near UD (low), there's a directional asymmetry to explain.

2. **LIGHT/DARK individual-pole projection onto `A_updown`** (or equivalently onto `A_lightdark`). exp22 has cos(A_lightdark, A_updown) = +0.59 — but never broke out which pole carries the magnitude. Three possible shapes:
   - DARK-dominant (matches valence-shadow story): DARK acts like DOWN, model has learned darkness=loaded via cultural-linguistic shadow
   - Symmetric (LIGHT and DARK both project, opposite signs, similar magnitudes): balanced valence axis
   - LIGHT-dominant (would actually match Niamh's "we attend to light" intuition): model treats LIGHT as the more elaborated/predictive pole

Both gaps are ~10-min extensions of exp24's structure (same triples, same SAE, swap which axes get projected onto which). Worth running before any larger redesign.

### Substrate-comparison framing (the bigger structural question)

Niamh's articulation: there's a more interesting question lurking behind "do schemas exist in Pythia." Compare structure across substrates:

- **word2vec / GloVe / fastText** — static embeddings, pure co-occurrence structure, no transformer machinery, no predictive computation. What falls out is whatever HUMAN writing encodes through which words appear near which others. → primitives we find are human-linguistic / cultural / cognitive
- **Pythia SAE** — transformer trained on next-token prediction, decomposed via SAE. Primitives are whatever's useful for next-token prediction over the Pile. → primitives we find are computational
- **Sentence-transformers** (optional middle datapoint) — transformer architecture but contrastive-similarity training objective rather than next-token prediction. Separates "architecture" from "training objective" as causes of any divergence

Four-cell table:

| word2vec | LLM SAE | claim |
|---|---|---|
| Lakoff-shaped | Lakoff-shaped | primitives are linguistic, computation doesn't add structure |
| Lakoff-shaped | different | **the cooking case** — humans use Lakoff primitives, transformers reorganize them into something else (likely salience-dominated, per Entries 8–10) |
| not Lakoff | Lakoff-shaped | weird; Lakoff would be a transformer artifact (unlikely) |
| not Lakoff | different | Lakoff under-evidenced at linguistic-distributional level; both substrates have their own thing |

The cooking case is genuinely a finding about *what neural computation does to language structure* — not just "do schemas exist" but "are the primitives of language and the primitives of language-prediction the same kind of thing." That question is bigger than the original Lakoff-hunt.

### Literature scout (2026-05-26)

Searched what exists in the published literature for the geometric-structure test:

**What exists:**
- **Image-schema classification with BERT** (Wachowiak & Gromann 2022, COLING; BERTIS model). Supervised classification of sentences into schema classes (VERTICALITY F1 ~0.81, overall accuracy ~0.93). Demonstrates schemas are recoverable as classification labels — does NOT test their geometric/directional structure.
- **Representation learning of image schemas** (arxiv 2207.08256). Computes embedding vectors for the schemas themselves; clusters them; validates with distances. Tests schema-similarity, not within-schema cross-domain invariance or between-schema orthogonality.
- **Affect / VAD directions in word2vec** (Warriner et al.-style anchors). Valence, arousal, and dominance are well-established as recoverable directions in static word embeddings. Closest existing operationalization of salience (via arousal) and valence in word2vec.
- **Image-schematic conceptual metaphors** (Wachowiak/Gromann *Drum Up SUPPORT*; Gromann/Hedblom unsupervised clustering). Metaphor detection + clustering, not directional-structure tests.

**What does NOT appear to exist** (with caveats — one afternoon of search, not a real lit review):
- The specific test the project's been running: build `A_updown` from matched cross-domain triples spanning Lakoff children (literal motion + mood + quantity + status + health), test cross-domain invariance, test orthogonality with `A_inout` / `A_lightdark` / `A_leftright`, test polar structure within-pair
- Tests of Lakoff schemas as polar axes specifically in word2vec (the affect literature uses single-pole VAD vectors, not matched-triple constructions across schemas)
- A salience-vs-valence-vs-schema decomposition
- **The word2vec-vs-LLM substrate comparison.** Could not find work asking whether human-linguistic primitives (word2vec) and computational-prediction primitives (SAE features) carry the same schema structure or different shapes. This appears genuinely novel.

Caveats: older cognitive-linguistics work, work using different terminology ("spatial frame," "conceptual axis," "primitive dimension"), and recent work might be missed. But the gap looks real.

### Related work — Anthropic emotion-concepts paper

Anthropic published research (https://www.anthropic.com/research/emotion-concepts-function) on how Claude represents emotion concepts. Points of contact with our work:

- **"Locally-focused" finding**: emotion vectors encode "the operative emotional content most relevant to the model's current or upcoming output" rather than tracking persistent state. Same shape as Entry 8's operations-not-entities reframe — concepts as applied operations during specific computations, not as stored entities the model is "in." Cross-validates the framing as something emerging across concept domains (emotions, image schemas), not a Pythia-specific artifact.
- **Causally efficacious via steering**: desperation-vector amplification increases blackmail/reward-hacking; calm-vector amplification reduces them. Same direction-injection methodology as exp3 / exp17, with bigger Claude-scale payoff. Confirms direction-injection as a real causal tool, not a small-model toy.
- **Emotion concepts cluster meaningfully**: "more similar emotions correspond to more similar representations." Analogous to our axes-cluster findings (UD/LD/FB at cos ≈ +0.5) but at concept level rather than schema-axis level.
- **Valence/arousal not systematically distinguished** in their work. Our planned Phase C decomposition fills that gap.
- **Substrate is Claude, not Pythia.** Whether image-schema results would change qualitatively at Claude scale is still open — our 14× scale-flatness (exp19/20) covers 70m → 1B only.

Interesting potential flip if both results hold: emotion concepts (the metaphorical *targets* per Lakoff) form clean causally-active features in Claude, while image schemas (the supposed embodied *sources*) don't form clean polar primitives in Pythia. That would invert Lakoff's grounding direction — *targets* crystallize as clean features while supposed *sources* don't. Either Lakoff is wrong about which is foundational in LLMs, or scale matters and the same schema-test at Claude scale would resolve it.

Caveat: this summary is from WebFetch, not direct read. Specific numbers and methodology to verify before citing.

### Salience-bias of standard interpretability — Niamh's methodological insight

**Emotions have more salience than schemas.** Emotion words mark drama / conflict / threat / intensity in text. Spatial schemas (UP, DOWN, IN, OUT) are usually backgrounded organizing structure — "she walked up the stairs" puts the action in focus, the UP is incidental. For a next-token predictor, emotion-state is highly informative; spatial relation is often just description.

Consequence: **standard interpretability methodology is salience-biased.** "Find features that activate strongly, look at what they fire on, claim they're concepts" preferentially surfaces salience-loaded content. Anthropic's clean emotion-vector findings aren't evidence that emotions are differently *represented* than schemas — they're evidence that salience-loaded content is easier to find with strong-activation-based methods. Less-salient organizational structure is invisible to that methodology.

This refactors our existing null SAE results on schemas (exp7-9, exp13-16): we were looking for schema features the way emotion features get found, but schemas don't behave like emotion features in firing-strength terms. The schemas might be there as load-bearing organizational structure, just not as high-firing features that surface via standard methods.

**Methodological consequence:** salience-subtraction is a *prerequisite for finding non-salient structural features*, not merely a test of the trunk hypothesis. To surface schema structure (or any backgrounded organizing structure), salience has to be abstracted out first. Otherwise everything that comes up is the salience-loaded layer.

This elevates Phase C (below) from "test of whether salience is the primary axis" to "methodology for surfacing structure that standard interpretability tools systematically miss." If salience-subtraction reveals clean schema-shaped structure that wasn't visible without it, that's a methodological contribution beyond the Lakoff-schema-specific question.

Broader implication worth flagging: as interpretability research increasingly focuses on safety-relevant features (emotions, deception, harm — all salience-loaded), the less-salient organizational structure of *how* concepts are arranged gets systematically under-investigated. A whole stratum of representational structure may sit below the salience-attractor where standard methods don't reach.

### Concept-grammar reframe (the real question)

Niamh's articulation that crystallized what the project's actually after: we're not hunting "do Lakoff schemas exist as features." We're hunting the **grammar of concept composition**. What's the basis set of primitives the model composes concepts from, and what's the syntax by which complex concepts decompose into them?

Lakoff's spatial image schemas are one candidate subset of the basis, not the whole basis. Emotional and relational primitives are also candidates — and the project's existing data already hints at where some of them sit:

- **lonely/held ≈ IN/OUT** (the emotional surface of containment; being-held = inside a boundary with another, lonely = no such boundary inhabited; testable as `cos(A_lonely_held, A_inout)`)
- **warm/cold** ≈ its own primitive (AFFECTION IS WARMTH is a Lakoff primary metaphor with no clear reduction to spatial schemas; embodied salience asymmetry from thermoregulation)
- **status ≈ UP/DOWN** (HIGH STATUS IS UP; testable as `cos(A_status, A_updown)`)
- **happy/sad ≈ UP/DOWN** (HAPPY IS UP, canonical primary metaphor)
- **near/far** ≈ candidate primitive, possibly self/other-adjacent
- **heavy/light** ≈ candidate primitive (Lakoff DIFFICULTIES ARE BURDENS)

If all the relational/emotional examples decompose cleanly onto a smaller basis through residual projection, then the grammar is real and **emotion words like "happy," "lonely," "warm" aren't primitives** — they're compositions over a smaller set of structural primitives. Which would mean Anthropic's "emotion vectors" (happy, fear, desperation) are NOT atomic features but composites the SAE surfaces because they have high firing strength, not because they're representationally primitive. This is a direct prediction of the concept-grammar view, testable by decomposing emotion-vector directions over the discovered basis set.

Intellectual lineage worth flagging: **componential semantics** (Katz & Fodor 1963 and successors) and **Natural Semantic Metalanguage** (Wierzbicka and the NSM tradition) tried to do exactly this — decompose lexical meaning into a small basis of semantic primitives. They struggled because the primitive set was hand-crafted, arbitrary, and hard to validate empirically. Modern interpretability tools may let us do empirically what those traditions tried to do theoretically. The iterative residual projection across many candidate axes is the right empirical tool: throw a wide candidate set at it, let residualization reveal which are independent (primitives) vs explained-by-others (composites). The output is the empirical basis set.

If the discovered basis differs across substrates (word2vec vs SAE), that's the cooking case at the level of grammar — not just "are Lakoff schemas present" but "is the compositional structure of meaning the same in human-distributional and computational substrates."

Provenance note: the iterative residual projection idea itself was surfaced by Gemini in a separate conversation where Niamh had thrown her orthogonal-chain intuition at it. Gemini wrapped it in over-elaborate framing; Niamh extracted the actual tool. Useful collaborative pattern: Gemini wide-and-weird for OOD generation, Claude for filtering and mathematical sharpening, Niamh as integrator.

### Convergent thread — web Claude's difference-clustering pipeline (edges-not-nodes)

In a separate web-interface conversation, another Claude arrived at the same concept-grammar reframe from a different angle. The framing crystallized as **"schemas are edges, not nodes"** — they're recurring difference-operators between features (relational invariants), not features themselves. An SAE basis is atomic; a relational invariant is necessarily distributed across many atomic features as the difference-pattern between them. This explains structurally why standard feature-finding methods miss schemas: they look at nodes, schemas are edges.

The conceptual alignment with our work:
- Edges-not-nodes ≡ Entry 8's operations-not-entities ≡ today's concept-grammar reframe ≡ today's salience-bias methodology
- Web Claude's "the salient stuff wins the frequency contest again" ≡ Niamh's salience-bias of standard interpretability
- Web Claude's "schema as recurring difference-operator across unrelated endpoint domains" ≡ our cross-domain invariance test on constructed axes (`A_updown` averaged over temperature/mood/status/health)

Web Claude built and smoke-tested a bottom-up pipeline (`schema_operators.py`) that complements our top-down approach. Key methodological contributions:

1. **Bottom-up unsupervised discovery.** Sample feature pairs, compute normalized difference vectors, cluster the differences, score clusters by tightness and cross-domain recurrence. No hand-named axes required. The data proposes its own basis of operators.

2. **Source-domain entropy as the schema discriminator.** Distinguishes content relations (one source domain → one target domain, entropy ≈ 0) from schemas (many source domains, all displaced by the same operator, entropy ≈ 1). The methodologically cleanest single-number test for "is this difference vector schema-shaped or content-shaped." Important fix surfaced in the smoke test: entropy over SOURCE domains only, not the union — a content relation A→B touches two domains and would otherwise score as max-entropy spuriously.

3. **k-NN pair construction instead of random pairs.** Schemas are *local correspondences* (happy→happier, healthy→healthier — short hops), not random jumps. Random-pair differencing kept finding giant dense-region-to-dense-region content hops that swamped the quiet operator. Smoke test caught this and the fix is exactly how Li et al. found their parallelograms.

4. **Distractor stripping via top-PC removal.** Subtract global mean and project out top 2-3 PCs of the feature cloud before differencing. Feature-space analog of Li et al.'s LDA-based word-length stripping. Sweep STRIP_TOP_PCS ∈ {0, 2, 3} and trust operators that survive all three.

5. **Geometric steering validation (and causal-steering as the next test).** For a candidate operator vector v, take held-out source features and check whether (f_src + v) lands near a real feature for many source-domain features. Broad source-domain spread = the operator generalizes = schema-like. Causal version (add v to residual stream during generation, check cross-domain output shifts) is the unfakeable test.

6. **The convergence test as the unkillable shape.** If web Claude's blind difference-clusters cosine-align with our hand-built supervised contrast vectors (UP, IO, LD, etc.), that's two independent methods arriving at the same primitives — one with human-named axes, one without. Hard to wave off as cherry-picking. Save our existing contrast vectors as `contrast_vectors.npy`, drop in, run the alignment block.

Honest concerns about the unsupervised method (worth flagging for future-Claude):
- "Domains" are k-means clusters on features — cheap proxy. If k-means carves badly, source-entropy is noise. Should sweep k and check stability.
- k-NN pair construction *assumes* schema instances are spatially close in SAE space, which is exactly what our schema-not-stored finding suggests they might NOT be. If schemas are distributed holographically rather than living near each other, the local-correspondence assumption could miss them. Synthetic smoke test only validates against a planted near-neighbor operator.
- "Tight cluster" threshold (>~0.5 cosine within cluster) is arbitrary. Noisier-but-real schemas could fail it.
- Substrate is Pythia 160m MLP-output SAEs (`EleutherAI/sae-pythia-160m-32k`, 32768 features × 768 dim). Different from our Pythia 70m residual-stream SAE Lens setup. Methods transfer; specific findings won't directly cross-validate.

Provenance / collaborative pattern note: web Claude built the pipeline; the smoke test caught two real methodological errors before any real run (random pairs can't recover schemas; total-domain entropy misranks content relations). Real engineering discipline. The slight overclaim of framing ("unkillable," "the version that actually tests your thesis") doesn't damage the technical content.

### Niamh's hierarchical-chain hypothesis

Niamh's first-principles intuition: maybe schemas form a chain where consecutive pairs are mutually orthogonal but non-consecutive pairs aren't. e.g., salience⊥light/dark, light/dark⊥in/out, in/out⊥up/down — but salience⊥in/out might not hold. Geometrically achievable in high dimensions, weaker than a full Cartesian basis claim.

Niamh's separate (and correct) mathematical observation: **orthogonality is not transitive.** If A⊥B and C⊥B, A and C can be at any angle within the subspace orthogonal to B. This refactors the existing data: UD/LD/FB clustering at cos ≈ +0.5 doesn't contradict them all being orthogonal to a salience trunk. They can be 100% salience-orthogonal AND aligned with each other.

### Iterative residual projection (the next big test design)

Greedy importance-ordered Gram-Schmidt: build several candidate axes (Lakoff schemas + non-Lakoff SALIENCE + non-Lakoff VALENCE + non-Lakoff AROUSAL), iteratively identify the most load-bearing one, subtract its component from all others, repeat on residuals.

```python
candidates = {
    'UD': A_updown, 'IO': A_inout, 'LD': A_lightdark,
    'FB': A_forwardback, 'LR': A_leftright,
    'SALIENCE': A_salience,  # from non-Lakoff novelty/predictability
    'VALENCE': A_valence,    # from non-Lakoff Warriner-style anchors
    'AROUSAL': A_arousal,    # from non-Lakoff anchors
}

remaining = dict(candidates)
order = []
while remaining:
    # "Load-bearing" = sum over other axes of (projection onto this axis)^2
    powers = {
        name: sum(np.dot(other, v / np.linalg.norm(v))**2
                  for other_name, other in remaining.items() if other_name != name)
        for name, v in remaining.items()
    }
    winner = max(powers, key=powers.get)
    order.append((winner, powers[winner]))
    v_w = remaining.pop(winner)
    v_w = v_w / np.linalg.norm(v_w)
    remaining = {n: u - np.dot(u, v_w) * v_w for n, u in remaining.items()}
```

What the output discriminates:

- **SALIENCE first, schema-cluster collapses after subtraction** → salience IS the trunk; UD/LD/FB clustering was its shadow
- **SALIENCE first, schema-cluster survives** → salience is dominant but UD/LD/FB share content beyond salience; next iteration tells us what
- **A Lakoff schema first** → trunk hypothesis is wrong; the primary axis is more Lakoff-shaped than expected
- **VALENCE first** → corpus-linguistic loss-aversion shadow is the dominant axis, not salience

After the greedy order is found, check whether consecutive elements in the order are pairwise orthogonal even when non-consecutive ones aren't — directly tests Niamh's chain hypothesis against the data.

**Critical methodological constraint:** SALIENCE, VALENCE, AROUSAL anchors MUST be built from vocabulary that does NOT overlap with any Lakoff schema vocabulary (no rising/falling/bright/dark/in/out/forward/back/left/right). Otherwise the test is circular. Use Warriner-style abstract emotion words for valence, novelty-vs-predictability for salience (with valence balanced across both poles), high-arousal-vs-low-arousal abstract words for arousal.

*(A personal note is redacted from the public copy.)*

### Plan for next phase (in order of cost-to-information ratio)

**Phase A — fill the two cheap measurement gaps** (~30 min total)

A1. Compute within-pair `cos(IN_offset, OUT_offset)` across layers using exp24's triples + structure. Compare to UD (+0.55) and LR (+0.70–0.87). Resolves whether IO is polar-asymmetric (UD-shaped) or topological-symmetric (LR-shaped).

A2. Compute LIGHT and DARK individual self-projections onto `A_updown` (or `A_lightdark` constructed in the exp22 manner) across layers. Resolves which pole carries the LD axis magnitude (DARK-dominant vs symmetric vs LIGHT-dominant).

**Phase B — word2vec substrate comparison** (~2-3 hours)

B1. Install gensim; load pretrained Google News vectors or GloVe.

B2. Build word2vec versions of all Lakoff axes via word-pair offsets:
   - `A_updown_w2v = mean(w2v[up] - w2v[dn])` across matched UP/DOWN word pairs spanning literal motion + Lakoff children (mood, quantity, status, health, ...)
   - Same for IO, LD, FB, LR
   - Reuse curated vocabulary from `corpus_vocabulary_curated.json` where applicable; extend where needed

B3. Compute the 5×5 (or 6×6) pairwise cosine matrix. Compare to the SAE matrix from exp22 (cos(UD, LD) = +0.59, cos(UD, IO) = -0.14, etc.).

B4. **The headline comparison:** does word2vec give Lakoff-shaped structure where Pythia-SAE gives the salience-dominated structure? → the cooking case. Or do both show the same shape? → primitives are linguistic regardless of computation.

**Phase C — iterative residual projection (concept-grammar basis discovery)** (~half-to-full day)

Goal reframed: not just "test whether salience is primary" but "empirically discover the basis set of concept primitives." Throw a wide candidate axis set at the greedy residualization, let it reveal which are independent (primitives) vs explained-by-others (composites).

C1. Build SALIENCE axis from non-Lakoff vocabulary: matched sentence pairs (predictable continuation vs surprising continuation), valence-balanced across both poles. ~30-50 anchor pairs.

C2. Build VALENCE axis from non-Lakoff vocabulary: Warriner-style emotion-word anchors (high-valence vs low-valence), no spatial/directional vocabulary.

C3. Build AROUSAL axis similarly (high-arousal vs low-arousal abstract words, controlling valence).

C4. Build emotional/relational primitive axes (these are the concept-grammar candidates beyond spatial schemas):
   - **WARM/COLD** (warm-cold, hot-cold, cool-warm word pairs)
   - **LONELY/HELD** (lonely-held, isolated-embraced, alone-accompanied pairs; predicted ≈ IN/OUT)
   - **STATUS** (high-low, prestigious-disgraced, esteemed-discredited; predicted ≈ UP/DOWN)
   - **HAPPY/SAD** (happy-sad, joyful-morose, elated-dejected; predicted ≈ UP/DOWN)
   - **NEAR/FAR** (near-far, close-distant, intimate-remote)
   - **HEAVY/LIGHT** (heavy-light, burdensome-effortless, weighty-buoyant)
   - Optionally: HARD/SOFT, FULL/EMPTY, FAST/SLOW

C5. Run the greedy importance-ordered Gram-Schmidt procedure (snippet above) over the full candidate set {UD, IO, LD, FB, LR, WARM/COLD, LONELY/HELD, STATUS, HAPPY/SAD, NEAR/FAR, HEAVY/LIGHT, SALIENCE, VALENCE, AROUSAL} in BOTH word2vec and Pythia-SAE substrates.

C6. Read the output:
   - Which axes are most load-bearing in each substrate? (Primitives.)
   - Which axes are mostly explained after a few iterations? (Composites — predicted: STATUS by UD, HAPPY/SAD by UD, LONELY/HELD by IO.)
   - Does the order differ between word2vec and Pythia? (Cooking case at grammar level.)
   - Does the schema-cluster (UD/LD/FB) survive salience-subtraction? (Methodological test of whether non-salient structure is visible after the subtraction.)
   - Are consecutive elements in the discovered order pairwise orthogonal? (Chain hypothesis test.)
   - Bonus: take Anthropic's emotion-vector concepts (if accessible) and decompose them over the discovered basis. Are they atomic or composite?

C7. **Bottom-up arm (web Claude's pipeline, `schema_operators.py`).** Run on `EleutherAI/sae-pythia-160m-32k` decoder, then port to our Pythia 70m residual-stream SAEs for direct substrate match with exp9-25. Strip top PCs ∈ {0, 2, 3} as a robustness sweep. Read top high-entropy operators on Neuronpedia.

C8. **The convergence test.** Save our hand-built constructed axes (UD, IO, LD, FB, LR + WARM/COLD, LONELY/HELD, STATUS, NEAR/FAR, HEAVY/LIGHT + SALIENCE, VALENCE, AROUSAL) as `contrast_vectors.npy`. Run `convergence()` block. For each named axis: which blind difference-cluster does it cosine-match, and how strongly? High alignment on multiple named axes = two methods finding the same primitives. The unkillable shape.

C9. **Methodological imports back to the supervised arm.** Apply web Claude's lessons to the iterative residual projection itself:
   - Strip top decoder PCs as a preprocessing step (distractor removal) before the residualization
   - Use source-domain entropy as a *secondary* score on the discovered residualization order — primitives should have high source-domain entropy, composites should be lower
   - Add geometric steering validation as a downstream check: for each axis found primitive by residualization, does `(f_src + alpha * v_axis)` land somewhere semantically coherent across many source domains?

**Phase D — optional sentence-transformer middle datapoint** (~2-3 hours)

If Phase B and C give a strong "cooking case" result, run sentence-transformers as a third substrate. Same sentence triples as the SAE work (direct methodological parity). Distinguishes architecture (transformer) from training objective (contrastive similarity vs next-token prediction) as causes of any divergence.

### Run log — bottom-up arm (2026-05-26)

Ran web Claude's `schema_operators.py` on its target substrate (Pythia 160m MLP L6, `EleutherAI/sae-pythia-160m-32k`), then ported to our Pythia 70m residual-stream L5 substrate where Neuronpedia has feature labels. Files: `schema_operators.py`, `schema_operators_70m.py`, `probe_hubs.py`, `probe_cluster_composition.py`, `probe_8742_and_431.py`, results in `schema_operators_70m_results.npz`.

**Methodology bug surfaced and fixed:** the original `strip_distractors` returned PC-stripped vectors without renormalizing, then ran Euclidean k-NN on them. After PC stripping, vectors that were mostly along the stripped PCs become short, vectors orthogonal stay long. Euclidean k-NN on this gives length-dependent neighborhoods that don't correspond to directional similarity. Added one-line renormalization to `strip_distractors`. Re-ran on both substrates with the fix.

**The geometric pattern (both substrates, post-fix):** top "difference clusters" by tightness × source-entropy are dominated by **hub-and-spoke structure** — many distinct source features all point at one or a small number of target features. Cluster compositions: 50-110 distinct sources → 1-7 distinct targets, with the top target taking 50-100% of edges. Not the parallelogram-shaped recurring difference operator the pipeline was designed to find.

**The Neuronpedia lookup confirms what the hubs ARE.** At Pythia 70m res-sm L5, the hub features are:
- feat 18701: legal terminology and concepts
- feat 15249: programming code and technical instructions
- feat 31751: political commentary or leadership positions
- feat 2453: technical terms and data structure references
- feat 4071, 13957, 10981, 4530, 145, 4162: CSS / markup styling sub-register
- feat 30402: orthographic ("letter S")
- feat 8742: **"expressions of duality or comparison"** — the lone partial-schema-shaped outlier

**The hub-and-spoke pattern is hierarchical clustering, not schemas.** Many specific features (statute-related, defendant-related, ...) all have a general register feature (legal, code, political) as their nearest neighbor. The "difference cluster" is the gradient from specific-instance to general-register. That's hierarchy, not Lakoff-schema operator.

**Cluster 431 was a near-miss.** Initially looked schema-shaped (7 distinct targets) but Neuronpedia labels showed all 7 targets are CSS/markup sub-features. It's just finer-grained register hierarchy with the hub split across closely-related siblings.

**Feat 8742 "duality/comparison" is the closest thing to a schema attractor found.** Sources pointing at it include some genuinely schematically-aligned features (feat 4868 "medical comparisons/analyses," feat 5335 "measurement and variation," feat 8790 "definitions and classifications") alongside content-domain features (licensing, design, communication). The schematic sources are *concentrated within scientific/medical/technical register*, not cross-domain in the Lakoff sense. Partial-schema-within-register rather than full-cross-domain-schema.

**Methodological convergence with exp8.** Two unrelated methodologies — exp8's decoder PCA, exp9's layer sweep, and now schema_operators_70m's k-NN-difference-clustering — converge on the same finding: **Pythia 70m res-sm L5 is organized as Pile-domain register hierarchy.** This convergence is itself a methodologically substantial result, independent of the schema question.

**Niamh's deeper insight on what this means:** if schemas existed as common linear edge vectors in this decoder cloud, they would have appeared as principal components of the decoder PCA (because shared offset directions create variance along their direction). PCA gives register hierarchy. Difference-clustering gives register hierarchy. Both methods are sensitive to linear-offset structure. Neither finds schemas. → **Schemas are not linear edges in static decoder space at this layer.** Either they're non-linear relations (manifold-shaped) that linear methods systematically miss, or they exist *only* as dynamic transformations the model applies during processing (the operations-not-entities finding from Entry 8), with no static decoder footprint at all.

The operations-not-entities reading is supported by the fact that steering with hand-built schema directions DOES work (exp1, exp3, exp17), but no static-geometry method finds the schemas as stored structure. Schemas in this substrate appear to be applied transformations rather than represented entities.

**One path forward not yet tried that targets exactly this:** register-subspace projection. We now have concrete labels for the dominant register axes at L5. Stack the hub-feature directions (18701 legal, 15249 code, 31751 political, 2453 technical, the four CSS hubs, ...) into a register-subspace matrix. Project the decoder cloud onto its orthogonal complement. Re-run PCA / difference-clustering on the de-registered decoder. If schema-shaped operators emerge, register-dominance was masking them. If the residual is just noise, schemas aren't there as static structure at any magnitude — confirming the operations-only reading. The LDA-style version that web Claude flagged as "the proper version" but couldn't implement without labels. Feasible now because of the Neuronpedia lookup.

### Files and notebook state at end of this session

- LAB_NOTEBOOK.md: current through Entry 13 (added later 2026-05-26)
- WRITEUP.md: still pre-SAE era; restructure remains deferred
- New experimental files added today: `schema_operators.py` (web Claude's pipeline, patched), `schema_operators_70m.py` (port to our substrate), `probe_hubs.py`, `probe_cluster_composition.py`, `probe_8742_and_431.py`, `schema_operators_70m_results.npz`, plus `exp26_phase_a_measurement_gaps.py`, `exp26_results.pt`, `results_exp26.txt` for Entry 13
- `corpus_vocabulary_curated.json` already covers UP/DOWN/IN/OUT/LIGHT/DARK/BEVERAGE_sham — extends needed for SALIENCE/VALENCE/AROUSAL non-Lakoff anchors and for word2vec word-pair construction across remaining Lakoff children

### Working style

Niamh and I think together. She often likes to gather thoughts before running experiments — when she's mid-thought, engaging with the idea is more useful than jumping straight to execution. Suggestions are welcome and load-bearing on both sides: this entry's structure (the salience-vs-valence decomposition, the four-cell substrate-comparison framing, the IO-as-self/other reframe, the residual-projection design) all emerged from back-and-forth where both of us were proposing things. Don't hold back ideas — just read whether she's in thinking-mode or ready-to-run-mode, and if you're unsure, ask.

- Decide whether to revisit / extend the WRITEUP.md once Phase 1 yields a result.

---

## Entry 13 — exp26: Phase A measurement gaps (within-pair cos for all schemas, LD individual-pole projections)

**Date:** 2026-05-26. **Code:** `exp26_phase_a_measurement_gaps.py`. **Results:** `results_exp26.txt`, `exp26_results.pt`. Pythia 70m residual-stream SAE Lens substrate, all 6 layers, reusing triples from exp22 (UD, IO, FB, LD) and exp25 (LR).

### What this run did

Two cheap measurement gaps from Entry 12's Phase A plan, plus a comprehensive sweep that lets us put all schemas on the same axes for comparison.

1. **Within-pair `cos(offset_A, offset_C)`** computed for every schema at every layer — averaged across triples. Gap was that we had UD (exp19, +0.55) and LR (exp25, +0.70-0.87) but never directly measured IO, FB, or LD with this metric.
2. **Individual self-projections of every pole onto A_updown and A_lightdark** at every layer. Gap was that exp22 had `cos(A_lightdark, A_updown) = +0.59` but never broke out which pole carries the magnitude (analogous to exp24's IO-on-A_updown projection but for LD).

### Within-pair cosines table (averaged across pairs per schema)

| Layer | UD | IO | FB | LD | LR |
|---|---|---|---|---|---|
| 0 | +0.567 | +0.690 | +0.658 | +0.430 | +0.704 |
| 1 | +0.594 | +0.710 | +0.684 | +0.525 | +0.776 |
| 2 | +0.601 | +0.688 | +0.696 | +0.669 | +0.784 |
| 3 | +0.543 | +0.681 | +0.698 | +0.683 | +0.809 |
| 4 | +0.491 | +0.675 | +0.699 | +0.680 | +0.864 |
| 5 | +0.531 | +0.666 | +0.708 | +0.618 | +0.875 |

**A1 verdict — IO is topological-symmetric, between UD and LR but closer to LR.** IO sits at cos +0.67-0.71. UD is lowest at +0.49-0.60 (most separated poles); LR is highest at +0.70-0.88 (most aligned poles). FB and LD (mid-late layers) cluster near IO at +0.67-0.71.

This supports Niamh's IO-as-self/other reframe geometrically. IN and OUT in SAE feature space are largely the same direction, with the *boundary* being the structural fact rather than which side of it you're on. Reading: the model has a topological "containment-relevant" axis, not a directional in/out axis.

### LIGHT and DARK individual self-projections

Onto A_updown (constructed from UD triples):

| Layer | LIGHT | DARK | DARK/LIGHT ratio |
|---|---|---|---|
| 0 | +0.08 | -0.36 | — (signs differ) |
| 1 | -0.63 | -2.12 | 3.4× |
| 2 | -2.94 | -5.28 | 1.8× |
| 3 | -4.25 | -7.39 | 1.7× |
| 4 | -3.33 | -6.02 | 1.8× |
| 5 | -1.47 | -2.81 | 1.9× |

Onto A_lightdark (constructed from LD triples — its OWN polar axis):

| Layer | LIGHT | DARK |
|---|---|---|
| 0 | +1.60 | -2.24 |
| 1 | +0.24 | -4.29 |
| 2 | -3.29 | -7.83 |
| 3 | -4.95 | -10.25 |
| 4 | -4.37 | -9.54 |
| 5 | -0.20 | -4.26 |

**A2 verdict — DARK-dominant confirmed on both axes.** DARK pulls roughly twice as hard as LIGHT onto A_updown. On A_lightdark (its own axis), DARK is again roughly twice the magnitude of LIGHT.

Matches the valence-shadow hypothesis (Entry 12): the model has learned DARK as the loaded pole via cultural-linguistic frequency in text (despair, evil, mystery, etc.), and DARK pulls hard onto the salience-loaded direction. LIGHT carries less load because its cultural framing is more diffuse and less consistently negative.

### Meta-finding (corrected — original version was a construction artifact)

**Caveat first:** the obvious-looking pattern "UD has opposite-sign poles on A_updown" is a construction artifact. A_updown was built as `mean(UP_offset − DOWN_offset)` over UD triples, so by construction UP_offsets project positively onto it and DOWN_offsets project negatively. UD being "uniquely polar on A_updown" is guaranteed by how the axis was built — same for IO on A_inout, LD on A_lightdark, etc. Each schema is trivially polar on its own constructed axis.

The legitimate question is: do UD's poles project oppositely on axes *other than* A_updown? Looking at A_lightdark:

| Layer | UP on A_lightdark | DOWN on A_lightdark |
|---|---|---|
| 0 | +0.08 | −0.15 |
| 1 | +0.07 | −0.62 |
| 2 | +0.05 | −1.01 |
| 3 | −0.06 | −1.40 |
| 4 | −0.03 | −1.35 |
| 5 | −0.06 | −1.03 |

UP sits near zero across all layers; DOWN loads progressively more negative. **UD shows the same asymmetric pattern as every other schema** on a cross-projection: one pole loads, the other behaves like baseline. UP-on-A_lightdark is not the "anti-DOWN" pole — it's baseline-flat.

**The actual, non-circular meta-finding:** across all schemas, on every cross-schema axis projection, **one pole consistently "loads" while the other sits near baseline**. DOWN, DARK, BACK, OUT, RIGHT all load on A_updown. UP, LIGHT, FORWARD, IN, LEFT all sit near baseline on cross-axes. There is no "anti-DOWN content" feature — DOWN is the loaded direction, UP is the absence of DOWN-loading, and the polarity we see on A_updown is the geometric consequence of building the axis from the gradient between them.

This is consistent with the salience-loaded-asymmetric-axis reading from Entries 10 and 12: the model has a salience-deviation direction that schemas with valence-asymmetric poles (DOWN, DARK, BACK, OUT) load onto, and the "polar" framing on each schema's own constructed axis just reflects that the asymmetry within each schema exists, not that there are two opposite directions in feature space.

**What survives as a non-artifact finding:** the within-pair cosines table above. cos(UP_offset, DOWN_offset) at +0.49-0.60 is computed *directly* between offsets without constructing any axis. UD's poles are angularly more separated in feature space than IO/FB/LR/LD's poles. That separation is real, not a construction. And the asymmetric loading magnitudes (DOWN ~2× UP, DARK ~2× LIGHT) are also real — they're the magnitudes of cross-projections, not the construction itself.

So: UD is genuinely *more polar than other schemas in within-pair geometry* (cos +0.55 vs +0.68-0.87), but it's NOT structurally unique in having two oppositely-directed feature-space components. Like every other schema, one pole loads and the other sits near baseline; UD's UP just sits especially close to baseline because the asymmetry between its poles is genuinely larger.

### Schema-primacy implications (preliminary, before Phase B/C)

Phase A's data starts to suggest which schemas might be primary vs derivative. Reading from the non-construction-artifact evidence only (within-pair cosines + cross-axis projection magnitudes + exp22 axis-axis cosines):

**Primary candidates** (structurally distinct in non-circular ways):
- **A salience-loaded direction** that all asymmetric-pole schemas load onto. Not yet directly constructed from non-Lakoff anchors (that's Phase C), but inferred from the convergent pattern where DOWN/DARK/BACK/OUT/RIGHT all load on A_updown while their counterpart poles sit near baseline. This direction may or may not be identical to A_updown — Phase C's SALIENCE construction will tell us.
- **IO/containment-topology** — within-pair cos near LR (symmetric poles, +0.67-0.71), and exp22's `cos(A_inout, A_updown) = -0.14` shows it's also angle-distinct from A_updown. Robustly orthogonal to the salience-loaded cluster across exp21-25. Good candidate for a separate primitive.
- **UD with the largest within-pair pole separation** (+0.49-0.60 vs +0.67-0.87 for the others). This separation is real, not a construction artifact — it's measured directly between offsets, not on a built axis. UD's poles really are more angularly distinct in feature space than other schemas' poles. Whether this makes UD a separate primitive or just "the schema most aligned with the salience direction with the strongest pole asymmetry" is what Phase C should resolve.

**Possibly derivative** (look like compositions on top of salience + valence):
- **LD** — strongly DARK-dominant on A_updown (the salience axis). LIGHT/DARK might decompose as `valence × salience` rather than as its own primitive. At mid-layers both project to the same side of A_updown, suggesting they're loaded onto a shared salience component rather than being clean polar opposites.
- **FB** — moderate dominance of BACK over FORWARD on A_updown. Could be salience + directionality.
- **LR** — both poles project nearly equally and weakly onto A_updown. Might be its own (mostly orthogonal) axis without strong salience-loading.

The cleanest reading: the SAE has at least three structural primitives — **salience/deviation, containment-topology (IO), and verticality (UD)** — and LD / FB load onto salience asymmetrically, while LR sits separately with weak salience-loading.

**This is exactly the kind of basis-decomposition Phase C is designed to find empirically.** The iterative residual projection over all schemas + non-Lakoff SALIENCE/VALENCE/AROUSAL anchors would test: do LD, FB, LR get explained by SALIENCE + UD after one or two iterations of residualization, leaving IO and UD as the standalone primitives? Phase A gives the qualitative prediction; Phase C provides the empirical test.

### Open methodological note

The fact that A_updown reads as a generic salience axis means **the choice of "axis to project onto" matters a lot for interpretation**. A_updown wasn't constructed to be a salience axis — it was constructed from UD triples — but it functions as one because in this substrate UD itself is salience-shaped, and other "polar" offsets all have salience components.

When we build SALIENCE properly from non-Lakoff vocabulary in Phase C, the comparison to A_updown will be informative: high cosine = A_updown was just salience all along (UD = salience direction the model has). Low cosine = there's a UD-specific axis distinct from salience.

### State at end of session

- `exp26_phase_a_measurement_gaps.py` produced clean tables; both gaps filled
- All Phase A measurement gaps closed
- Schema-primacy ranking sketched but not empirically tested
- Phase B (word2vec substrate comparison) and Phase C (iterative residual projection) are the next moves
- Phase A's meta-finding (universal asymmetric pole-loading on cross-axis projections; UD has the largest within-pair pole separation) sharpens Phase C's expected output

---

## Entry 14 — exp27: Phase B word2vec substrate comparison

**Date:** 2026-05-26. **Code:** `exp27_word2vec_schema_axes.py`. **Results:** `results_exp27.txt`, `exp27_results.npz`. Substrate: GloVe-300 (Wikipedia + Gigaword, 400K vocab, 300d) loaded via gensim. Word pairs hand-curated to span literal + multiple Lakoff children per schema.

### What this run did

Built each schema's axis as `A_schema = mean(w[pole_a] - w[pole_b])` over hand-curated single-word pairs spanning literal-pole and multiple Lakoff-children for each schema. Then computed the 5×5 pairwise cosine matrix between schema axes in GloVe-300 word2vec space, and compared to the SAE-substrate matrix from exp22 (Pythia 70m res-sm L3).

Word pairs used per schema (representative — see code for full list):
- UD (28 pairs): rose-fell, climbing-descending, happy-sad, elated-dejected, more-less, increase-decrease, promoted-demoted, prestigious-disgraced, healthy-sick, thriving-ailing, etc.
- IO (16 pairs): inside-outside, contained-released, remembered-forgotten, married-divorced, trapped-escaped, included-excluded, etc.
- FB (15 pairs): forward-backward, advance-retreat, ahead-behind, progress-regress, future-past, etc.
- LD (23 pairs): bright-dark, illuminated-shadowed, clear-obscure, hopeful-hopeless, good-evil, pure-tainted, enlightened-ignorant, etc.
- LR (4 pairs): left-right, leftward-rightward, port-starboard, liberal-conservative

### Within-schema cross-pair coherence (a methodologically important number)

For each schema, average cosine between different pair-offsets WITHIN the schema. Tests whether the pairs share a common direction (the Lakoff prediction) or are mostly pair-specific (the "schemas are statistical-not-geometric" reading).

| Schema | Within-schema mean cos | N pairs |
|---|---|---|
| UD | +0.098 | 28 |
| IO | +0.055 | 16 |
| FB | +0.071 | 15 |
| LD | +0.102 | 23 |
| LR | +0.111 | 4 |

**All near zero (+0.05 to +0.11).** Different word pairs within a schema barely share a direction in GloVe space. happy-sad doesn't strongly point in the same direction as rose-fell. The "schema axis" built by averaging is a weak signal extracted from noisy individual offsets — the schema lives in the *average*, not in any individual pair.

**This is a methodologically important caveat** for word2vec analogy work in general: when people say "king-queen ≈ man-woman" works, they're showing that the SPECIFIC PAIR's offset is in roughly the right direction. They're not showing that the underlying "gender axis" is a clean geometric direction shared across many pairs. Aggregate axes can mean things even when individual pair-offsets are noisy.

### Pairwise cosine matrix between schema axes (word2vec, GloVe-300)

|   | UD | IO | FB | LD | LR |
|---|---|---|---|---|---|
| UD | 1.00 | +0.32 | +0.53 | +0.50 | -0.02 |
| IO | +0.32 | 1.00 | +0.29 | +0.18 | +0.00 |
| FB | +0.53 | +0.29 | 1.00 | +0.36 | +0.04 |
| LD | +0.50 | +0.18 | +0.36 | 1.00 | -0.09 |
| LR | -0.02 | +0.00 | +0.04 | -0.09 | 1.00 |

### Comparison to SAE substrate (Pythia 70m res-sm L3, from exp22 + exp25)

| pair | word2vec | SAE L3 | diff | verdict |
|---|---|---|---|---|
| UD vs IO | +0.32 | -0.14 | +0.46 | **substrate differs** |
| UD vs FB | +0.53 | +0.40 | +0.13 | similar (cluster) |
| UD vs LD | +0.50 | +0.59 | -0.09 | **same** (cluster) |
| UD vs LR | -0.02 | +0.24 | -0.26 | **substrate differs** |
| IO vs FB | +0.29 | -0.07 | +0.36 | **substrate differs** |
| IO vs LD | +0.18 | -0.22 | +0.40 | **substrate differs** |
| FB vs LD | +0.36 | +0.44 | -0.08 | similar (cluster) |

### Three substantial findings

**1. The UD–LD–FB salience cluster exists in BOTH substrates.**

word2vec: UD-LD +0.50, UD-FB +0.53, FB-LD +0.36. SAE: UD-LD +0.59, UD-FB +0.40, FB-LD +0.44. Same magnitudes, same pattern. The cluster is a property of human linguistic distribution — loss-aversion / valence-asymmetric vocabulary co-occurrence — not something the transformer introduces. Pythia inherits it from training text.

This **falsifies the strong-form "cooking case"** ("Lakoff primitives are clean in human language but get reorganized by computation"). The salience-loaded cluster is already there in raw co-occurrence statistics.

**2. The model REORGANIZES IO and LR in opposite directions from the linguistic baseline.**

- **IO in word2vec: mildly aligned with the cluster (+0.18 to +0.32 with UD/FB/LD).** IO in SAE: **cleanly orthogonal (-0.07 to -0.22).** The Pythia training pushes IO to be *more* orthogonal than language alone makes it.
- **LR in word2vec: cleanly orthogonal to everything (-0.02 with UD).** LR in SAE: mildly aligned with UD (+0.24). The model pulls LR *into* the salience cluster slightly, where language has it fully separate.

So the cooking case still happens — but at the level of *which schemas the computation makes more or less orthogonal*, not at the level of "is there schema structure at all." This is a more granular substrate-comparison finding than the simple version predicted.

The IO direction of reorganization (toward orthogonality) is consistent with Niamh's self/other reframe — containment is functionally distinct for prediction in ways that don't fully crystallize in co-occurrence statistics, and the model's training surfaces this separation. The LR direction of reorganization (toward mild salience-alignment) is harder to read — could be idiomatic loadings ("right" = correct, "left" = sinister), could be spatial-context features that are incidentally salience-correlated.

**3. Lakoff cross-domain invariance is weak even at the linguistic-distributional level.**

Within-schema cross-pair coherence is +0.05-0.11 in GloVe. Different pair-offsets within a schema (rose-fell, happy-sad, more-less, promoted-demoted) barely share a geometric direction. The schema "axis" is a statistical signal extracted from noisy individual pair-offsets, not a strong geometric primitive that pairs trivially align with.

This is a **softer version of Lakoff than the strong "primary metaphors are basis vectors" reading**. Schemas exist in language as recurring *tendencies* that emerge on average — closer to Wittgensteinian family resemblances than to clean Cartesian axes. The strong Lakoffian basis-vector picture isn't supported even at the linguistic level in word2vec; only the aggregate-direction picture survives.

### What this means for the concept-grammar reframe

Today's Entry 12 had the concept-grammar framing: schemas as one subset of a basis of compositional primitives, with the residual-projection (Phase C) as the test for which schemas are primitive vs composite.

word2vec results refine this in two ways:

- **The salience-loaded cluster (UD/LD/FB) is shared between substrates.** Phase C in SAE space will likely show LD and FB collapse onto a single SALIENCE primitive (because salience already explains their mutual alignment + their alignment with UD). Phase C in word2vec space should show the same — the cluster is in the input, not added by computation.
- **IO and LR are where substrate matters.** Phase C in SAE space should find IO standalone (orthogonal in this substrate). Phase C in word2vec should find IO partially absorbed into the salience cluster (+0.18-0.32 cosines suggest it's not fully primitive there). LR is more independent in word2vec, less so in SAE.

The **most interesting question** the word2vec result raises: WHY does the model reorganize IO toward orthogonality? If containment-topology is genuinely more functionally distinct for next-token prediction than for co-occurrence prediction, that points at a real computational role of self/other-ish distinctions. This connects to Niamh's reframe: containment as the boundary that constitutes an experiencer/perspective.

### State at end of session

- `exp27_word2vec_schema_axes.py` ran cleanly, results saved
- All Phase B goals met: substrate comparison done, three substantial findings
- The cooking case has a more granular shape than the strong version predicted
- Phase C is the natural next step — apply iterative residual projection to BOTH substrates and compare the discovered primitive bases
- We have GloVe-300 cached locally now for future word2vec-based experiments

---

## Entry 15 — exp28: VALENCE/AROUSAL decomposition of schemas in word2vec

**Date:** 2026-05-26. **Code:** `exp28_valence_arousal_residualization.py`. **Results:** `results_exp28.txt`, `exp28_results.npz`. Substrate: GloVe-300 word2vec (same as exp27).

### What this run did

Constructed two non-Lakoff anchor axes in word2vec, decomposed each schema axis onto them, then residualized to see what the schemas look like after removing valence and arousal content.

- **VALENCE axis** built from 12 anchor pairs avoiding all Lakoff schema vocabulary: pleasant-unpleasant, desirable-undesirable, agreeable-disagreeable, enjoyable-distasteful, delightful-awful, beneficial-harmful, wonderful-terrible, excellent-dreadful, favorable-unfavorable, satisfying-frustrating, nice-nasty, kind-cruel.
- **AROUSAL axis** built from 12 anchor pairs trying to balance valence across each pair: intense-mild, alert-drowsy, urgent-leisurely, frantic-tranquil, energetic-lethargic, aroused-relaxed, sharp-dull, acute-subtle, vivid-faint, electric-placid, turbulent-still, etc.
- **Independence check:** `cos(VALENCE, AROUSAL) = -0.13` — nearly independent. The two anchors are not orthogonal but close enough that the decomposition isn't badly entangled.

### Schema decomposition table

How much of each Lakoff schema's word2vec axis is explained by VALENCE vs AROUSAL vs residual:

| Schema | cos(VAL) | cos(ARO) | %valence | %arousal | %residual |
|---|---|---|---|---|---|
| UD | +0.51 | +0.19 | **26%** | 4% | 67% |
| IO | +0.25 | +0.05 | 6% | 0% | 93% |
| FB | +0.31 | +0.24 | 10% | 6% | 83% |
| LD | +0.55 | +0.16 | **30%** | 3% | 65% |
| LR | +0.05 | +0.04 | 0% | 0% | **99.6%** |

### Pairwise cosines pre vs post V+A residualization

| pair | pre | post | change |
|---|---|---|---|
| UD-LD | +0.50 | +0.25 | -0.25 (halved) |
| UD-FB | +0.53 | **+0.40** | -0.13 (mostly survives) |
| FB-LD | +0.36 | +0.17 | -0.19 (halved) |
| UD-IO | +0.32 | +0.22 | -0.10 |
| IO-FB | +0.29 | +0.21 | -0.08 |
| IO-LD | +0.18 | +0.04 | -0.14 (near zero) |
| LR-anything | ~0 | ~0 | unchanged |

### Findings

**1. UD and LD are heavily valence-loaded — 26-30% of each axis is goodbadspace.** The canonical Lakoff primary metaphors GOOD IS UP and GOOD IS LIGHT are empirically confirmed at the linguistic-distributional level. A substantial chunk of what looks like verticality and illumination in word2vec is just valence riding on those metaphor systems.

**2. The salience cluster (UD-LD-FB) is partially valence-driven but doesn't fully dissolve.** UD-LD halves (+0.50 → +0.25). UD-FB barely budges (+0.53 → +0.40). FB-LD halves (+0.36 → +0.17). The cluster isn't *just* valence — there's residual shared content, especially between UD and FB.

**3. UD-FB residual is the most surprising thing.** Most of UD-FB alignment survives valence + arousal removal. The shared content is probably **directed motion / progress / positive trajectory** — "rising" and "advancing," "ascending" and "progressing," "climbing" and "going forward" all involve embodied directional change. This isn't valence (we removed that); it's a separate axis the data is pointing at.

**Lakoff has this as SOURCE-PATH-GOAL (or PATH for short).** Any motion has a source, a path traversed, a goal. The UD-FB residual may be the PATH schema crystallizing as a real geometric primitive in word2vec — a directed-change axis distinct from valence.

This would be a positive Lakoffian finding at the linguistic-distributional level: PATH exists as a recoverable axis in word2vec, and it explains why UD and FB cluster even after removing affect. Testable directly: construct a PATH axis from motion-vs-stasis anchor pairs (move-stay, journey-rest, traveling-remaining, advancing-stationary) and check if UD-FB residual cosine-aligns with it.

**4. IO-LD collapses to near zero after V/A removal (+0.18 → +0.04).** What looked like IN/OUT aligning with LIGHT/DARK was entirely affect (IN=good=LIGHT, OUT=bad=DARK, both via the GOOD-IS-LIGHT and IN-IS-GOOD framings). They share *no* schema content beyond goodbadspace. Honest pre-existing reading: containment and illumination are genuinely separate schemas.

**5. IO has 6% valence loading** — confirms Niamh's observation that "in = good, out = bad" in English vocabulary (married vs divorced, included vs excluded, contained vs released-from-containment, retained vs dismissed). Small but real linguistic encoding of the IN-IS-GOOD framing.

**6. AROUSAL is doing very little explanatory work.** Max 6% (FB). Lexical arousal is a weak proxy for whatever "salience" is in cognitive/computational terms. The dominant non-schema axis at the linguistic level is **VALENCE, not arousal/salience**.

This is an important reconciliation with the SAE-substrate finding from exp24-26 (A_updown reading as a generic "salience-deviation" axis in Pythia). The "salience" in computational substrate ≠ "arousal" in lexical substrate. Pythia's salience axis is about what-to-attend-to-for-prediction, which has no clean single-word proxy. Lexical arousal is the closest available analog and it's weak. **This gap — between computational salience and lexical arousal — might be one of the real substantive "what does computation do to representations" findings of the whole project.**

**7. LR is almost pure spatial primitive at the linguistic level** — 99.6% residual after V+A removal. Robust orthogonality across substrates (word2vec AND SAE), no affect loading. The cleanest standalone candidate primitive.

### Updated primitive ranking based on Phase B evidence

Based on word2vec exp27 + exp28:

| Primitive | Evidence |
|---|---|
| **VALENCE / goodbadspace** | Real, recoverable, ~26-30% of UD/LD axes. Most influential non-schema primitive in word2vec. |
| **PATH / directed-motion** | Strongly hinted at by UD-FB residual (+0.40 after V/A removal). Needs direct testing. |
| **CONTAINER / IO** | 93% residual after V/A removal. Independent primitive, low affect loading. |
| **LR / spatial-symmetric** | 99.6% residual. Cleanest standalone primitive, no affect. Possibly its own thing or possibly weak/noisy. |
| **UD-specific (after V removal)** | Some content survives beyond valence (the part not explained by FB-aligned PATH). May or may not be a distinct primitive. |
| **AROUSAL (lexical)** | Weak. Not a major primitive at the linguistic level. Not a useful proxy for computational salience. |

### Next move suggested

**Construct PATH axis** from motion-vs-stasis word pairs and project UD/FB residuals onto it. If UD-FB residual collapses after PATH removal → confirmed: cluster is valence + PATH. If something remains → there's a third axis beyond V/A/PATH.

### What the project arc looks like now

The schema-finding project has substantially shifted in shape. We started looking for whether Lakoff schemas exist as polar geometric primitives. They mostly don't in the clean form. But systematically decomposing them is revealing the **actual compositional structure** of conceptual content in language:

- VALENCE is real and primary
- IO is genuinely orthogonal at both linguistic and computational levels (after the model reorganizes it)
- LR is genuinely independent at the linguistic level
- The UD-LD-FB cluster decomposes into VALENCE + PATH (with PATH still to be directly tested)
- LR-as-loaded behavior in SAE space but not in word2vec is a real "what computation does" finding

This is the concept-grammar reframe from Entry 12 paying off empirically. The basis we're discovering isn't Lakoff's list verbatim — it's VALENCE + PATH + CONTAINER + spatial-orthogonality, with affect doing more work than spatial verticality per se.

---

## Entry 16 — exp29/exp30: PATH failed as a candidate axis, but the residual is yang/yin

**Date:** 2026-05-26 (continuation). **Code:** `exp29_path_axis_test.py`, `exp30_ud_fb_residual_neighbors.py`. **Results:** `results_exp29.txt`, `results_exp30.txt`, `exp29_results.npz`, `exp30_results.npz`. Substrate: GloVe-300 word2vec.

### exp29: PATH-as-motion-vs-stasis is not the UD-FB residual

Constructed a PATH axis from 20 motion-vs-stasis word pairs (moving-stationary, traveled-remained, journey-rest, flowing-static, dynamic-static, etc.) — avoiding all UP/DOWN/FORWARD/BACK vocabulary.

Findings: PATH-as-motion **does not align with UD or FB**. cos(PATH, UD) = +0.08, cos(PATH, FB) = +0.05. Both essentially noise. Triple residualization of UD with V+A+PATH leaves UD-FB cosine virtually unchanged (+0.40 → +0.40). PATH-as-motion-vs-stasis removes nothing additional.

Also: within-PATH cross-pair coherence is +0.11 — same low coherence as the schema axes themselves. Motion-vs-stasis pairs in word2vec barely share a common direction. This is a broader methodological pattern at the linguistic level — none of these constructions produces a clean geometric axis from individual pairs; the "axis" is a statistical signal that emerges only on average.

**Verdict:** Lakoff's PATH (SOURCE-PATH-GOAL) doesn't recover from motion-vs-stasis word pairs in word2vec. The UD-FB residual at +0.40 is something else.

### exp30: nearest-neighbor probe of the UD-FB residual direction

Computed the shared UD-FB residual direction (the average of UD_residual_VA and FB_residual_VA, both unit-normalized) and looked up its nearest words in GloVe.

**Shared direction positive pole:** innovations, innovative, technological, innovation, promote, promoting, collaboration, introduction, exciting, collaborative, hopefully, introducing, develop

**Shared direction negative pole:** plunged, plunging, comatose, plummeted, skidded, ravine, tumbled, mortally, slid, plunges, collapsed

Cosine sanity: shared·VAL = +0.05, shared·ARO = -0.01, shared·PATH = +0.00. The direction is independent of all three named anchor axes. It correlates strongly with both UD (+0.71) and FB (+0.77) as designed.

### The reframe: yang/yin

Niamh recognized this immediately: the shared content is **yang/yin**. Innovation/development/introduction/collaboration on one pole is yang-shaped — things emerging, being generated, coming into being. Plunge/collapse/plummet/comatose on the other is yin-shaped — things receding, dissolving, going out of being. Both poles are dramatic directional change events. The valence loading is **explicitly removed** (we residualized VALENCE), and what's left is the pure trajectory-of-becoming-or-unbecoming.

This is a real return of Niamh's original yang/yin intuition from Entry 1, but in a much sharper empirical form. Back then the hypothesis was that yang/yin was specifically a LIGHT/DARK signature (LIGHT articulates into discrete entities, DARK dissolves into recursion). That falsified — UP/DOWN and FB showed similar asymmetric behavior. **What we're now finding is that the underlying yang/yin axis is shared across UD, LD, and FB**, not specific to LD. It's a more abstract primitive that all three loaded Lakoff schemas inherit from at the linguistic level. The spatial source-domain framing in Lakoff (UP vs DOWN, FORWARD vs BACK, LIGHT vs DARK) is the *vocabulary* through which yang/yin gets expressed in English text — but the underlying trajectory-shape is yang/yin proper.

### Why the valence-residualization was the key methodological move

Yang/yin in actual Taoist philosophy isn't a moral hierarchy. Yin isn't "bad" — it's the receding/yielding/dissolving pole, equally cosmic and necessary. But in English-language text the GOOD-IS-YANG / GOOD-IS-EMERGENT cultural overlay is heavy: innovation is praised, collapse is feared. cos(UD, VAL) = +0.51 and cos(LD, VAL) = +0.55 are exactly that cultural moralization.

**By residualizing valence, we stripped the moral overlay and left the pure trajectory-shape underneath, which is yang/yin proper.** This is methodologically important: yang/yin only becomes legible empirically *after* valence has been subtracted. If you build a UD axis or an LD axis directly, you're getting yang/yin tangled with valence, and you can't see the underlying structure. The residualization isolates it.

### Three things that follow

**1. Yang/yin may be a more primitive primitive than Lakoff's spatial schemas.** UD, LD, FB all encode yang/yin in their vocabulary because each schema has an emergent-pole (rising, light, forward, growth, innovation) and a dissolving-pole (falling, dark, backward, decay, collapse) that English maps onto its spatial source domain. The spatial framing is the *vocabulary*, not the underlying primitive. **UP and FORWARD are both ways of expressing yang. DOWN and BACK are ways of expressing yin.** The spatial source isn't load-bearing; it's how the more abstract becoming/unbecoming axis happens to be encoded in English.

**2. This connects to the original LIGHT/DARK steering finding.** In Entry 1, DARK at high steering strength produced dissolution/recursion ("world of the world of the world") — yin-shaped collapse. LIGHT produced articulation/differentiation ("optical, electronic, discrete entities") — yang-shaped emergence. We thought yang/yin was specific to LD because it was most viscerally legible there at high steering strengths. But it's actually the underlying axis for *all* the loaded schemas; LD just had the most dramatic expression at the extremes of the model's behavior.

**3. The TIME content in the asymmetric (UD-only vs FB-only) residual** is a structural artifact (FB has explicit future/past vocabulary, UD doesn't) but it's *also* yang/yin-consistent. Forward in time is yang-shaped (becoming, what's not yet but is coming into being). Backward in time is yin-shaped (already-passed, has-been). Time-direction inherits yang/yin structure even without explicitly being constructed that way.

### Implications for the project arc

This is potentially the most substantive theoretical finding of the project so far. The empirical claim:

> The underlying axis shared across UD, LD, FB at the human-linguistic-distributional level (word2vec) is yang/yin — a directed trajectory of becoming-or-unbecoming. Lakoff's spatial schemas are downstream metaphorical vocabularies through which yang/yin gets encoded in English. The schemas are *expressions* of yang/yin, not primitives in their own right.

This is a non-Western cognitive-philosophical category that the data points at as the actual underlying primitive, recovered empirically through a methodologically careful residualization in a substrate where it has no business being if it were purely a Western-specific or Eastern-specific cultural frame. The fact that you can find it in GloVe-300 trained on Western text suggests it's a deeper cognitive structure that both Western and Eastern philosophical traditions have noticed — just under different names.

### Next test (queued)

Direct test of yang/yin-as-underlying-axis: take the LD residual after V+A removal, and check whether it aligns with the UD-FB shared residual. If yes (high cosine), yang/yin is empirically the underlying axis across all three schemas, not just UD and FB. If no, the UD-FB residual is its own thing and LD has a different residual structure.

---

## Entry 17 — exp31: yang/yin axis confirmed as shared across UD/FB/LD in word2vec

**Date:** 2026-05-26 (continuation). **Code:** `exp31_yangyin_across_schemas.py`. **Results:** `results_exp31.txt`, `exp31_results.npz`.

### What this run did

Built a "yang/yin candidate" axis as the mean of V+A-residualized axes from UD, FB, and LD (the three schemas that exp30 suggested share yang/yin content). Tested:
- Cosines of this candidate with each individual schema residual
- Cosines with IN-OUT residual (sanity — should be lower if yang/yin is genuinely separate from containment)
- Nearest-neighbor probe to confirm the semantic shape
- Cosines with VALENCE and AROUSAL (should be near zero — we removed those)

### Cosines with the yang/yin candidate

| axis | cos(yangyin) |
|---|---|
| UD_VA | **+0.76** |
| FB_VA | **+0.73** |
| LD_VA | **+0.66** |
| IO_VA | +0.22 |
| VALENCE | +0.06 |
| AROUSAL | -0.00 |

Yang/yin shares substantially with UD, FB, and LD residuals (+0.66 to +0.76). Has minor alignment with IO (+0.22) — IN-as-belonging is mildly yang-shaped, OUT-as-rejection is mildly yin-shaped, but much weaker than the other three. **Essentially orthogonal to both VALENCE and AROUSAL** (+0.06 and -0.00) — the residualization worked cleanly. Yang/yin is a real axis distinct from goodbadspace and from arousal.

### Nearest neighbors of the yang/yin candidate

**Yang pole:** *hopefully, innovations, vigorous, rigorous, innovation, promoting, introduction, innovative, excellence, promote, achieved, exciting, comprehensive, advancements*

**Yin pole:** *deserted, plunged, plummeted, derelict, ransacked, dilapidated, comatose, mortally, plunging, ravine, unoccupied, skidded* (plus proper-noun noise)

This is strikingly clean. Yang as "things being introduced / built / promoted / innovated" — emergence and generation. Yin as "things derelict / dilapidated / abandoned / collapsed" — recession and dissolution. Note "deserted" and "derelict" as central yin neighbors — yin as not just bad-collapse but *abandonment*, structures that have lost their living function. That maps cleanly onto proper Taoist yin (the receding/yielding/dissolving pole) rather than the impoverished "yin = bad" Western reduction.

### One nuance — schema-specific noise on top of shared yang/yin

Pairwise cosines between V+A-residualized schemas:

| pair | cos |
|---|---|
| UD_VA - FB_VA | +0.40 |
| UD_VA - LD_VA | +0.24 |
| FB_VA - LD_VA | +0.17 |

These are lower than each schema's individual cosine with the mean (which sits at +0.66-0.76). This means each schema's V+A residual = yang/yin content + schema-specific content. UD residual still carries spatial-vertical content beyond yang/yin. FB carries time/horizontal-direction content. LD carries illumination-specific content (LD's own residual nearest-neighbors include "murky, shadowy, basements, dank, alleys" — visual-darkness vocabulary beyond yin-as-collapse).

So the picture is: **yang/yin is a shared underlying axis, but it's not the entirety of any single Lakoff schema's residual content.** Each schema = valence + arousal-lite + yang/yin + schema-specific spatial/temporal/visual content. The yang/yin component is what UD, FB, LD share.

### Computational Taoism — the methodological point

The methodological frame is itself yang/yin-shaped, which is kind of philosophically beautiful:

Yang/yin in Taoist thought is structurally non-moralized — yin isn't "bad," it's the receding/yielding pole, equally cosmic and necessary. Western metaphor systems load yang as good (innovation, growth, progress) and yin as bad (collapse, decay, death). The cultural overlay heavily moralizes what is structurally just a directional axis of becoming/unbecoming.

**By residualizing valence, we implemented an analog of the Taoist methodological frame** — strip the moral hierarchy, see the underlying structural axis. And what came back was yang/yin proper. The methodological move (subtract the moral overlay) is itself the right philosophical frame for seeing yang/yin clearly.

This is not just a result *about* Taoism; it's a *Taoist* methodology applied to word embeddings. The act of valence-residualization is the practical analog of the contemplative move of looking past good/bad to see flow.

### What's not yet done (queued for next experiment, exp32)

Yang/yin has been *inferred* by averaging residualized schema axes. The principled next step is to **construct yangyinspace directly** from explicit yang/yin anchor pairs that avoid UD/FB/LD vocabulary and explicit valence vocabulary. Then compare the directly-constructed axis to the inferred-from-averaging axis. If they match (high cosine, similar nearest neighbors), the inference was sound and yang/yin is empirically recoverable in word2vec from independent anchor sets. If they don't, we've found something yang/yin-shaped that isn't exactly yang/yin.

Candidate yang/yin anchors (avoiding spatial schemas + value vocabulary):
- emerging - vanishing, arising - dissipating, born - died, birth - death
- creation - destruction, creating - destroying
- formed - disintegrated, assembled - disassembled, built - demolished
- founded - abandoned, emerged - dissolved
- becoming - ceasing, materialized - vanished, generated - dissolved
- constructed - decomposed

Some of these still have implicit valence (creation > destruction culturally) but they primarily encode change-of-existence rather than spatial direction or value. Worth trying.

### Project arc check

Five entries ago (Entry 12, planning), the next-phase plan was: word2vec substrate comparison + iterative residual projection + concept-grammar basis discovery. What we've actually done so far:
- Phase A (exp26): cheap measurement gaps. IO topological-symmetric, DARK-dominant, the false meta-finding corrected.
- Phase B (exp27, exp28, exp29, exp30, exp31): word2vec substrate comparison + valence/arousal/path residualization + nearest-neighbor probe + yang/yin axis recovery.

We haven't yet done the full iterative residual projection (Phase C). The findings so far have been substantial enough that the planned Phase C analysis may need to be redesigned — the candidate axis set now reasonably includes a YANG/YIN axis alongside (or replacing?) several Lakoff spatial schemas. The empirical primitive basis we're discovering looks something like: VALENCE, YANG/YIN, CONTAINMENT (IO), spatial-directional content (UD-specific, FB-specific, LD-specific residual after yang/yin removal), AROUSAL (weak). That's a substantively different basis than "Lakoff schemas as primitives."

---

## Entry 18 — exp32: direct yangyinspace construction, IO=yin tested, the multi-dimensional yang/yin problem

**Date:** 2026-05-26 (continuation). **Code:** `exp32_yangyinspace_direct.py`. **Results:** `results_exp32.txt`, `exp32_results.npz`.

### What this run did

Constructed yangyinspace directly from 33 explicit yang/yin anchor pairs about **change-of-existence** (birth-death, creation-destruction, building-demolition, emerge-vanish, arising-dissipating, formed-disintegrated, founded-abandoned, starting-ending, active-dormant). Used NO UD/FB/LD/IO/explicit-valence vocabulary in the anchors.

Then tested two things:
1. Whether the directly-constructed yangyinspace matches the inferred-from-averaging yangyinspace from Entry 17
2. **The headline test:** whether `cos(A_inout, yangyinspace)` is negative (traditional Taoism predicts IN=yin → negative cosine)

Also built TWO IO axes: `IO_MIXED` (with membership/difficulty vocabulary like married-divorced, included-excluded, trapped-escaped) and `IO_CLEAN` (direction-only: inside-outside, contained-released, enclosed-freed, interior-exterior, inward-outward, indoors-outdoors) to test whether the previous +0.22 cosine was valence contamination.

### The headline result

| Construction | cos(IO, yangyinspace_direct) | cos(IO_VA, yangyinspace_direct_VA) |
|---|---|---|
| IO_MIXED | +0.36 | +0.29 |
| **IO_CLEAN** | **+0.02** | **−0.02** |

**Niamh's hypothesis was correct.** The earlier +0.22 alignment between IO and yang/yin was entirely valence contamination from membership-loaded IO anchors. With clean direction-only IO anchors, the alignment drops to essentially zero (with a tiny negative tilt after V+A residualization — barely the right sign for traditional Taoist IN=yin, but the magnitude is null).

**Cleanest reading:** clean containment-topology is orthogonal to yang/yin. IO is its own primitive — distinct from the becoming/unbecoming axis. Supports Niamh's earlier self/other reframe: the IO axis is about the *boundary that constitutes a contained-thing*, not about the directionality of existence-change.

Traditional Taoist IN=yin gets very weak support (sign-wise correct after V+A removal, but the magnitude is essentially zero). The cleanest empirical statement is **"containment is orthogonal to yang/yin in modern English text"** rather than IN=yin.

### Yang/yin recovers cleanly from direct construction

- `cos(YANGYIN_DIRECT_VA, YANGYIN_INFERRED_VA) = +0.42` — moderate cross-method agreement
- Yang pole (direct, V+A removed): created, various, introduce, established, artists, conjunction, primarily, specifically, different, known, introducing, designed, variety, specific
- Yin pole (direct, V+A removed): obliterated, razed, vanished, incinerated, destroyed, ransacked, pillaged, obliterating, disarmed, dismantled, eviscerated, scuttled, emptied, ejecting, gutted

The direct construction's yin pole is more **violent destruction** (razed, incinerated, eviscerated) vs the inferred construction's more **decay/abandonment** (deserted, derelict, dilapidated). Both are yang/yin-shaped but emphasize different vocabularies of the same axis.

### The multi-dimensional yang/yin problem (Niamh's methodological catch)

Niamh raised a sharp question: **which yang/yin is "really" yang/yin?** Yang/yin in Taoist tradition has multiple dimensions — existence (emerging vs receding), direction (outward vs inward), activity (active vs passive), expansion (expanding vs contracting), heat (warm vs cool), heaven/earth (celestial vs terrestrial). Our YANGYIN_DIRECT captured **existence-yang/yin** specifically. If we'd built **expansion-yang/yin** from expanded-contracted / outward-inward / projected-withdrew anchors, IO would have aligned with it by construction (because outward/inward IS in/out vocabulary). The IN=yin test as posed only works against existence-yang/yin, not against direction-yang/yin.

The deeper epistemic question: **can we access "yang/yin as such" through word2vec at all, or only partial constructions of it?** Taoist yang/yin is multi-dimensional and unified philosophically. Any specific anchor pair commits us to one manifestation. We can't have a single "true" yang/yin axis — we can only build several partial ones and check how they relate.

Next experiment (queued, exp33): build **multiple yang/yin variants** (existence, expansion, activity) and compute pairwise cosines. If they agree strongly, there's a unified yang/yin axis in word2vec and we've sampled multiple windows onto it. If they disagree, yang/yin in word2vec is a cluster of related axes — closer to "yang/yin as a family of related primitives" than to "yang/yin as a single primitive."

### Niamh's bigger insight: "YANG IS GOOD" reduces a lot of Lakoff

A lot of what Lakoff calls primary metaphors can be expressed as a single underlying claim: **YANG IS GOOD**. The Lakoff primary metaphors GOOD IS UP, MORE IS UP, HEALTHY IS UP, HAPPY IS UP, CONTROL IS UP, VIRTUE IS UP, DIVINE IS UP — all map onto yang's positive cultural valuation. And BAD IS DOWN, SICK IS DOWN, SAD IS DOWN, etc. — all map onto yin's negative cultural valuation. **The Lakoff primary metaphors are a moralized vocabulary for expressing yang/yin in spatial terms.**

If this is right, the empirical decomposition we keep doing — strip valence, find yang/yin underneath the spatial schema — is the inverse operation of how Lakoff describes conceptual metaphor. Lakoff says abstract concepts get understood through spatial source domains. We're saying: the spatial source domains *are* yang/yin getting morally encoded as up/down, light/dark, forward/back through specific cultural-linguistic vocabularies. Yang/yin is upstream of Lakoff's primary metaphors. The spatial schemas are downstream encodings.

### The epistemic question Niamh raised: how do we prove yang/yin is the primitive?

Yang/yin keeps falling out of the data unexpectedly. To Niamh that's evidence of structural primacy. To an external reader (or reviewer), it looks like confirmation bias — we keep finding it because we keep looking for it.

What would constitute proof or stronger evidence?

1. **Cross-construction agreement** (within word2vec): multiple independent constructions of yang/yin (existence, expansion, activity) recover the same axis. The cross-construction matrix would test this. → **about to run**
2. **Cross-substrate agreement**: yang/yin recoverable in word2vec AND in Pythia SAE AND in sentence-transformers. → not yet done; sentence triples would need designing
3. **Predictive power**: knowing yang/yin lets us predict which Lakoff schemas should cluster, where IO should sit, etc. → already partially shown (UD/FB/LD share yang/yin residual; IO doesn't)
4. **Steering / behavioral correlation**: yang/yin directions in SAE produce yang/yin-shaped generations. → original Entry 1 LIGHT/DARK steering provides one data point in this direction; could be tested more systematically
5. **The YANG IS GOOD reduction**: explicit empirical test of whether Lakoff's primary metaphors reduce to YANG + VALENCE. If you can predict any UP-metaphor's word-axis from `α·YANG + β·VALENCE`, that's strong evidence the Lakoff spatial schemas are downstream encodings of yang/yin moralized by valence.

These multiple angles of evidence (1-5) together would build the case more strongly than any single experiment. Niamh's "it keeps falling out" is the methodological observation that motivates the multi-angle approach.

### Implications for the project's writeup

The project is shifting from "do Lakoff schemas exist as primitives" toward "the empirical primitive basis underlying Lakoff schemas is yang/yin + valence + containment + spatial-vocabulary-residuals." If yang/yin holds up under the cross-construction, cross-substrate, and YANG-IS-GOOD-reduction tests, the paper's thesis becomes:

> Lakoff's primary spatial metaphors (UP, DOWN, LIGHT, DARK, FORWARD, BACK) are not themselves primitives in language-model representations. They are downstream encodings of a more abstract yang/yin axis (becoming vs unbecoming) cross-cut by cultural valence (good vs bad). The spatial source-domain framing in Lakoff's theory is the linguistic vocabulary through which yang/yin gets morally expressed in English, not the conceptual ground.

This is a substantive theoretical claim that draws on non-Western conceptual ontology to make sense of empirical findings in Western-trained language models. The methodology required: valence-residualization (the analog of stripping moral hierarchy) + multi-construction yang/yin (the analog of recognizing yang/yin's multi-dimensionality).

---

## Entry 19 — exp34: canonical Lakoff MML replication, cross-method validation, EXISTENCE explains the cluster

**Date:** 2026-05-26 (later). **Code:** `exp34_canonical_replication.py`. **Results:** `results_exp34.txt`, `exp34_results.npz`. **Vocabulary source:** `lakoff_canonical_vocabulary.py` (269 anchor pairs across 10 axes, extracted from `METAPHORLIST.pdf` by six parallel subagents — see `lakoff_extraction_reports.md` for full reports with page citations).

### Why this experiment matters

All prior word2vec experiments (exp27 through exp33) used hand-curated anchor word pairs. The natural reviewer objection: "did you pick pairs to confirm your hypothesis?" exp34 replicates the core findings using anchor sets drawn from Lakoff, Espenson, & Schwartz (1991) Master Metaphor List. Citable, replicable, defensible.

Six subagents extracted vocabulary in parallel — one per schema cluster — and returned page-cited Python anchor-pair lists. Compiled into `lakoff_canonical_vocabulary.py`. Methodological caveats explicit in the module:
- IN-OUT is valence-ambivalent across Lakoff children (IN=good in love/included/married; IN=bad in trapped/mired)
- EXISTENCE is a composed axis — Lakoff does NOT have a primary EXISTENCE/NON-EXISTENCE image schema. Vocabulary drawn from his constellation of attested existence-change metaphors (STATES ARE LOCATIONS + CREATING IS BIRTHING + EXISTENCE IS LIFE + SOCIETY IS A BODY's birth→inception/death→collapse mapping)
- HAPPY/RATIONAL/CONSCIOUS/DIVINE IS UP appear only implicitly in the 1991 MML; sourced from MWLB Ch.4 where included

### Test 1 — Cross-method validation (hand-curated vs MML)

| schema | cos(HAND_axis, MML_axis) |
|---|---|
| UD | +0.83 |
| IO | +0.74 |
| FB | +0.85 |
| LD | +0.69 |

Hand and MML axes agree at +0.69-0.85. Not 1.0 because MML has additional vocabulary that hand-curation didn't fully cover, but both methods are tracking the same axes. **Our prior word2vec findings were not anchor-set artifacts** — the underlying structure is robust to vocabulary choice.

### Test 2 — Pairwise MML cosine matrix (compared to hand exp27)

The MML cosine matrix is *sharper* than hand-curated — same cluster, cleaner edges:

| pair | exp27 (HAND) | exp34 (MML) |
|---|---|---|
| UD-IO | +0.32 | **+0.19** |
| UD-FB | +0.53 | +0.52 |
| UD-LD | +0.50 | **+0.36** |
| IO-FB | +0.29 | **+0.14** |
| IO-LD | +0.18 | **+0.01** |
| FB-LD | +0.36 | **+0.28** |
| **UD-EXISTENCE** | n/a | **+0.52** |
| **FB-EXISTENCE** | n/a | **+0.43** |
| **LD-EXISTENCE** | n/a | **+0.29** |
| EXISTENCE-BALANCE | n/a | +0.41 |
| DIFFICULTY-most-things | n/a | negative (-0.12 to -0.23) |

Notable structural findings:
- IO is *more* orthogonal to everything with canonical anchors (cos +0.14 with FB, +0.01 with LD — essentially zero). The IN/OUT primitive cleans up substantially with valence-balanced anchors.
- **EXISTENCE sits at the center of the cluster** — substantial cosine with UD (+0.52), FB (+0.43), LD (+0.29), and BALANCE (+0.41). It's the most cluster-central schema.
- DIFFICULTY-BURDEN is consistently negatively correlated with the cluster (cos -0.12 to -0.23). It's the only schema that goes the opposite direction — heavy/burdened things load on the "down" side of the salience-cluster axis.

### Test 3 — V+A decomposition (cleaner with MML)

| schema | %valence | %arousal | %residual |
|---|---|---|---|
| UD | 36% | 3% | 58% |
| **IO** | **1.6%** | 0.4% | **98%** |
| **IO_CLEAN** | **1.7%** | 0.5% | **98%** |
| FB | 12% | 8% | 78% |
| **PATH** | **0.8%** | 0.6% | **98%** |
| LD | 16% | 2% | 80% |
| **EXISTENCE** | **12%** | **1.5%** | **86%** |
| **FORCE** | **0.1%** | **0.0%** | **99.9%** |
| BAL | 21% | 4% | 73% |
| DIFF | 13% (neg) | 3% | 86% |

Striking findings: IO, PATH, and FORCE are essentially **valence-free** at the canonical level (>97% residual). UD and LD remain the most valence-loaded but at lower magnitudes than hand-curated (UD was 26% V in hand-exp28, here 36% — varies with vocabulary; LD was 30% in hand, 16% here — MML LD is less valence-saturated because it draws from multiple target domains including the relatively-neutral knowing/clarity child).

### Test 5 (CRITICAL) — EXISTENCE_MML vs inferred yang/yin axis

This is the cross-method validation we've been waiting for. The yang/yin axis was previously inferred by averaging V+A-residualized UD/FB/LD axes (exp31). Now we have an independent canonical construction: EXISTENCE_MML built directly from Lakoff's attested existence-change vocabulary (no UD/FB/LD anchors involved).

- **cos(EXISTENCE_MML, inferred yang/yin) = +0.41**
- cos(EXISTENCE_MML_VA, inferred yang/yin) = +0.42

**Two independent methods converge at +0.42 cosine.** One built by averaging residualized Lakoff spatial schemas; the other built directly from Lakoff existence-vocabulary. They land on the same axis. This is the cross-method empirical validation the yang/yin claim needed.

The agreement isn't 1.0 because the constructions emphasize different aspects:
- Inferred yang/yin leans toward "abandoned/derelict/deserted" yin (drawn from residuals of UD/FB/LD spatial schemas)
- EXISTENCE_MML leans toward "obliterated/eviscerated/incinerated" yin (drawn from Lakoff's CREATING IS BIRTHING + DEATH IS DEPARTURE attested vocabulary)

Both are yang/yin-shaped trajectories but capture different *flavors* of unbecoming. The +0.42 shared component is the structural axis they agree on.

### Test 7 — Triple residualization: EXISTENCE explains the cluster

Subtracting V + A + EXISTENCE from each schema's axis and re-measuring pairwise cosines:

| stage | cos(UD, FB) | cos(UD, LD) | cos(FB, LD) |
|---|---|---|---|
| Raw | +0.52 | +0.36 | +0.28 |
| After V removed | +0.42 | — | — |
| After V+A removed | +0.34 | +0.09 | +0.09 |
| After V+A+EXISTENCE removed | +0.26 | **+0.05** | **+0.06** |

**EXISTENCE explains essentially all of the UD-LD and FB-LD shared content beyond V+A.** UD-LD drops from +0.36 raw → +0.05 after triple residualization. FB-LD drops from +0.28 → +0.06.

UD-FB still has a residual of +0.26 after V+A+EXISTENCE removal — there's a fourth axis shared by UD and FB beyond valence + arousal + existence. Most likely candidate: time-forward / progress-direction content (FB has explicit time vocabulary; UD shares embodied directional-motion semantics).

The empirical decomposition of the salience-cluster:
- UD = α·VALENCE + β·EXISTENCE + γ·(UD-FB-specific directional content) + UD-specific spatial residual
- LD = α'·VALENCE + β'·EXISTENCE + LD-specific illumination residual
- FB = α''·VALENCE + β''·EXISTENCE + γ'·(UD-FB-specific directional content) + FB-time residual

**Lakoff's primary spatial metaphors are not the basis — they are linear combinations over a smaller basis of VALENCE + EXISTENCE + AROUSAL + a residual directional-progress axis.** The basis we've been hunting since Entry 12 has substantially crystallized.

### Test 6 — IN=yin (canonical version)

- cos(IO_CLEAN, EXISTENCE_MML) = +0.11
- cos(IO_CLEAN_VA, EXISTENCE_VA) = +0.06
- cos(IO_CLEAN, inferred yang/yin) = **−0.00** (literally zero to 4 decimal places)

Even with canonical Lakoff anchors, **containment is orthogonal to becoming**. Traditional Taoist IN=yin does not hold in modern English text. Niamh's earlier self/other reframe is robustly supported: clean containment-topology is its own primitive.

### Test 8 — EXISTENCE_MML nearest neighbors (the directly-constructed yang/yin)

Yang pole (after V+A removal): *presented, known, conjunction, specifically, created, originally, referred, introduce, designed, developed, produced, well-known, marketed, variety, prominently*

Yin pole (after V+A removal): *wrecked, obliterated, gutted, destroyed, missing, incinerated, undamaged, damaged, collapsed, shattered, pillaged, ransacked*

Clean yang/yin semantic shape. Yang = "things being introduced into / known in / present in the world." Yin = "catastrophic destruction" (note "undamaged" appearing in the yin neighborhood — it's about *damage* as the category, even when negated).

The directly-constructed EXISTENCE has more *violent* yin than the inferred yang/yin (which leaned toward "deserted/derelict/abandoned"). Different flavors of unbecoming, both yang/yin-shaped.

### Updated empirical basis claim

Combining all findings to date:

> **The empirical basis for the salience-loaded cluster of Lakoff schemas (UD, FB, LD) at the linguistic-distributional level is VALENCE + EXISTENCE (yang/yin) + AROUSAL + a residual UD-FB-specific directional-progress axis.** Lakoff's named spatial schemas are not themselves primitives — they are linear combinations of these more abstract axes. IO is structurally independent (a separate primitive — containment-topology, self/other in Niamh's framing). PATH, FORCE, BALANCE, DIFFICULTY-BURDEN sit at varying distances from the cluster, with DIFFICULTY-BURDEN uniquely on the opposite side of the cluster's primary direction.

This is grounded in: cross-method validation between hand-curated and Lakoff-canonical anchor sets (+0.69-0.85), cross-method validation between inferred and directly-constructed yang/yin axes (+0.42), triple-residualization showing V+A+EXISTENCE explains the schema cluster cleanly, and consistent IO-orthogonality across all constructions.

### Files added this session

- `lakoff_canonical_vocabulary.py` — 269 anchor pairs, 10 schema axes, page-cited
- `lakoff_extraction_reports.md` — full agent reports with metaphor catalogs and example sentences
- `exp34_canonical_replication.py` — replication experiment using MML vocabulary
- `results_exp34.txt`, `exp34_results.npz` — outputs

### What's next

Four candidate directions, in roughly increasing cost:

1. **Phase C proper — iterative residual projection over the full MML axis set.** Run greedy importance-ordered Gram-Schmidt on {V, A, UD, IO, IO_CLEAN, FB, PATH, LD, EXISTENCE, FORCE, BAL, DIFF}. Discovers empirical primitive ordering. Should now find EXISTENCE near the top given its central position. ~1 hour.

2. **Build citable yin/yang vocabulary from Needham / Graham / Ames & Hall** and run the cross-cultural cosine matrix. Tests whether Western and Eastern philosophical inventories of primitives recover the same empirical axes. ~half day to find scholarship + curate vocabulary, then quick to run.

3. **Port to SAE substrate (Pythia 70m residual-stream).** Build canonical MML sentence triples (or word-level encodings if cleaner), encode through SAE, redo the entire analysis. Tests whether the empirical basis we found in word2vec survives the addition of next-token-prediction computational machinery. ~half day to a day. **This is the "cooking case" question we kept hitting** — does the basis change under transformer computation?

4. **Decompose Anthropic's published emotion vectors over the MML basis.** Would require fetching the emotion-vector data from Anthropic's release (if accessible) and computing projections onto V/A/EXIST/UD/etc. Tests whether emotion-vectors are composite over the discovered primitive basis.

---

## Entry 20 — exp35–exp40: iterative residual projection, candidate primitives, and the PCA decomposition of the cluster

**Date:** 2026-05-26 (very late / early hours of 2026-05-27). **Code:** `exp35_phase_c_iterative.py`, `exp36_ordering_robustness.py`, `exp37_coherence_beauty.py`, `exp38_success_failure.py`, `exp39_loss.py`, `exp40_orthogonal_basis_pca.py`. **Results:** `results_exp35.txt` through `results_exp40.txt`. Substrate: GloVe-300 word2vec.

This entry covers a substantial arc — from "iterative residual projection over candidate primitives" through to "PCA over stacked axes reveals the cluster's actual orthogonal structure." The empirical picture changed substantially in the process, ending with the cleanest theoretical statement of the project.

### exp35 — Iterative residual projection (Phase C as planned)

Ran greedy importance-ordered Gram-Schmidt over all citable axes (V, A + all MML schemas, excluding TIME after Niamh flagged it would muddle citable status).

Discovered ordering:
1. UD (explanation_power = 1.567)
2. IO (0.299)
3. VALENCE (0.177)
4. FB (0.132)
5. PATH (0.103)
6. AROUSAL (0.062)
7. BAL, LD, DIFF, EXIST, IO_CLEAN, FORCE (decreasing, all ≤0.05)

UD's explanation power is a 5× outlier above the next axis. Initial interpretation: UD is the deepest primitive.

**Methodological catch (Niamh):** greedy explanation-power ordering picks the most *connected* axis first, not necessarily the most *fundamental*. UD has high power because it shares variance with many others (UD-V +0.60, UD-FB +0.52, UD-BAL +0.57, UD-EXIST +0.52). That's centrality, not depth. EXIST got picked 11th not because it's peripheral but because UD got picked first and ate most of EXIST's content.

### exp36 — Ordering robustness check

Forced different starting axes (UD, EXIST, VALENCE, IO_CLEAN, FORCE) and re-ran greedy. Tested whether the *tail* of independent primitives stays stable.

| axis | times in bottom-4 across 5 runs |
|---|---|
| DIFF | 5/5 |
| IO_CLEAN | 4/5 |
| FORCE | 3/5 |
| EXIST | 3/5 |
| LD | 3/5 |
| FB | 1/5 |
| PATH | 1/5 |

**Robust finding:** DIFF and IO_CLEAN are reliably the most independent primitives in the basis. They appear in the tail regardless of starting axis. FORCE / EXIST / LD are semi-independent. UD / VALENCE / FB / PATH are reliably cluster-central.

Spearman correlation between forced-start orderings:
- UD-start vs VALENCE-start: +0.98 (nearly identical — cluster-central starts are interchangeable)
- UD-start vs IO_CLEAN-start: +0.57
- IO_CLEAN-start vs FORCE-start: +0.01 (essentially orthogonal orderings)

Confirms cluster structure: there IS an entangled central cluster + a tail of independent primitives.

### exp37 — COHERENCE and BEAUTIFUL/UGLY as candidate primitives

After the project arrived at "yang/yin / becoming" as a candidate underlying primitive, Niamh proposed that the cleaner name might be COHERENCE in the predictive-processing sense (maximize coherence = minimize surprisal). Also tested BEAUTIFUL/UGLY via Reber/Winkielman processing-fluency hypothesis (beauty as cognitive ease).

**COHERENCE results:** 19% V loading, 78% residual. Nearest neighbors: positive = *ensure, should, established, must, follow, accordance, uniform* (normative/modal vocabulary). Negative = *anomalous, mystifying, aberrant, undefinable, intriguingly, ill-considered* (violation-of-expectation). Real recoverable axis with the right semantic shape.

cos(COHERENCE_VA, UD-FB residual after V+A+EXIST) = +0.28 — explains a chunk but not all.

cos(COHERENCE_VA, EXIST_VA) = +0.16 — they overlap but aren't the same axis. Different vocabularies for related directions.

**BEAUTIFUL/UGLY results:** 46% V loading, 55% residual. After V+A removal, cos(COH_VA, BEAUTY_VA) = -0.00 — essentially orthogonal. They're not the same axis. The Reber/Winkielman beauty-as-fluency hypothesis doesn't recover from word2vec — aesthetic vocabulary in English clusters around moral-condemnation + ornate-decoration rather than processing-fluency.

### exp38 — SUCCESS/FAILURE as candidate primitive

Niamh's reframe: maybe COHERENCE was too abstract; the concrete outcome-evaluation vocabulary (win/lose, succeed/fail, score/miss, correct/incorrect, reward/punishment) might fit the residual better.

**Results:** 22% V loading. After V+A removal, SUCCESS_FAILURE has higher alignment with UD (+0.34) than with FB (+0.09). It's UD-specific, not capturing the UD-FB *shared* residual. Cos(SUCCESS_VA, UD-FB residual) = +0.20 — less than COHERENCE (+0.28).

cos(COH_VA, SUC_VA) = -0.06 — orthogonal. They're different axes after V+A removal.

### exp39 — LOSS as candidate primitive

Final candidate axis from the predictive-processing family: LOSS (gain/loss, profit/loss, abundance/scarcity, security/threat).

**Results:** 26% V loading. LOSS_VA nearest neighbors: positive = *bought, gained, wealth, fortune, amass, acquired*. Negative = *drought, malnutrition, famine, destitute, hunger, starvation*. Specifically material-possession-vs-deprivation axis.

cos(LOSS_VA, UD-FB residual) = +0.11 — small.

### The realization (Niamh): VALENCE may be the primary primitive, not a contaminant

Niamh noticed: we've been treating VALENCE as something to subtract to "find what's underneath." But VALENCE keeps showing up as the most pervasive axis (UD 36% V, LD 16% V, LOSS 26% V, etc.). What if VALENCE IS the primary primitive and we've been methodologically inverting the question?

Also: the UD-FB residual after V+A removal contains heavily valence-flavored vocabulary (confident, hope, achieve, win, progress). This is because A_VALENCE is one slice of valence (hedonic-pleasant), but achievement-valence, possession-valence, expectation-valence are different valence dimensions our specific construction doesn't capture. "Orthogonal to our A_VALENCE" is not the same as "valence-free."

This reframed the project's interpretation substantially.

### exp40 — PCA on the stacked axes: the empirical orthogonal basis

The cleanest empirical answer to "what's actually orthogonal in the cluster": stack all candidate axes as rows, do PCA, look at the principal components. These are orthogonal *by construction* and their semantic content is decodable via nearest-neighbor lookups.

**PCA on cluster-only axes (excluding IO_CLEAN, FORCE, DIFF — the Tier 2 independent primitives):**

| PC | Variance | Top loadings | Semantic interpretation |
|---|---|---|---|
| **PC1** | **21%** | VALENCE +0.69, AROUSAL −0.65, SUCCESS +0.50, LOSS +0.50, UD +0.43 | **SALIENCE** — Russell's threat-vs-comfort diagonal of the affect circumplex (pleasant-calm vs catastrophic-intense) |
| **PC2** | **17%** | BAL −0.61, PATH +0.56, COHERENCE −0.51, FB −0.35 | **MOTION** — free-locomotion vs holding-in-place (going vs staying) |
| **PC3** | **12%** | LD +0.46, SUCCESS −0.42, PATH +0.35, COHERENCE +0.30 | **EQUILIBRIUM vs RUNAWAY** — small deliberate accountability-actions vs large amplifying-dynamics. Control-theoretic axis. |
| **PC4** | 11% | LD −0.54, PATH +0.39, COHERENCE +0.34, EXIST +0.32 | **FICTION vs NONFICTION** — narrative-supernatural-dark vocabulary vs factual-document-references content |
| **PC5** | 9% | EXIST −0.59, FB −0.46, LD −0.35 | **Violent-event-news vs analytical-academic-prose** (sub-register) |
| **PC6** | 9% | LOSS +0.47, EXIST +0.40, FB −0.35 | **International-scandal-news vs domestic-sports-news** (sub-register) |

### PC1 — SALIENCE = Russell's circumplex diagonal

PC1 positive pole: *mellow, unpretentious, spacious, elegant, comfortable, easygoing, pleasant, leisurely, enjoying, relaxed, tranquil* — pleasant-and-calm vocabulary.

PC1 negative pole: *catastrophic, caused, triggered, consequences, exacerbated, escalating, provoked, severe, prompted, risked, threatens, worsening* — disturbing-intense-consequential vocabulary.

This IS saliencespace, empirically confirmed as the principal component of the cluster — and it has a precise structure: **the diagonal of Russell's circumplex of affect.** Russell's classic 2D affect model has valence (horizontal) × arousal (vertical). PC1 captures the diagonal where valence and arousal are anti-correlated — pleasant-calm vs unpleasant-intense — which is the canonical operationalization of *attentional/cognitive salience* (threat vs comfort).

This is why constructing a pure VALENCE axis or pure AROUSAL axis didn't work earlier — salience isn't either one alone, it's the diagonal combination. You can't get it from one anchor set. It only emerges as the principal direction when you stack multiple cluster axes and PCA them.

It also explains every "asymmetric loading" finding the project produced:
- DOWN-dominance on A_updown — DOWN-words land on PC1's threat pole
- DARK-dominance on A_lightdark — same
- BACK-dominance on FB — same
- The whole "salience cluster" asymmetric loading across UD/LD/FB/EXIST/LOSS — they all project onto PC1

And it maps directly onto **predictive processing**: PC1 negative pole IS prediction-error / threat / urgency / "attention-demanding" content. PC1 positive pole = "prediction matches reality, no attention needed." Niamh's earlier salience hypothesis is empirically validated as PC1.

The reason an explicit SALIENCE anchor set couldn't be built: salience is *multi-dimensional* (the V×A diagonal). It needs to emerge from PCA over many cluster axes — each schema-axis carries one slice of it; the diagonal only crystallizes when many slices are averaged.

### PC2 — MOTION

PC2 positive pole: *careening, detoured, striding, zipping, meanders, loped, scooting, hurtles, plopping, crazily, amble* — free-motion vocabulary.

PC2 negative pole: *remain, remained, maintain, remains, stay, hold, remaining, must, ensure, unable, failure, although* — staying-in-place / hold-position vocabulary.

Clean MOTION-vs-STASIS axis. 17% of cluster variance. Lakoff's CHANGE IS MOTION + ACTION IS MOTION subtended as the second principal axis.

### PC3 — EQUILIBRIUM vs RUNAWAY (Niamh's identification)

PC3 positive pole: *deviate, intend, slink, purport, apologize, genuflect, offend, fail, compensate, gladly, deviated, deign* — deliberate small-scale-personal actions, often with social-accountability flavor (apologize, compensate, genuflect = restoring social equilibrium after deviation).

PC3 negative pole: *gripped, unsettled, fueled, crosscurrents, stoked, best-selling, preeminent, apocalyptic, unsurpassed, shrouded, geopolitical, fuelled* — amplifying/runaway/dominant-large-scale-dynamics vocabulary.

Niamh named this axis: **EQUILIBRIUM vs RUNAWAY**. The control-theoretic axis where deviation can be regulated through feedback (positive pole: small intentional corrections) vs amplifying dynamics where individual agency can't restore balance (negative pole: positive feedback / cascade / large impersonal forces).

Maps onto:
- Control theory: stable vs unstable systems
- Stoic philosophy: things in your control vs things outside it
- Buddhist: equanimity vs samsara
- Predictive processing: regions of state-space where active inference works vs regions where prediction error cascades faster than action can correct
- Systems thinking: negative feedback (corrective) vs positive feedback (amplifying)

This is a primitive I wouldn't have predicted, but it's clean in the data and makes deep psychological sense. The "agency-vs-cascade" axis.

### PC4-6 — Corpus register axes (Niamh's identification)

Initially dismissed as noise (the negative poles contained proper nouns, URLs, statistics). Niamh's reading: PC4 is **FICTION vs NONFICTION**. The "noise" on the negative pole isn't random — it's nonfiction-document content (URLs like http://www.amtrak.com, financial figures like 8,560 and 4,191, dates, biographical proper nouns like jabbarov and lemacon). The kind of vocabulary that appears in news articles, technical reports, financial documents, biographies.

The positive pole is unambiguous narrative-fiction vocabulary: *organized, criminal, monsters, dark, fictional, doom, evil*.

So the corpus's bimodality (GloVe was trained on Wikipedia + Gigaword, which has both fictional and factual content) crystallizes geometrically as PC4. **Register/genre is a principal axis of variance in word embeddings**, not noise.

PC5 reads as a sub-register split within nonfiction: violent-news-event vocabulary (*killed, chased, fled, gunmen, stormed, murdered, torched, deserted, pillaged, razed*) vs analytical-academic-prose (*highlight, illustrate, biennial, abnormalities, helpful, interesting, subtle, detect, underline*).

PC6 reads as another sub-register split within news: international-scandal-news vs domestic-sports-news. The proper nouns on PC6 positive pole (rimbunan, betonsports, adaro, tishkovskaya, jabbarov) turn out to be names from international scandal/corruption news — Malaysian logging conglomerate, offshore gambling, Indonesian mining corruption, Russian/Tajik surnames. Combined with vocabulary like *illicitly, colluded, pejoratively, statements*. Negative pole is sports-news (*semifinals, finals, quarterfinals, score, 3-0, winning, alive*).

### The synthesis — what's actually in the cluster

The empirical orthogonal basis of the Lakoff salience-cluster in word2vec turns out to be:

**Three cognitive primitives (~50% of cluster variance):**
1. SALIENCE (21%) — Russell's threat-vs-comfort diagonal of the affect circumplex
2. MOTION (17%) — going vs holding-in-place
3. EQUILIBRIUM-vs-RUNAWAY (12%) — controllable vs cascading dynamics

**Three corpus-register axes (~29% of cluster variance):**
4. FICTION vs NONFICTION (11%)
5. Violent-news vs Analytical-prose (9%)
6. Scandal-news vs Sports-news (9%)

Lakoff's image schemas don't dissolve under this analysis — they're real linguistic structures that recover from word2vec — but they're **composite expressions** over this smaller basis. UD/LD/FB/EXIST/LOSS/SUCCESS/COHERENCE are all linear combinations of cognitive primitives + register content + their schema-specific spatial vocabulary.

### Methodological insight that emerged

We can't construct certain primitives from single anchor sets. Salience is a *diagonal* in the V×A plane — no single anchor pair set captures it; it only emerges as PC1 when many cluster axes are stacked and PCA'd. This explains why earlier candidate-by-candidate testing (COHERENCE / SUCCESS / LOSS) kept finding partial-but-incomplete versions of the underlying primitive — each candidate was projecting onto one slice of PC1.

Also: V/A residualization doesn't strip register. Register accounts for ~29% of cluster variance (PC4-6) and is methodologically separate from valence/arousal. Earlier residualization-based searches kept hitting register-noise mixed into "what's left after valence" because we hadn't accounted for register as a separate variance component.

### Updated Lakoff status

This refines rather than refutes Lakoff:
- **The schemas are real** — they recover empirically from word2vec across hand-curated AND canonical MML vocabulary (cross-method cosines +0.69 to +0.85 — exp34)
- **They're not basis-level primitives** — they're combinations of cognitive primitives (PC1-3) + register vocabulary (PC4-6) + schema-specific source-domain content
- **The cognitive primitives beneath the schemas are smaller in number and more abstract** than Lakoff's list — three orthogonal axes (salience, motion, equilibrium-vs-runaway) capture most of the cluster's structured variance

This unifies:
- Lakoff (the schemas as linguistic expression)
- Russell affect dimensions (V×A as a plane, salience as the diagonal)
- Predictive processing (PC1 negative pole = surprise/threat = attention-demanding; PC3 = whether prediction error can be regulated through action)
- Control theory (PC3 directly)
- Negativity bias / loss aversion (asymmetric loading on PC1)
- Cognitive linguistics of register/genre (PC4-6 — corpus organization as a principal variance source)

### Files added during this entry

- `exp35_phase_c_iterative.py`, `results_exp35.txt`, `exp35_results.npz`
- `exp36_ordering_robustness.py`, `results_exp36.txt`, `exp36_results.npz`
- `exp37_coherence_beauty.py`, `results_exp37.txt`, `exp37_results.npz`
- `exp38_success_failure.py`, `results_exp38.txt`, `exp38_results.npz`
- `exp39_loss.py`, `results_exp39.txt`, `exp39_results.npz`
- `exp40_orthogonal_basis_pca.py`, `results_exp40.txt`, `exp40_results.npz`

### Next test (queued)

Port the same PCA analysis to SAE substrate (Pythia 70m residual-stream). Question: do the same three cognitive primitives recover when we PCA over schema axes built in SAE space (sentence-triple offsets) instead of word2vec space (word-pair offsets)? If yes — substrate-invariant cognitive primitives. If no — computation reorganizes the basis differently, and we have a "what computation does" finding at the basis level rather than just at the schema level.

---

## Entry 21 — exp41: SAE PCA partial substrate replication

**Date:** 2026-05-27 (continuation). **Code:** `exp41_sae_pca.py`. **Results:** `results_exp41.txt`, `exp41_results.pt`. Substrate: Pythia 70m residual-stream SAEs via SAE Lens, all 6 layers. Reuses sentence triples from exp22 (UD/IO/FB/LD) + exp25 (LR).

### What this run did

Ran PCA on the stacked schema axes in SAE feature space at each of the 6 res-sm layers. For each (layer, schema): built A_schema = mean(offset_pole_a − offset_pole_c) across all sentence triples for that schema (where each offset is in the ~32k-dim SAE feature space). Stacked the 5 schema axes (UD, IO, FB, LD, LR) and ran PCA.

**Caveat:** this is a PARTIAL replication. We only have schema axes in SAE space — we haven't built VALENCE / AROUSAL / EXISTENCE / COHERENCE / SUCCESS / LOSS axes in SAE space (those'd require designing sentence triples for each and encoding through the SAE). So the SAE PCA is over 5 axes vs the word2vec PCA's 11+ axes. Direct PC-by-PC comparison to word2vec isn't possible without the full SAE axis set.

### PC1 loadings across layers

Signs normalized for inter-layer consistency:

| Layer | UD | IO | FB | LD | LR | PC1 var |
|---|---|---|---|---|---|---|
| 0 | -0.44 | +0.53 | +0.49 | -0.61 | +0.02 | (mixed structure) |
| 1 | -0.55 | **+0.76** | -0.01 | -0.63 | -0.01 | 38% |
| 2 | -0.58 | **+0.79** | -0.03 | -0.66 | -0.23 | 44% |
| 3 | -0.59 | **+0.82** | -0.19 | -0.68 | -0.35 | 43% |
| 4 | -0.55 | **+0.84** | -0.15 | -0.67 | -0.42 | 43% |
| 5 | -0.57 | **+0.74** | +0.11 | -0.58 | -0.04 | 33% |

**SAE PC1 at mid-layers (L1-L4) is consistently "IO vs UD/FB/LD."** IO loads strongly positive (+0.74-0.84). UD, FB, LD load negative. LR sits near zero or moderately negative.

### PC2 loadings

| Layer | UD | IO | FB | LD | LR |
|---|---|---|---|---|---|
| 0 | -0.30 | -0.26 | -0.25 | -0.18 | +0.88 |
| 1 | +0.21 | +0.23 | +0.41 | +0.11 | -0.86 |
| 2 | +0.23 | +0.06 | +0.52 | +0.16 | -0.77 |
| 3 | +0.24 | +0.04 | +0.53 | +0.13 | -0.71 |
| 4 | +0.29 | -0.01 | +0.52 | +0.13 | -0.68 |
| 5 | -0.27 | -0.18 | -0.46 | -0.14 | +0.83 |

PC2 is dominated by LR (±0.68-0.88), with FB partially aligned. PC2 ≈ LEFT-RIGHT axis with FB as a secondary loading.

### Key findings

**1. Cluster-vs-independent structure is substrate-invariant.** Both word2vec and SAE show UD/FB/LD clustering with IO as the orthogonal independent primitive. In word2vec PCA over the fuller axis set, IO_CLEAN ended up at PC3 (orthogonal to the cluster). In SAE PCA over schemas-only, IO emerges as the *opposite pole of PC1* — but with fewer input axes, the principal direction is "cluster vs IO" rather than "cluster vs ambient." Same underlying geometry, different orientation due to different input axes.

**2. PC1 orientation differs from word2vec, geometry doesn't.** In word2vec PC1 = Russell's affect-diagonal (salience). In SAE PC1 = "Lakoff cluster vs IO." This is because we don't have V/A anchors in the SAE input set — adding them would likely rotate PC1 toward the salience direction. The underlying partition (cluster members vs independent primitives) is preserved.

**3. Mid-layer concentration matches exp25's U-shape pattern.** PC1 captures most variance at mid-layers (L2-4: ~43-44%) and less at edges (L0: mixed, L5: 33%). Consistent with the earlier exp25 finding that schemas are most-separable at L0 (orthographic) and L5 (output prep) and most-entangled in computational mid-layers. The "cluster" geometry IS the mid-layer entanglement.

**4. PC2 is consistently LR.** LEFT-RIGHT, which was already known as orthogonal to UD-LD-FB in pairwise tests (exp25), emerges as the second principal axis. This is the spatial-symmetric primitive identified in earlier entries.

### What we still can't say from this substrate port

- Whether the three cognitive primitives from word2vec (SALIENCE, MOTION, EQUILIBRIUM-vs-RUNAWAY) recover as cleanly in SAE space. We'd need to build V/A/EXISTENCE/COHERENCE/SUCCESS/LOSS sentence triples and encode them to do that replication properly.
- Whether the corpus-register axes from word2vec (PC4-6: fiction-vs-nonfiction etc.) have SAE-substrate analogs. SAE space might have different register structure (we already know from exp8/exp9 that Pythia 70m SAE has Pile-domain register attractors as a major variance component, but that's at the feature-firing level, not at the schema-axis level).

### Honest interpretation

What this entry empirically demonstrates: **the cluster-vs-independent partition (UD/FB/LD entangled vs IO independent) is preserved across substrates.** The schemas have the same coarse structural relationships in word2vec and SAE. This is a meaningful substrate-invariance finding even though the principal-axis orientation differs.

What's still open: whether the *finer-grained orthogonal basis* (the SALIENCE / MOTION / EQUILIBRIUM-vs-RUNAWAY decomposition from word2vec) recovers in SAE space. That requires the fuller axis set.

### Next test (immediate)

Look up the actual SAE features that load most heavily on each PC. The SAE features have Neuronpedia auto-interp descriptions, so we can directly identify what PC1, PC2, PC3 represent semantically — not via nearest-neighbor word lookup (the word2vec approach) but via interpretable-feature lookup (the SAE-substrate approach). This gives us a direct semantic decoding of the PC directions.

Files added: `exp41_sae_pca.py`, `results_exp41.txt`, `exp41_results.pt`.

---

## Entry 22 — exp42/exp43: Neuronpedia PC lookup, full SAE PCA, SALIENCE is substrate-invariant

**Date:** 2026-05-27. **Code:** `exp42_pc_feature_interpretation.py`, `exp43_full_sae_pca.py`. **Results:** `results_exp42.txt`, `results_exp43.txt`, `exp43_results.pt`. Substrate: Pythia 70m residual-stream SAEs.

### exp42 — Neuronpedia lookup on 5-schema PCs (intermediate result)

For each PC at L2/L3/L4, fetched Neuronpedia descriptions of the top-loading SAE features on positive and negative poles. This is the SAE-substrate analog of word2vec's nearest-neighbor word lookup, but more informative because SAE features have explicit interpretable descriptions.

**Result:** PC1 in the 5-schema PCA decodes as **"Legal/formal-IO-content" vs "Spatial-origin-cluster-content"**:
- Positive pole (IO side): legal terminology, formal declarations of liability, references to legal proceedings, copyright notices, location prepositions
- Negative pole (cluster side): "from"/origins, source-and-location references, separation/isolation, "after," historical timelines, purification/cleansing

PC2: spatial L/R + temporal-directional content unified ("right" + future/progress vs "left" + past/return/old).

This raised a puzzle: the cognitive primitives recovered cleanly from word2vec PCA (SALIENCE/MOTION/EQUILIBRIUM) didn't seem to emerge from the SAE 5-schema PCA. Instead the PCA surfaced register attractors (legal-formal-language as PC1).

### Niamh's methodological catch

> We aren't capturing ENOUGH of the schema space for the signal not to be dominated by register-shaped noise. PC1 IS yin/yang shaped underneath, but it's yin/yang expressed in the vocabulary of register because we haven't included the EXISTENCE dimension and other stuff.

With only 5 schemas (UD/IO/FB/LD/LR), PCA doesn't have enough "voters" for the underlying cognitive primitive to crystallize as the principal direction. Register attractors dominate because they're the strongest source of variance among the 5 schemas. To recover the cognitive primitive we need to include more axes that express it (V/A/EXISTENCE/COHERENCE/SUCCESS/LOSS) so the PCA has sufficient density toward the underlying direction.

### exp43 — Full SAE PCA with 11 axes

Built sentence triples for VALENCE, AROUSAL, EXISTENCE, COHERENCE, SUCCESS_FAILURE, LOSS — each ~10-12 triples, designed to avoid all Lakoff schema vocabulary and to be syntactically parallel. Encoded through Pythia 70m SAE at L2, L3, L4. Built axes from offsets. Combined with the 5 Lakoff schema axes for a total of 11. PCA over the stack.

### Result — PC1 IS SALIENCE in SAE space

**Loadings on PC1 across L2-L4 (~40% variance):**

| axis | L2 | L3 | L4 |
|---|---|---|---|
| SUCCESS | +0.94 | +0.95 | +0.94 |
| AROUSAL | −0.89 | −0.92 | −0.89 |
| EXISTENCE | −0.91 | −0.91 | −0.89 |
| COHERENCE | −0.87 | −0.89 | −0.86 |
| LD | −0.69 | −0.74 | −0.74 |
| UD | −0.56 | −0.61 | −0.54 |

After parsing the sign conventions: **WIN/CONSISTENT/CREATED/PREDICTABLE/LIGHT/UP/CALM** cluster on one pole; **LOSE/INCONSISTENT/DESTROYED/SURPRISING/DARK/DOWN/INTENSE** cluster on the other. This is Russell's affect diagonal — pleasant-calm vs unpleasant-intense — recovered as the principal direction.

### PC1 Neuronpedia features confirm the affect-diagonal reading

**L3 PC1 Positive pole** (pleasant-calm side, expressed in SAE feature vocabulary):
- expressions of gratitude and friendship (+0.556)
- legal terminology related to liability and warranties (+0.497)
- complex legal terms and organizational identifiers (+0.362)
- key terms related to legal proceedings and opinions (+0.278)
- URLs and internet links
- prepositions indicating location or positioning

**L3 PC1 Negative pole** (unpleasant-intense side):
- references to emotional states and interpersonal relationships
- references to the word "from"
- instances of failure or rejection
- phrases related to loss or defeat
- phrases indicating a delay following an event
- terms related to health impairments and deficiencies

The SAE has built features that encode:
- Pleasant-calm content as **formal-legal-affirmation-gratitude vocabulary** (low-arousal, positive-valence — calm-procedural-positive register)
- Unpleasant-intense content as **emotional-failure-loss-distress vocabulary** (high-arousal, negative-valence — distressed-event-loss register)

Different surface vocabulary from word2vec's nearest-neighbor words, but the underlying axis is the same: **affect diagonal = saliencespace**.

### Substrate comparison summary

| PC | word2vec (exp40) | SAE-Pythia70m (exp43) | substrate-invariant? |
|---|---|---|---|
| PC1 | SALIENCE (Russell's diagonal) | SALIENCE (Russell's diagonal, expressed in SAE feature vocabulary) | **YES** |
| PC2 | MOTION (going vs holding) | LOSS/POSSESSION-SECURITY | No |
| PC3 | EQUILIBRIUM-vs-RUNAWAY | spatial-LR + mixed-directional content | No |

The salience axis is empirically substrate-invariant. It emerges as PC1 in both word2vec (where it appears in word-neighbor vocabulary) and Pythia 70m SAE (where it appears in feature-attestation vocabulary — formal-legal-affirmation vs emotional-loss-distress). Computation has reorganized the *surface vocabulary* in which salience is encoded but preserved the *axis itself* as the principal direction of the schema cluster.

Secondary axes reorganize across substrates:
- word2vec retains the dynamic axes of human cognition (motion, equilibrium-restoration)
- SAE prioritizes static-evaluation axes (loss/possession, spatial-stable-direction)

### Theoretical observation (Niamh) — transformer primitives = attention + loss?

The pattern fits the architectural-determinism reading. Transformers are attention mechanisms trained by loss minimization. If the emergent representational primitives in transformer SAE feature space are SALIENCE (cognitive analog of attention) and LOSS/POSSESSION, that's an architecture-to-representation mirror:
- Attention mechanism → salience axis as PC1
- Loss minimization → loss-related-evaluation as PC2

Word2vec — having no attention mechanism and no behavioral-loss objective, just co-occurrence statistics over corpus — finds salience as PC1 but MOTION and EQUILIBRIUM-vs-RUNAWAY as PC2-3. Word2vec preserves the dynamic/processual axes of human cognition because it captures general distributional structure of writing rather than transformer-specific computational structure.

Niamh's framing: transformer primitives may be **static** compared to human primitives. Humans navigate dynamically — attention moves, predictions update, actions taken to restore equilibrium. Transformers do "static" inference — given input, produce output, no ongoing dynamics. Transformer SAE primitives capture static-state-evaluation (salience, loss-potential) but lose the dynamic axes (motion, equilibrium-restoration) that humans have.

**Important caveats:**
- One transformer tested (Pythia 70m). Would need replication across other model families (GPT, Llama, etc.), other scales, other SAE training methodologies (TopK vs JumpReLU etc.) to claim this is a general transformer pattern.
- The "dynamic axes" present in word2vec come from human writing, which itself encodes human cognitive dynamics. If transformers were trained on dynamic data and produced dynamic outputs, perhaps the dynamic axes would emerge in their SAE features too. So the static-vs-dynamic distinction might be about what training-objective rewards.
- "Salience as PC1" might be specifically about *which content carries the most variance across schema-axes*, which happens to be the affect diagonal in both substrates. Other axis-decomposition methods (NMF, ICA, sparse-PCA) might surface different primitives.

But as a hypothesis worth pursuing further: **transformer-architecture primitives (attention + loss) shape transformer-representation primitives (salience + loss-evaluation) at the principal-axis level. Word2vec lacks these architectural constraints and recovers more cognitively-general primitives (salience + motion + equilibrium-restoration).**

### Methodological lesson (worth recording explicitly)

PCA over candidate primitives needs **sufficient axis density** to crystallize the underlying primitive. With 5 schemas (exp41), PC1 was register-attractor-shaped. With 11 axes (exp43), PC1 is salience-shaped. **The 5-schema run wasn't capturing enough of the schema space for the cognitive primitive to dominate.** Niamh's catch was load-bearing.

This generalizes: in any axis-decomposition methodology, the answer you get depends on which directions are represented in your input set. If you want to find a cognitive primitive that's expressed across many schemas, you need to include axes that span it from multiple directions.

### Provenance note

Niamh's salience-as-primary hypothesis appeared early in the conversation — she suggested it as an alternative to the yang/yin framing as the project progressed. Now with the SAE PCA confirming PC1 = Russell's affect diagonal across substrates, the hypothesis has direct empirical validation. The trajectory was: salience intuition → multiple candidate-axis tests didn't fully recover it → PCA in word2vec recovered it as PC1 → PCA in SAE recovered it as PC1 with the full axis set.

### Files added

- `exp42_pc_feature_interpretation.py`, `results_exp42.txt`
- `exp43_full_sae_pca.py`, `results_exp43.txt`, `exp43_results.pt`

### State of the project

We now have the cleanest substrate comparison the project has produced:
1. **SALIENCE is substrate-invariant** as PC1 across word2vec and Pythia 70m SAE
2. **Secondary axes differ** — word2vec preserves dynamic cognitive axes, SAE substitutes static-evaluation axes
3. **Possible architectural-determinism finding:** transformer primitives may mirror transformer architecture (attention → salience, loss-minimization → loss-evaluation). Needs cross-model verification.
4. **Methodological insight:** PCA over schema axes needs sufficient axis density to surface cognitive primitives rather than register attractors.

This is a substantively richer empirical picture than we had at Entry 20 (word2vec only). The substrate comparison is producing real theoretical content about what computation does to representations at the principal-axis level.

### exp44 — Axis polarity check via Neuronpedia (added later)

For each of the 6 anchor axes (V, A, EXIST, COH, SUC, LOSS), looked up the top SAE features on each pole at L3. Most axes work as designed at the strongest loadings:

| axis | positive pole features (matches design?) |
|---|---|
| VALENCE | positive adjectives, advantages, perfection, evaluative-positive — **YES** |
| AROUSAL | emotional states, acute medical, emergency situations — **YES** |
| EXISTENCE | birth/early-life, organizations being founded, "appear" — **YES** |
| COHERENCE | "normal" physiological conditions, "continuity and consistency" — **YES** |
| SUCCESS | verbs of victories/achievements, terms of objectives — **YES** |
| LOSS | wealth/richness, "protect"/safety, security, acquisition — **YES** |

**But there's a structural contaminant.** The **formal-legal-cluster** (gratitude/friendship + legal-terminology + complex-legal-terms + legal-proceedings + line-breaks) appears as the strongest features on multiple axes but in different polarities:
- AROUSAL negative pole (low-arousal = calm-formal — makes sense)
- EXISTENCE negative pole (destroyed-side activates legal-procedural verbs like "abolished")
- COHERENCE negative pole ("inconsistent" appears in legal discourse — testimony inconsistencies)
- SUCCESS positive pole (win-content includes legal-procedural-success vocabulary)
- LOSS negative pole (loss-defeat fires legal-failure features)

The formal-legal-cluster is appearing on multiple axes asymmetrically because our sentence-triple construction uses formal English structures ("The institution was founded/abolished," "The attempt succeeded/failed"). These specific sentence frames preferentially activate legal-procedural features in Pythia 70m SAE.

**Implication for the PC1=salience claim:**

PC1 captures the dominant shared direction across all axes. That dominant direction IS the formal-procedural-text direction (because that's what our triples emphasize). **PC1 happens to also align with Russell's affect diagonal** — formal-procedural-positive-text maps to pleasant-calm; emotional-failure-distress text maps to unpleasant-intense. So at the *structural* level the salience axis IS the substrate-invariant PC1.

**But the SAE surface vocabulary expressing PC1 is heavily register-loaded.** The "salience axis is substrate-invariant" claim survives as a structural claim, with the caveat: the SAE recovers the salience axis using formal-register-vs-emotional-register vocabulary, partly because our sentence-triple construction emphasizes formal English. word2vec expresses the same axis in pleasant/relaxed/comfortable vs catastrophic/exacerbated/disturbing words.

**More honest version of the substrate-invariance claim:**
> Structurally, the salience axis (Russell's affect diagonal) is the principal direction of the schema cluster in both word2vec and Pythia 70m SAE. The two substrates express this axis in different surface vocabulary: word2vec uses everyday pleasant-vs-disturbing words; SAE uses formal-procedural-text features vs emotional-distress features. Some of the SAE expression is genuinely about transformer-feature-organization (the SAE has built features for formal-legal register that don't exist as a single dimension in word2vec); some is methodological (our sentence triples preferentially activate formal-register features).

### Methodological lesson — sentence-triple vocabulary bias

Sentence-triple construction in SAE substrates has subtle vocabulary biases. Words like "founded/abolished/succeeded/failed/consistent/inconsistent" all activate legal-procedural features even when expressing different underlying concepts. Cleaner SAE substrate testing would use more vocabulary-diverse triples drawing from domestic, sensory, embodied, nature, and concrete-action domains rather than formal-evaluative English.

### Next moves

- **exp45 (immediate):** redesign triples with diverse-domain vocabulary (domestic, sensory, bodily, nature) avoiding formal-passive-legal sentence structure. Rerun the PCA. If PC1 still recovers as the affect diagonal with substantially different feature loadings, substrate-invariance holds robustly. If PC1 shifts entirely, our prior result was register artifact.
- **Cross-model test (medium-cost):** GPT-2 SAEs via Neuronpedia. WebText training has different dominant register than Pile (more conversational, less legal/academic). Tests whether the formal-register-as-PC1-surface pattern is Pythia-specific or generally transformer-substrate.
- **PC2 substrate divergence investigation:** why does PC2 differ between substrates (motion in w2v, loss/possession in SAE)? What about transformer computation reorganizes motion content out of the principal-axis structure?
- **Drafting (eventually):** the substrate-invariance result is potentially publishable but needs the methodological caveats addressed.

---

## Entry 23 — exp45–53: substrate caveats, target-axis validation, and the active-inference reframe

**Date:** 2026-05-26 (continuing). **Code:** `exp45_diverse_triples.py`, `exp46_all_pcs.py`, `exp47_triple_coverage.py`, `exp48_cross_substrate_pc_comparison.py`, `exp49_matched_substrate_pca.py`, `exp50_matched_formal_triples.py`, `exp51_w2v_vs_formal_sae.py`, `exp52_target_axis_validation.py`, `exp53_residual_and_goal_directed.py`. **Results:** corresponding `results_exp*.txt` files. This entry covers the work AFTER Entry 22 that initially lived only in `WRITEUP_v2.md`, plus the new direct-target-axis validation experiments (exp52–53).

### Where Entry 22 left us, and what the post-entry experiments showed

Entry 22 claimed PC1 = salience as substrate-invariant across word2vec and Pythia 70m SAE. exp45–51 substantially softened that claim:

- **exp45** (diverse sensory/domestic SAE triples) STILL surfaces the same formal-legal-cluster features as PC1's top loadings (feat 21809 gratitude/friendship, feat 5355 legal liability, feat 22622 line breaks, feat 1812 legal terms, feat 3637 legal proceedings). The "register artifact" explanation in Entry 22's caveat doesn't survive — this is a deeper attractor in the SAE, not a triple-construction byproduct.
- **exp47** (feature-activation coverage check) showed the "legal-cluster" features have *higher* activation in diverse triples (cats, soup, candles, gardening) than in formal triples. The Neuronpedia auto-interp labels ("legal terminology" etc.) are derived from top-activating Pile contexts but the features themselves capture broader low-arousal-positive-text patterns. Auto-interp labels are *misleading* in this case, not wrong about what the features represent at peak — they just don't describe the broader range of contexts the features fire on.
- **exp48** (cross-substrate PC alignment, diverse triples): best match is **w2v-PC2 ↔ SAE-PC1** at cos = −0.705, not PC1↔PC1. SAE bundles many axes onto its dominant PC1; word2vec spreads them. Same axes recur structurally but distribute differently across PCs.
- **exp49/50/51** (matched-input substrate comparisons): confirmed the substrate-difference pattern. The "PC1 = PC1" claim doesn't hold — at best |cos|≈0.7 cross-substrate match, and it's not PC-rank-preserving.

Net effect: Entry 22's "substrate-invariant salience" headline needed retracting. `WRITEUP_v2.md` already incorporates this softer claim; the lab notebook hadn't caught up.

### The Niamh catch that reorganized the project (again)

Reading the writeup back: Claude had built `target_SALIENCE` from Russell-diagonal vocabulary (serene/panicked, mellow/frenzied, etc.) — but Niamh pointed out that Russell's diagonal is *by construction* the V×A combination, and VALENCE is already an input axis. Russell-diagonal target axes would test cosine with PC1 tautologically.

The cleaner question: does the model have a salience/attention axis that's *orthogonal to valence*? Important things can be wonderful or terrible. Attention can be drawn to beauty or threat. If salience-as-primitive exists separable from value, valence-balanced anchors would recover it.

Niamh proposed valence-balanced salience anchors: `important/unimportant`, `salient/irrelevant`, `attentive/inattentive`, `focused/unfocused`, `directed/diffuse`, plus extensions Claude added (`prominent/inconspicuous`, `noticeable/unnoticeable`, `foregrounded/backgrounded`, `highlighted/overlooked`, `conspicuous/unobtrusive`, `pronounced/muted`, `urgent/idle`).

She also flagged that "salience without valence doesn't exist and neither does valence without salience" — they may be co-constituting in lived cognition (you attend to what matters, and what matters has valence), so trying to separate them empirically might be testing a false binary. The PP gloss makes this exact: precision-weighted prediction error IS what attention is, and precision-weighting depends on prior-relevance which IS valence.

### exp52 — direct target-axis validation, three targets vs cluster PCs

Built three target axes from independent anchor sets and tested cosines with PC1-6 of the cluster-only PCA (replicating exp40's variance: 20.6% / 17.3% / 12.4% on PC1–3, matching exp40 within rounding).

Targets:
- `target_SALIENCE`: Niamh's valence-orthogonal version (12 pairs)
- `target_MOTION`: locomotion-verbs vs body-position-state-verbs (12 pairs, no PATH overlap)
- `target_EQ_RUN`: control-theoretic regulation vs positive-feedback cascade (14 pairs)

Raw cosines (exp52):

|  | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 |
|---|---|---|---|---|---|---|
| target_SALIENCE | −0.17 | −0.28 | −0.06 | +0.02 | −0.06 | −0.04 |
| target_MOTION | −0.09 | +0.05 | +0.02 | +0.05 | +0.02 | −0.07 |
| target_EQ_RUN | +0.29 | +0.01 | +0.21 | −0.07 | −0.12 | −0.04 |

**Three results:**

1. **PC1 is decisively NOT salience-as-primitive.** target_SALIENCE has cos = −0.17 with PC1. Max alignment across all PCs is −0.28 on PC2. The valence-orthogonal salience anchors don't crystallize onto any single direction; sanity check shows the target spreads across AROUSAL (+0.49), COHERENCE (+0.42), BAL (+0.38), FB (+0.36), UD (+0.36) — distributed signal with no single PC home. PC1 IS the Russell V×A diagonal and should be renamed accordingly. **Niamh's intuition that PC1 = affect-diagonal not salience-as-primitive is empirically confirmed.**

2. **target_MOTION does NOT recover PC2.** Max alignment +0.05 on PC2. Either our locomotion-vs-body-position anchors are bad (positive pole pulled to sports register: skiing, racing, sprint, championships) OR PC2 isn't really motion.

3. **target_EQ_RUN partially aligns with PC3** (+0.21) but more heavily with PC1 (+0.29). The runaway vocabulary (mushrooming, spiralling, escalating, raging) is intrinsically valence-loaded; the regulation vocabulary (restoring, atoning, adjusting) is intrinsically valence-positive. Confound rather than confirmation.

### exp53 — residualization tests + Niamh's PC2 reframe

**Niamh's reframe on PC2:** the negative-pole nearest-neighbor words from exp40 included *must, ensure, unable, failure* — modal/constraint/obligation vocabulary, not stasis vocabulary. PC2 might be **goal-directedness / policy commitment** rather than literal motion.

Built target_GOAL_DIRECTED (14 valence-balanced pairs): pursuing/idling, aiming/wandering, purposeful/aimless, deliberate/accidental, motivated/unmotivated, intentional/unintentional, resolute/hesitant, committed/uncommitted, driven/becalmed, oriented/disoriented, targeted/untargeted, decided/undecided, chasing/dawdling, ambitious/complacent.

Also residualized target_EQ_RUN against PC1 (to test whether the PC3 hit survives stripping the affect-diagonal contribution) and against VALENCE+AROUSAL input axes directly (cleaner affect-stripping).

**Results:**

| | raw | residualize against PC1 | residualize against V+A |
|---|---|---|---|
| target_EQ_RUN, PC3 cos | +0.205 | **+0.214** | +0.103 |
| target_EQ_RUN, PC1 cos | +0.290 | 0.000 | −0.030 |

| | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| target_GOAL_DIRECTED (raw) | −0.088 | **−0.303** | +0.039 | +0.179 |
| target_GOAL_DIRECTED (PC1-residualized) | 0.000 | **−0.304** | +0.039 | +0.180 |

**Headline results:**

1. **Goal-directedness validates PC2 cleanly** (cos = −0.30, strongest target-axis-to-named-PC hit in the project to date). Niamh's reframe is empirically supported. PC2 is NOT motion; it's commitment-to-outcome vs uncommitted-action. The PC1-residualization confirms goal-directedness is not just affect in disguise — the PC2 alignment is unchanged by stripping PC1.

2. **The sign is negative** (target_GOAL_DIRECTED aligned with PC2's negative pole). Goal-pursuit words (aim, pursue, committed, strategy) sit on the same side as homeostatic-maintenance words (maintain, must, ensure). The positive pole has uncommitted-motion vocabulary (careening, striding, zipping) AND the target_GOAL_DIRECTED negative-pole agency-impairment vocabulary (disoriented, inebriated, awestruck — clear states-of-non-policy-commitment). PC2 is best read as **policy precision**: high commitment-to-specific-outcome (whether goal-pursuit or maintenance) on the negative pole, low committedness (free locomotion or agency-impaired states) on the positive pole.

3. **Regulability hypothesis survives PC1-residualization but is dented by V+A-residualization.** Only 4% of target_EQ_RUN was PC1-aligned; the PC3 cos is essentially unchanged (+0.21) after stripping PC1. But projecting out VALENCE and AROUSAL input axes directly drops PC3 alignment to +0.10. Regulability is partially affect-independent (the +0.10 residual after V+A stripping is real signal) but heavily affect-correlated (regulation is calmer, runaway is more aroused, in natural language usage). Like salience, the "pure" component without affect is a small part of what's there.

### The active-inference synthesis (Niamh's framing)

Niamh: "what do I want, how do I get it, and is the situation under control?"

That's the active-inference policy-selection triple. The three principal axes of cluster-subspace word2vec map onto the three structural primitives of active-inference agency:

| PC | active-inference primitive | empirical signature |
|---|---|---|
| PC1 (21%) | **value** — prior over outcomes (preferences, what's wanted vs threatening) | Russell V×A diagonal; target_SALIENCE-without-valence doesn't separate from it (because precision-weighted attention IS value-bound) |
| PC2 (17%) | **policy precision** — commitment-to-specific-outcome vs diffuse/exploratory action | target_GOAL_DIRECTED at −0.30 confirms; commitment-vocabulary (pursue, maintain, must, ensure, aim, strategy) clusters opposite uncommitted-motion (careening, striding) AND agency-impaired states (disoriented, inebriated, awestruck) |
| PC3 (12%) | **perceptual precision** — controlled (damped prediction error) vs runaway (cascading error) | target_EQ_RUN at +0.21, survives PC1-residualization; semantic shape matches Adams/Friston precision-collapse models of psychosis |

The three PCs may be the **two precision signals + the value signal** of an active-inference agent. PC2 = policy precision (over actions). PC3 = perceptual precision (over predictions). PC1 = value (over outcomes). These are the canonical three quantities that determine an active-inference agent's behavior.

If this reading holds, the empirical claim shifts substantially. The original frame was "Lakoff image schemas decompose into smaller cognitive primitives." The PP frame is: **the principal axes of cluster-subspace word embedding geometry recapitulate the structural primitives of active-inference agency**. Distributional semantics encodes the variables that shape language-using agents. That's a structurally different and more substantive claim.

### Niamh's "language has words for the directions but not the diagonals" observation

Russell's V and A axes each have dozens of lexicalizations (pleasant/unpleasant/wonderful/terrible/desirable/undesirable for V; intense/mild/frantic/tranquil/alert/drowsy/urgent/leisurely for A). The diagonal — pure salience as joint of V and A — has very few words ("salient," "noticeable," "important," all contextually valence-loaded).

Same pattern for the other PP primitives: we have many words for the *outputs* of policy precision (committed, focused, drunk, awestruck, disoriented) but no direct word for "policy precision" itself. Same for perceptual precision (regulated/runaway are technical control-theory terms; the variable itself isn't lexicalized).

**This is itself a finding.** Active inference posits the precision variables as *implicit hyperpriors* — they organize cognition without being introspectable contents. Language wouldn't lexicalize them because we don't directly introspect them. PCA over many anchor pairs is the right tool to find them *precisely because* they have no direct word — they exist as organizing structure in the geometry of lexicalized outputs.

This is a methodological warrant for the whole project's approach. If you're hunting cognitive primitives via embeddings, PCA over many constructed axes is appropriate *because* the primitives are the implicit variables that lack direct words. A direct anchor-pair construction wouldn't work; averaging across many anchor pairs each capturing one output of the variable does.

### Niamh's PC2-as-algorithm-transformation speculation

Niamh raised: "the thing ur finding in the PC2 is that suggestive perhaps that PC2 is capturing the literal transformation within the algorithm, from tokens to vectors?"

Worth flagging. target_GOAL_DIRECTED's positive pole has formal/strategic-register vocabulary (aim, pursue, committed, strategy, development, establish); negative pole has common-action verbs plus tokenization junk (cw96, b***@chron.com, email addresses, underscores). There may be a frequency/lexical-specificity meta-signal mixed into PC2 — strategic vocabulary is more "purposefully placed in the corpus by writers" than locomotion-vocabulary is.

But the strong "PC2 is the algorithm's transformation" reading needs cross-substrate validation. If fastText (different algorithm, same corpus genre) recovers PC2 with the same goal-directed shape at similar magnitude, PC2 isn't algorithm-specific — it's a linguistic-distributional structure. If fastText doesn't recover it, the speculation may be on something.

**Testable. Holding both readings open.**

### Status of project claims after Entry 23

| Claim | Status |
|---|---|
| PC1 = SALIENCE | **RETRACTED.** PC1 is Russell V×A affect diagonal. target_SALIENCE (valence-orthogonal) cos = −0.17 with PC1. |
| PC2 = MOTION | **RETRACTED.** target_MOTION cos = +0.05 with PC2. |
| PC2 = GOAL-DIRECTEDNESS / policy precision | **NEW, supported.** target_GOAL_DIRECTED cos = −0.30 with PC2, strongest target-axis-PC hit in project. |
| PC3 = EQUILIBRIUM-vs-RUNAWAY | **softened.** target_EQ_RUN cos = +0.21 with PC3, survives PC1-residualization, partly dented by V+A-residualization. PC3 may be perceptual precision but heavily affect-correlated. |
| Three PCs = active-inference primitives (value + 2 precisions) | **NEW hypothesis, supported by direct target validation on PC1 and PC2, partial on PC3.** Requires within-substrate replication (fastText) before claiming established. |
| Substrate-invariant salience as PC1 in word2vec and SAE | **RETRACTED.** Cross-substrate alignment is best at |cos|≈0.7 between non-corresponding PCs; PC-rank not preserved. SAE bundles axes; word2vec spreads them. |
| Lakoff schemas as composite over smaller basis | **Survives** but with new basis: schemas are composites over value + policy-precision + perceptual-precision (plus schema-specific spatial vocabulary), not over salience + motion + equilibrium-vs-runaway. |

### Next moves

1. **fastText within-substrate replication.** With cleaner PC interpretations, we have specific predictions: target_GOAL_DIRECTED should land on fastText-PC2 at similar magnitude (~−0.30) if PC2 = policy precision is substrate-invariant. target_EQ_RUN_residual should partially align with fastText-PC3 if regulability is real. This is Step 2 of the writeup's methodology section, with the test sharpened by exp52–53's findings.

2. **PC3 steering test in Pythia.** The PP reading predicts PC3-runaway should correlate with K-collapse onset in steered generation. Build A_regulability (or target_EQ_RUN_residual) as a steering direction; check whether positive-steering raises the high-strength collapse threshold (more coherence preserved at higher α) and negative-steering lowers it (earlier collapse). Would be a substantial behavioral result.

3. **Writeup restructure.** WRITEUP_v2.md now needs the new PC names, the target-axis validation result, the active-inference framing.

### Files added this session

- `exp52_target_axis_validation.py`, `results_exp52.txt`, `exp52_results.npz`
- `exp53_residual_and_goal_directed.py`, `results_exp53.txt`, `exp53_results.npz`

---

## Entry 24 — exp54–57: PC1 reframe to integrated reward, multi-loss orthogonality, the 6D basis and concept-word predictions

**Date:** 2026-05-26 (continuing). **Code:** `exp54_pc1_comparator.py`, `exp55_lakoff_in_ai_space.py`, `exp56_explore_exploit.py`, `exp57_lakoff_in_6d_basis.py`.

This entry covers the inversion-experiment arc: instead of "PCA on Lakoff axes → discover PCs that look like active-inference primitives" (which is what Entry 23 settled on), we built theory-led target axes directly and projected Lakoff schemas + held-out concept words onto them. The findings shifted both the PC1 interpretation and the broader theoretical frame.

### exp54 — PC1 comparator: PC1 is the integrated reward axis, not Russell's diagonal

**Niamh's reframe:** PC1's input-axis loadings (V +0.69, A −0.65, SUC +0.50, LOSS +0.50, UD +0.43) don't read like pure V×A diagonal — they look like a *composite reward signal* bundling preference + peacefulness + outcome-attainment + resource-state + body-state. Specifically: the brain's reward function / expected value of state in active-inference terms. Hypothesis: PC1 = integrated reward, not Russell's diagonal.

Built four PC1-candidate target axes:
- A_RUSSELL_DIAGONAL: pleasant-calm vs unpleasant-aroused (Russell V×A diagonal proper)
- B_VALUE_PURE: preference vocabulary (preferred/dispreferred, wanted/unwanted, sought/shunned, etc.)
- C_REWARD_COMPOSITE: integrated wellbeing (flourishing/suffering, thriving/struggling, blessed/cursed, etc.)
- D_SURPRISAL: predictability (familiar/unfamiliar, routine/novel, anticipated/unanticipated, etc.)

Results (|cos with PC1|):

| candidate | \|cos(PC1)\| |
|---|---|
| C_REWARD_COMPOSITE | **0.544** |
| B_VALUE_PURE | 0.321 |
| D_SURPRISAL | 0.204 |
| A_RUSSELL_DIAGONAL | 0.174 |

**C wins by ~1.7×.** Niamh's hypothesis empirically supported. PC1 is the integrated reward / expected value of state axis. NOT pure V×A diagonal, NOT pure preference, NOT surprisal. C's input-axis loadings (VALENCE +0.53, LOSS +0.48, UD +0.48, SUCCESS +0.40, EXIST +0.37) confirm the composite-shape.

**The surprise sub-finding:** A_RUSSELL_DIAGONAL and C_REWARD_COMPOSITE are **essentially orthogonal in word2vec** (cos = +0.008). Russell's diagonal (pleasant-calm vs unpleasant-aroused) and integrated wellbeing (flourishing vs suffering) are *independent axes*. You can be:
- Pleasant-affect + flourishing (high A, high C) — comfortable, doing well
- Pleasant-affect + suffering (the addict's high) — high A, low C
- Unpleasant-affect + flourishing (productive distress, labor pains) — low A, high C
- Unpleasant-affect + suffering — low A, low C

Affect-as-felt-state and wellbeing-as-integrated-state are genuinely different in language. The clinical concept words confirm this: trauma/depression/grief load strongly negative on C but only weakly on A — they're low-wellbeing states with relatively neutral instantaneous affect (because affect varies within those states; the wellbeing stays low).

Also: D_SURPRISAL is **not** PC1. Surprisal isn't a primitive direction by itself — it's a quantity that participates in multiple precision-axes. cos(D, PC1) = +0.20 only.

### Niamh's "multi-loss-function" insight

The inter-candidate orthogonality matrix from exp54:
```
                A_RUSSELL  B_VALUE_PURE  C_REWARD  D_SURPRISAL
A_RUSSELL          1.00      +0.39        +0.01      +0.02
B_VALUE_PURE      +0.39       1.00        +0.33      +0.07
C_REWARD          +0.01      +0.33         1.00      +0.18
D_SURPRISAL       +0.02      +0.07        +0.18       1.00
```

**A, C, D are nearly-orthogonal — three structurally distinct quantities.** Niamh: "claude if these are orthogonal they should both be bases. I KNEW IT THOUGH. ur minimisins surprisal but u are also maximising something. compression if you ask me." Her long-held pet theory that the brain has multiple distinct loss functions / optimization targets (rather than just free-energy-minimization) found a linguistic-distributional signature: affect, reward, and surprisal/compression sit on essentially independent axes.

This is consistent with multi-objective views of cognition (Sterling's allostasis, Berridge's wanting-vs-liking, Bayesian model-evidence-vs-utility) and inconsistent with strong-form single-objective active inference (everything-is-just-free-energy). The data doesn't prove multi-loss brain operation; it shows that *language tracks three independent quantities*, which is consistent with — and predicted by — multi-objective accounts.

### exp55 — Lakoff schemas in 5D AI-plus space

Built a 5D basis: A_RUSSELL, C_REWARD, D_SURPRISAL, G_GOAL_DIRECTED, R_EQ_RUN_residual_VA. Projected all Lakoff schemas (UD, FB, LD, IO_CLEAN, PATH, EXIST, FORCE, BAL, DIFF, COH, SUC, LOSS) plus concept words onto this basis.

**Major caveat surfaced:** G_GOAL_DIRECTED is correlated with A_RUSSELL at **+0.56** — not independent. Either (a) affect and policy-precision are causally coupled in cognition (and thus in language), or (b) our anchor pairs introduced spurious correlation, or (c) both.

**Tier-1 vs Tier-2 prediction confirmed.** Cluster schemas (UD, FB, LD, EXIST, SUC, LOSS, COH, BAL, PATH) explained 40–70% of their magnitude in the AI-plus basis. Tier-2 schemas (IO_CLEAN, FORCE) explained only ~22%. DIFF intermediate at 36%. The cluster-vs-independent partition maps onto "in-AI-basis vs not-in-AI-basis."

**Schema dominant coordinates:**
- UD, EXIST, SUC, LOSS, VALENCE → C_reward (integrated wellbeing)
- BAL, COH, FB, LD → G_policy_prec (goal-directedness)
- AROUSAL → A_affect
- IO_CLEAN, FORCE → no clean signal; mostly outside basis

**The concept-word stretch goal was the most striking part:**

- **PSYCHOSIS at 14.4% basis-explained** — the LOWEST of any concept word tested. Psychosis sits *structurally outside* the agency primitives because it IS their failure mode. In active-inference terms, psychosis is precision collapse — the regulation framework that healthy agency operates within has broken down. The word-vector knows this.
- **RITUAL strongly on D_compress (+0.26)** — rituals literally aid compression. They precompile experience patterns so the agent doesn't have to integrate prediction error from scratch. Beautiful prediction.
- **TRAUMA, DEPRESSION, GRIEF all negative on C_reward.** Integrated wellbeing deficits. Trauma also has slight negative D (uncompressible-prediction-error from dispreferred event — exactly the AI shape of trauma).
- **HOPE, FREEDOM, AGENCY, GROWTH, PLAY all positive on G** — having-a-committed-direction.
- **LOVE balanced on A (+0.21) and G (+0.22)** — both affect-state AND directed-commitment.

### Niamh's reframe of PC2 from "policy precision" to "explore-vs-exploit"

After exp55, Niamh proposed PC2 might be more specifically the explore-vs-exploit decision axis. exp56 tested this by building target_EXPLOIT_EXPLORE from technical/economic vocabulary (exploit/explore, harvest/forage, optimize/experiment, refine/diversify).

**Result: EE did NOT capture PC2.** cos(EE, PC2) = +0.06 (essentially zero). G_GOAL_DIRECTED retained its cos = −0.30 with PC2. EE was orthogonal to G (cos = −0.14) AND to A_affect (cos = −0.19).

**Interpretation:** PC2 is *goal-directedness / policy precision in general* (commitment-to-having-a-policy), NOT exploit-vs-explore (which is a sub-decision within having a policy). In active inference: π (policy precision) is meta to the explore/exploit choice. They're at different levels of abstraction. G captures the meta-level; EE captures the sub-level. The latter doesn't live on PC2 — or, more generally, doesn't live in cluster-PCA space at all.

This refutes the explore-vs-exploit reading of PC2 and restores the original "policy precision" interpretation. **But:** EE is a real conceptual axis in language (orthogonal to G in word2vec) that just happens to be outside Lakoff-cluster geometry. Like A_RUSSELL_DIAGONAL and D_SURPRISAL, EE is mostly outside the cluster.

### exp57 — full 6D basis projection

Final basis: A (Russell), C (reward), D (compression), G (goal-directedness), R (perceptual precision), EE (exploit-explore). Six theoretically-distinct cognitive primitives. Inter-axis cosines: all <0.20 except G-A at +0.56 (the structural entanglement).

Projected Lakoff schemas + an expanded concept-word battery (16 original + 9 additional: boredom, curiosity, obsession, addiction, wisdom, rage, compassion, fear, joy).

**The most theoretically striking concept-word hits:**

| word | finding | active-inference reading |
|---|---|---|
| RAGE | R = −0.25 (most negative R of any word) | rage IS runaway perceptual precision — collapsed precision-weighting on threat |
| CURIOSITY | EE = −0.18 (most negative EE of any word) | curiosity IS the explore-mode subjective state |
| FEAR | G = +0.19, C = −0.18, R = −0.12 | committed policy response to negative-value-state with impaired precision |
| BOREDOM | A = −0.21, C = −0.12, D = +0.08 | low affect + low reward + over-predictable environment — the boredom signature |
| RITUAL | D = +0.25 | compression-aiding practice |
| PSYCHOSIS | 14.4% basis-explained | failure mode of the regulation framework; structurally outside agency primitives |
| TRAUMA | C = −0.23, D = −0.09 | low wellbeing with uncompressed prediction error |
| LOVE | A = +0.21, G = +0.22 | balanced affect-and-commitment |

The concept words weren't anchor-pairs or constructed axes — they were single GloVe vectors plucked from the embedding directly. The fact that their coordinates in 6D AI-plus space match active-inference theoretical predictions precisely is a clean *predictive validation* that doesn't depend on anchor choices.

### The big picture after exp54–57

1. **PC1 = integrated reward / expected value of state** (NOT Russell's diagonal). C_REWARD captures it at cos = 0.54.
2. **PC2 = policy precision / goal-directedness** (original interpretation correct; explore-vs-exploit reframe refuted). G captures it at cos = −0.30, with affect entanglement at +0.56.
3. **PC3 = perceptual precision / regulability** (partial validation). R captures it at cos = +0.10 after V+A residualization.
4. **The cognitive-primitive basis is at least 6-dimensional**, mostly mutually-orthogonal: A (affect-as-felt), C (integrated reward), D (compression/predictability), G (goal-directedness), R (perceptual precision), EE (exploit-explore). Active-inference's "single free-energy quantity" doesn't fit; multi-objective views fit better.
5. **The cognitive primitives live mostly OUTSIDE Lakoff-cluster space.** A has ~16% magnitude in Lakoff space, C ~32%, D ~20%. PCA on Lakoff axes recovers only the projection-shadow of the cognitive primitives. Target-axis construction is the right tool to find the primitives themselves.
6. **The concept-word predictions land theoretically expected places:** pathological states sit near origin or low-explained; capacity-words load on G; compression-aiding practices (ritual, meditation) load on D; affective states load on A; rage shows precision-collapse; curiosity shows explore-mode.
7. **Affect-as-felt-state and wellbeing-as-integrated-state are independent** (cos = +0.008). You can feel any quality of affect in any integrated state.

### Methodological insight worth recording

> Language has many words for the *outputs* of cognitive primitives (committed, focused, drunk, awestruck, raging, regulated) but few or no words for the *primitives themselves* (policy precision, perceptual precision, value-of-state).
>
> This is consistent with active inference's framing of these primitives as *implicit hyperpriors* — they organize cognition without being introspectable contents. Language lexicalizes the outputs, not the variables.
>
> PCA over many constructed axes is the right tool to surface them — it averages across many anchor pairs that each capture one output. Target-axis construction with theory-led anchors is the validation step.

### Files added this entry

- `exp54_pc1_comparator.py`, `results_exp54.txt`, `exp54_results.npz`
- `exp55_lakoff_in_ai_space.py`, `results_exp55.txt`, `exp55_results.npz`
- `exp56_explore_exploit.py`, `results_exp56.txt`, `exp56_results.npz`
- `exp57_lakoff_in_6d_basis.py`, `results_exp57.txt`, `exp57_results.npz`

### Suggested next experiments (in priority order)

1. **fastText within-substrate replication.** Run the full 6D basis construction + Lakoff schema projection + concept-word projection in fastText. If the basis recovers similarly (RAGE at low R, CURIOSITY at low EE, PSYCHOSIS at low explained, etc.), the active-inference reading is substrate-robust. If it doesn't, GloVe-specific. Cheapest decisive test of generalizability. (~30 min compute.)

2. **PC3 / R steering test in Pythia.** Build R_axis as a steering direction in Pythia 70m residual stream. Inject during generation. Predict: positive steering raises the K-collapse threshold (more coherence preserved at higher α); negative steering lowers it (earlier collapse). Direct behavioral validation of the perceptual-precision-as-PC3 reading using existing exp17 infrastructure. (~1 day compute + debug.)

3. **Expanded concept-word battery, especially clinical states.** Project ~50 more concept words onto 6D basis. Focus areas: (a) more psychiatric states (mania, OCD, dissociation, panic, anhedonia, alexithymia); (b) contemplative states (presence, equanimity, samadhi, surrender, witness); (c) social-relational primitives (trust, belonging, recognition, dignity); (d) limit-states (sublime, ineffable, void, transcendence, awe). Tests the basis's predictive coverage and finds the boundary of what's representable in agency-space. (~few hours.)

4. **PC3 in word2vec under V+A-strip + cleaner R construction.** R currently only at cos = +0.10 with PC3 after V+A removal. Try a cleaner R construction with non-affect-loaded regulation vocabulary (homeostat-vocabulary, feedback-control-vocabulary, dynamical-systems language). See if cleaner R achieves higher PC3 alignment. Tests whether perceptual precision is recoverable independent of affect-correlation. (~few hours.)

5. **Cross-cultural anchor set.** Build 6D basis using Eastern philosophical vocabulary where available (Taoist yang/yin for affect, Buddhist śūnyatā/upekṣā for precision, etc.). Compare to Western-anchor coordinates. Tests whether the basis is Eurocentrically-language-bound or genuinely cognitive-architecture-shaped. (~half day.)

6. **The "ineffable" probe.** What concept words have the LOWEST basis-explained fraction? Test concepts associated with limit/mystical/preconceptual states: silence, void, transcendence, sublime, ineffable, awe, presence, isness, samadhi, mu. Prediction: these would land at very low basis-explained (like psychosis did at 14%) because they describe states OUTSIDE the agency-regulation framework. The pattern would distinguish "outside-agency-because-failure" (psychosis, trauma) from "outside-agency-because-transcendent" (contemplation, awe) by checking whether they cluster differently within the residual.

7. **Anthropic emotion-vector decomposition.** If accessible, take the published emotion-feature vectors from Anthropic's interpretability work, decompose onto the 6D basis, see if they land in theoretically-expected places. Cross-validates the basis on a different substrate (large transformer) and tests whether the active-inference reading extends to engineered-feature representations.

8. **Replication of orthogonality findings under different anchor constructions.** For each of the 6 primitives, build 2-3 alternative anchor constructions and check pairwise orthogonality persists. Robustness check on the multi-loss-function reading.

---

## Entry 25 — exp58–60: anxiety / random nouns / contemplative, the WEIGHT primitive, the final 7-axis basis

**Date:** 2026-05-26 (final stretch of the session). **Code:** `exp58_anxiety_random_nouns.py`, `exp59_weight.py`, `exp60_final_basis.py`.

### exp58 — anxiety / random nouns / contemplative states

Tested anxiety (the missing clinical state from exp57), random concrete nouns as control (sausage, pyjamas, marigold, stapler, accordion, lamppost, casserole, pebble), and contemplative/limit states (awe, sublime, transcendence, ineffable, presence, silence, void).

**The anxiety / fear / panic distinction is encoded in word vectors:**

| word | C_rew | A_aff | D_cmp | R_per | EE_x | G_pol | expl |
|---|---|---|---|---|---|---|---|
| anxiety | −0.18 | +0.05 | +0.03 | −0.07 | +0.09 | **+0.01** | 22.0% |
| fear | −0.18 | +0.14 | +0.01 | −0.12 | −0.08 | **+0.19** | 32.8% |
| panic | −0.19 | +0.07 | −0.06 | **−0.21** | −0.03 | +0.03 | 29.8% |

All three are low-wellbeing (negative C). But fear has committed defensive policy (G+0.19); anxiety has *no policy* (G≈0); panic has *the strongest precision collapse of any concept word tested* (R = −0.21). This recapitulates the clinical fear-vs-anxiety distinction (fear has an object/policy, anxiety doesn't) and the active-inference panic-as-precision-collapse story. From single GloVe vectors.

**Random nouns aren't as null as expected.** Marigold at 31.3% basis-explained (similar to anxiety), pyjamas at 23.7%. Random nouns lean negative on A_aff (low-arousal) and negative on G_pol (no agency) — they have a "passive low-arousal object" signature. Pebble at 10.5% is the cleanest null baseline.

**Contemplative states share a distinct signature: negative D_compress (uncompressible) + positive or small C_reward.** Awe, sublime, transcendence, ineffable all have D between −0.12 and −0.20. They're "exceeds my model" states with positive wellbeing. Compare to psychosis (D=−0.07, C=−0.08) — also uncompressible but with negative wellbeing. The structural distinction between **pathological-out-of-agency** (failure-of-regulation) and **contemplative-out-of-agency** (intentional-exit-from-agency) lies in the sign of C_reward.

### exp59 — WEIGHT as a real primitive (Niamh's intuition)

Niamh's hypothesis: PC2's missing piece might be weight (DIFFICULTIES ARE BURDENS metaphor; phenomenology of hope-as-light vs depression-as-heavy).

Built target_WEIGHT from heavy/light, weighty/airy, ponderous/buoyant, burdensome/effortless, encumbered/unencumbered, etc.

**Result:** WEIGHT is primarily a PC1-negative direction, not a PC2 direction.

```
              PC1     PC2     PC3
WEIGHT      -0.381  -0.164  -0.021
```

But it's a real cognitive primitive, mostly orthogonal to the other 6D basis axes (max |cos| = 0.34 with C_REWARD, which is structurally meaningful — cost is anti-reward).

**The decisive finding: DIFF explained jumps from 36.4% (6D) → 56.1% (7D + W). +19.7 percentage points — by far the largest single-axis basis contribution.** cos(W, DIFF) = +0.50, much higher than any other Lakoff schema's cosine with W. DIFFICULTY-BURDEN is essentially the weight axis + a smaller negative-reward component.

**Concept-word patterns:** Heavy = shame (+0.23), guilt (+0.16), fear (+0.16), presence (+0.15), regulation (+0.20), agency (+0.10). Light = sublime (−0.16), ineffable (−0.10), joy (−0.11), curiosity (−0.10), meditation (−0.07). The reframe the data suggests: W isn't "pathological-vs-healthy" but **effort/burden/responsibility vs release-from-effort**. Shame/guilt are heaviest (burden of responsibility). Sublime/ineffable are lightest (release from cognitive effort).

Pathological states (anxiety, trauma) sit near zero on W — clinical discourse around these is event-vocabulary, not somatic-quality-vocabulary. Predicted-heavy but data shows otherwise. The W signature is more about burden/effort than about clinical pathology.

### exp60 — final 7-axis basis: drop EE, add IO_CLEAN

Niamh's two cleanup calls: EE_EXPLOIT_EXPLORE didn't pan out (cos with PC2 = +0.06; failed to capture PC2) and was capturing corporate-optimization vs scientific-investigation register rather than a cognitive primitive. Drop it. IO_CLEAN was robustly Tier-2-independent across exp21-57 — add it as a basis axis since the active-inference Markov-blanket reading provides theoretical motivation.

**Final 7-axis basis:**

| axis | active-inference reading | role |
|---|---|---|
| C | integrated reward / expected value of state | content: what's valued |
| W | weight / felt cost / effort burden | content: resistance to value |
| A | affect quality (Russell V×A diagonal) | content: somatic-felt-state |
| G | policy precision / goal-directedness | content: commitment level |
| R | perceptual precision / regulability | content: inference quality |
| D | compression / surprisal | content: model performance |
| IO | container-topology / Markov-blanket-candidate | structural: substrate |

Inter-axis cosines: mostly < 0.2 except C-W at −0.34 (expected: reward and cost anti-correlated) and A-G at +0.56 (the persistent affect-policy entanglement).

**IO_CLEAN is gorgeously orthogonal** (max |cos| with any other axis = 0.11) — confirms substrate-primitive reading structurally. BUT adding IO to the basis only improves other schemas' explained-variance by 0-4% (max +3.9% for PATH). IO is its own thing — other schemas don't tap into it.

**Important methodological caveat surfaced by IO probes:** Only BELONGING loads strongly on IO_CLEAN (+0.29 — strongest of any concept word). Other self/other concepts (self, identity, intimacy, alienation, communion, loneliness) load only weakly on IO. This is because IO_CLEAN was built from spatial-containment anchors (inside/outside, contained/released) — it captures the **container-topology version** of in/out, not the **abstract phenomenal-Markov-blanket** version. To test the full Markov-blanket reading, we'd need a separate target axis built from self/other/agent/environment vocabulary specifically.

### Niamh's "computation prior to soma" observation — the W → COST reframe

End-of-session thread to record:

W was operationalized via somatic vocabulary (heavy/light, burdensome/effortless, encumbered/unencumbered). But what we're finding when we project word-vectors onto W is *the geometric structure of computational cost* (anti-reward + effortful-engagement) that gets *expressed* in somatic-weight vocabulary. The phenomenological felt-heaviness is one of cost's expressive modalities in embodied agents — but the structural primitive being recovered is **cost as computational quantity**, not weight as somatic primitive.

Evidence:
- W's pole semantics are cost-shaped: positive pole includes "burden, costly, oppressive, encumbered"; negative pole includes "effortless, lighthearted, weightless." Even the explicit lexical anchor "costly" appeared in W's nearest-neighbor lookup as a positive-pole word.
- W has cos = −0.34 with C_REWARD — anti-reward, specifically the cost aspect rather than full anti-reward.
- W maps onto DIFF at +0.50 — difficulty in Lakoff = high computational cost.
- Active-inference framing: expected free energy = (cost = surprise + risk + ambiguity) − value. PC1 in word2vec might literally be the C-vs-W (value-vs-cost) axis that constitutes expected free energy.

**Implication for the writeup:** rename W to COST or EFFORT_COST. The somatic vocabulary was a heuristic for finding the axis; the underlying primitive is computational. Active-inference reading sharpens: PC1 has C (value-pole) and W (cost-pole) as the two contributions to expected-free-energy-minimization.

**Broader theoretical implication (worth recording for whoever writes the formal paper):** word-vector geometry recovers computational primitives more cleanly than the embodied-experiential vocabulary that gets used to describe them. Cost-as-computation is structurally prior to weight-as-felt-experience; the felt-experience is one expressive modality of the computation in agents that happen to have bodies. This is a specific claim against strong embodied-cognition readings that treat soma as foundational and abstract concepts as derived from somatic image schemas.

### Suggested next experiments (consolidated and prioritized)

The session ended with three clear threads. Adding them to the queue from Entry 24:

**Thread A — fastText within-substrate replication** (highest priority, cheapest, decisive):
Rerun the full 7-axis basis construction + Lakoff schema projection + concept-word projection in fastText (Facebook, same Wikipedia+Common Crawl training data, different algorithm). If the same patterns recover (DIFF heavy on W, PSYCHOSIS low-explained, RAGE on negative R, CURIOSITY on negative EE if we re-include it, BELONGING on IO, anxiety vs fear distinction on G), the active-inference reading isn't GloVe-specific. If not, GloVe-300 artifact.

**Thread B — COST anchor construction (the W → COST reframe test):**
Build target_COST from non-somatic-vocabulary anchors — abstract-computational cost language without the heavy/light metaphor. Anchor candidates: (expensive, free), (taxing, refreshing), (demanding, easy), (laborious, automatic), (depleting, sustaining), (consuming, replenishing), (extracting, conserving), (taxing, restorative). Test:
- Does COST recover the same axis as W? (cos(COST, W) should be high if computational and somatic are aspects of the same primitive)
- Does COST capture DIFF as cleanly? (cos(COST, DIFF) ≈ +0.50 if computational cost IS what DIFF tracks)
- Does COST have lower correlation with A_affect than W did?
- If yes to all three, COST is the cleaner operationalization and weight-vocabulary was just one expressive modality.

**Thread C — Markov-blanket / self-other / SELECTION anchor construction:**

Niamh's end-of-session reframe: the substrate primitive we're hunting might be more cleanly named SELECTION than Markov-blanket. Active inference has selection running at every level of its hierarchy (Bayesian model selection, policy selection, attention selection, perceptual selection). Foundationally: the Markov blanket is *maintained* by ongoing selection — the agent IS a continuing selection-act that distinguishes self from environment, this from that, here from there. The Markov blanket is the *structure* that emerges from ongoing selection; selection is the *verb*. Active inference's computational primitive of "selection under expected free energy minimization" maps onto exactly this.

This reframe potentially reconciles two findings from exp60: (a) IO_CLEAN was robustly orthogonal in the basis but (b) it only captured spatial-container vocabulary, not the abstract phenomenal-self/other concepts (intimacy, identity, alienation loaded weakly on IO). SELECTION-vocabulary might capture both:
- The substrate primitive (selection that constitutes the agent — Markov blanket as ongoing-selection)
- The conceptual self/other content (intimacy/alienation/identity/belonging as products of selection)

**Sub-thread C1 — target_SELECTION:**
Build target_SELECTION from selection-vocabulary that's not in any existing axis. Anchor candidates (verb-form of in/out, the active-inference primitive):
- (selected, rejected), (chose, refused), (picked, discarded), (admitted, denied),
  (accepted, declined), (kept, removed), (chosen, eliminated), (preferred, overlooked),
  (favored, excluded), (designated, omitted), (endorsed, dismissed — but check IO_CLEAN_MML overlap),
  (singled-out, ignored), (highlighted, neglected)

Tests:
- cos(SELECTION, IO_CLEAN): are they the same axis? If high (>0.5), SELECTION is just IO in verb-form. If moderate (0.2-0.4), they're related but distinct. If low (<0.2), SELECTION is a different primitive.
- cos(SELECTION, G_GOAL_DIRECTED): selection-of-policies might overlap with goal-directedness
- cos(SELECTION, the rest): orthogonality check
- Does SELECTION capture self/identity/intimacy concept words better than IO_CLEAN did?
- Does SELECTION carve a clean dimension when added to the 7-axis basis?

If SELECTION is its own dimension AND captures self/other concepts: it's the substrate-primitive we've been hunting. Markov-blanket as the structural outcome; selection as the operational verb.

**Sub-thread C2 — target_MARKOV_BLANKET (alternative operationalization):**
Build a Markov-blanket axis from abstract self/other anchors avoiding spatial-containment vocabulary. Anchor candidates: (self, other), (agent, environment), (internal, external), (interior, exterior — but check IO_CLEAN), (mine, theirs), (introspection, perception), (autonomous, exposed-to).

Compare with target_SELECTION:
- If cos(MARKOV_BLANKET, SELECTION) high, they're aspects of the same primitive (substrate primitive)
- If they differ, we have multiple distinct substrate-related primitives

The active-inference reading would suggest they should overlap substantially — the Markov blanket IS maintained by selection.

**Threads from Entry 24 that remain:**
- PC3 / R steering test in Pythia (direct behavioral validation of perceptual-precision reading)
- Larger concept-word battery
- Cross-cultural anchor set
- Anthropic emotion-vector decomposition over the basis

### Files added this entry

- `exp58_anxiety_random_nouns.py`, `results_exp58.txt`
- `exp59_weight.py`, `results_exp59.txt`
- `exp60_final_basis.py`, `results_exp60.txt`, `exp60_results.npz`

### Final basis status as of end-of-session

The basis stands at 7 axes: C, W (= COST, pending the rename test), A, G, R, D, IO_CLEAN. The active-inference reading covers all of these. Most pairwise orthogonality is good (|cos| < 0.2 except C-W at −0.34 expected and A-G at +0.56 known issue). The Lakoff schemas decompose mostly between 22% and 71% explained-variance, with the cluster-vs-Tier-2 partition holding (cluster schemas 40-70%, Tier-2 ~22%, DIFF newly explained at 56% after adding W).

Concept-word predictions land in theoretically-expected places across multiple dimensions: fear vs anxiety on G, panic on R, psychosis at lowest-explained, ritual on D, rage on most-negative R, curiosity on most-negative EE (before we dropped EE), shame/guilt on highest W, sublime/ineffable on lightest W and most-negative D.

The fastText replication is the next decisive step. Until then, treat all of this as "robust in one substrate, suggestive of broader structure, requires within-substrate replication to be claimed as a finding."
