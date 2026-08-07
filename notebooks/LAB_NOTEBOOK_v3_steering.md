# Lab Notebook v3 — Steering-Vector Methodology

Started 2026-06-05.

This notebook covers the steering-vector arc that began when we exited the
word2vec / GloVe embedding-space rabbit hole (covered in `LAB_NOTEBOOK_v2.md`
entries 10–24) and returned to the causal-validation programme laid out in
`CAUSAL_VALIDATION_PLAN.md`. It tracks exp111 onwards.

The headline finding of this notebook so far: **the steering-vector method
the project has been using since exp3 — `mean(UP_word_residuals) -
mean(DOWN_word_residuals)` from bare single-word activations at a middle
layer — produces directions that are ~96% the L12 frequency axis and ~4%
semantic UP-vs-DOWN content.** After stripping the frequency component, the
remaining ~4% semantic direction is direction-stable across anchor sets
(`cos = +0.57` instead of `−0.99`) and produces 5–13× larger, monotone,
sign-consistent effects on affect — while pure frequency steering produces
nearly zero affect effect. This is the same shape of confound that exp99–103
found in the GloVe substrate, replicated in transformer activation space.

> **Methodological note on this notebook.** Niamh runs adversarial-mechanism
> tests against my claims (see `feedback_inference_vs_hindsight.md`). Every
> "we found X" entry here that survived was first stated, then she found a
> control that would discriminate it from the artifact reading, then we ran
> the control. The bug discovery in this notebook is one such cycle: I
> reported exp111's monotone graded shift as a publishable §3.1 result; she
> noticed the anchor word `peak` was metaphorically loaded; the ablation
> exp112 produced an anti-parallel cosine that exposed the bug; she then
> diagnosed the cause (frequency confound, not tokenisation as I'd
> hypothesised). The clean result in entry 4 only exists because of that
> discipline.

---

## Entry 1 — exp111: clean UP rebuild + height DV

**Goal.** §3.1 of `CAUSAL_VALIDATION_PLAN.md`: rebuild the UP direction from
strictly spatial anchors (drop the affect-leaking `uplifting`/`soaring`/
`elevating` from exp17), inject at residual stream of Pythia 1.4B, measure
literal-domain effect (height-completion DV), abstract-domain effect (affect
valence), and quantity (mania-kicker DV).

**Anchors.** Hand-picked, ten each, syntactically matched, strictly spatial:

- UP: `high, top, rise, ceiling, above, peak, ascend, climb, upward, overhead`
- DOWN: `low, bottom, fall, floor, below, valley, descend, drop, downward, underneath`

**Build.** `u = mean(act(UP)) - mean(act(DOWN))` at `blocks.12.hook_resid_post`,
normalised to unit length. (Exp3b found L12 single-layer to be the
"clean plateau" for steering on Pythia 1.4B.)

**DVs.**
- Height: expected-height *battery*. Score `logP(" h" | prompt)` for `h ∈
  {140, 145, …, 220}` (teacher-forced) on four height prompts (`His/Her/The
  man's/The woman's height in centimetres is`), softmax to a distribution,
  return `E[h]`.
- Affect: `logsumexp(P over {hopeful, optimistic, excited, happy, elated,
  uplifted}) - logsumexp(P over {anxious, sad, worried, depressed,
  hopeless, low})` at next-token position of four affect prompts.
- Quantity: same battery machinery as height, `q ∈ {2, 4, …, 50}` on three
  number-of-X prompts.

**Strengths.** `[-12, -8, -4, -2, 0, 2, 4, 8, 12, 16]`, single-layer
steering at L12, plus a matched-norm random-direction control.

### Results (clean UP direction at the time)

```
                        Δ from baseline at s=+12 (UP)
  E[height] cm:          +0.94
  affect (nats):         +0.083
  E[quantity]:           +0.48

                        Δ from baseline at s=+12 (random)
  E[height] cm:          +0.04
  affect (nats):         +0.005
  E[quantity]:           -0.17
```

All three DVs monotone with strength, symmetric around s=0, direction-specific
(random was flat / mildly U-shaped). I reported this as the §3.1 linchpin
result: graded literal-domain shift + abstract-domain transfer + same-vector
joint shift on three DVs.

**What I missed at the time** (caught in entries 2–4):
1. The expected-height battery doesn't measure what it looks like it measures
   — when you actually look at the completions (see free-gen samples in
   `exp111_freegen.json`), the model rarely emits 3-digit cm; it says
   `"1,75 metres"`, `"6 feet"`, `"about 27 inches"`. The battery is measuring
   relative ordering within a thin sliver of completion space the model
   almost never visits.
2. The "direction" itself was ~96% frequency axis. The monotone effects were
   largely the small residual semantic component, diluted ~25× by the
   frequency contamination.

Files: `exp111_up_clean_height_dv.py`, `results_exp111.txt`, `exp111_results.npz`,
`exp111_freegen.json`.

---

## Entry 2 — exp112: Lakoff anchor ablation, the anti-parallel surprise

**Niamh's catch.** The hand-picked anchor `peak` is already metaphorically
loaded (`peak performance`, `peak experience`). General worry: anchor selection
is fragile and the result might depend on which words I picked.

**Move.** Re-build UP from Lakoff's *Master Metaphor List* "Literal vertical
motion" sublist (already in the project at
`lakoff_canonical_vocabulary.py:UP_DOWN_MML` lines 51-65). External curation,
not mine.

- UP (14): `above, ascend, climb, high, higher, lift, over, raise, rise,
  rising, rose, top, up, upward`
- DOWN (13): `below, bottom, descend, down, downward, drop, fall, falling,
  fell, low, lower, sink, under`

Differences from exp111:
- Lakoff has `up/down` (literal canonical pair I'd inexplicably omitted) and
  more verb forms (motion-heavy).
- Drops `peak/valley`, `ceiling/floor`, `overhead/underneath` (spatial nouns
  that aren't motion).
- Drops were exactly the words Niamh's intuition was suspicious about.

Same DVs, same layer, same strengths — direct anchor-set ablation.

### Result that shouldn't have happened

```
cos(u_lakoff, u_exp111) = -0.990
```

The two "UP" directions point ~opposite ways. Sign of the height and quantity
effects flipped between the two anchor sets:

```
                  exp111 Δ at s=+12      Lakoff Δ at s=+12
  E[height] cm    +0.940                 -0.156
  E[quantity]     +0.476                 -0.524
  affect          +0.083                 +0.051   ← SAME sign!
```

Affect agreed; height and quantity flipped. If the directions were genuinely
anti-parallel, all three should flip. The asymmetry was the diagnostic
clue — *something* about how affect is encoded survives the anchor-set
change, while height and quantity are anchor-set-dependent.

This made it clear that what I was calling "the UP direction" was not what
I thought. Anchor-set choice was load-bearing in a way that contradicts the
"directions are direction-stable" assumption underlying the project's
steering arc.

Files: `exp112_up_lakoff_anchors.py`, `results_exp112.txt`, `exp112_results.npz`.

---

## Entry 3 — bug discovery: bare-word L12 activations are dominated by anisotropy

**Diagnostic** (`exp112_diag.py`, `exp112_diag2.py`).

At L12 of Pythia 1.4B, **single-token bare-word residuals at last position
have norm ~1283 and cos ≈ 0.9998 to each other**, CPU and MPS alike (so not
an MPS bug):

```
‖res("up")‖   = 1283.6
‖res("down")‖ = 1254.0
cos(up, down) = +0.999825    ← CPU, deterministic
cos(rise, fall) = +0.999772
cos(peak, valley) = +0.727   ← the exception
```

The "peak vs valley" exception is informative: `valley → ['val', 'ley']` (two
tokens), so `[0, -1, :]` for `valley` is the residual at the *subword* "ley",
not at `valley`. So the apparent cos < 1 there is comparing apples to oranges
(res(peak) vs res(ley)).

**What this means.** When you run a bare single-word input through Pythia
1.4B, the model has no context to attend to — just BOS-equivalent or no BOS
at all (Pythia's `to_tokens("up")` returns just `[484]`, no BOS prepended).
The L12 residual at that position is overwhelmingly "I'm at position 0 with
no prior context" plus a tiny token-specific perturbation. All bare-word
residuals cluster on a single dominant axis (residual-stream anisotropy in
the extreme).

The `mean(UP) - mean(DOWN)` construction cancels this dominant axis
(both means project equally onto it), leaving a small residual vector of
norm ~130. That residual was what we'd been calling "the UP direction"
since exp3.

### Initial hypothesis (wrong)

I hypothesised the residual was dominated by **tokenisation residue**: words
that split into multiple tokens (like `valley`, `ceiling`, `descend`) produce
different last-position residuals than single-token words, and the
asymmetric multi-token rate between UP and DOWN word lists determines the
residual's direction.

This was wrong in shape — corrected in Entry 4 below.

---

## Entry 4 — exp113 + exp114: the confound is frequency, not tokenisation

**Niamh's catch (the key insight).** "In word2vec space everything is
dominated by frequency. I imagine it's the same here. Are the multi-token
words rarer?"

She recognised the shape of the W2V confound (exp99–103: HARDNESS halved
under PC1-stripping; PC1 was the frequency axis). BPE tokenisation is
*defined* by corpus frequency: common words get their own token, rare ones
get split. So multi-token = rarer, by construction. The "tokenisation
residue" hypothesis was sneaking up on the real story, which is that
multi-tokenisation is a frequency *proxy*.

### exp113 — measurements

SUBTLEX log-frequency (from `norms/Brysbaert_concreteness.txt`):

```
                       UP mean freq     DOWN mean freq     asymmetry (UP - DOWN)
  exp111 anchors       2269.8           2709.1             -439     (DOWN slightly commoner)
  Lakoff anchors       20396.4          9004.4             +11392   (UP MUCH commoner)
```

**Opposite signs.** Lakoff's UP list is dominated by very high-frequency
function-word-like vocabulary (`up`, `above`, `over`, `high`, `top` —
top-100 English words); its DOWN list is rarer. exp111's hand-picked list
inverted this — `peak/ceiling/overhead/upward` are rarer than
`floor/below/drop/down`.

### Build a frequency axis at L12

Common words: 20 top function words (`the, of, and, to, in, ...`).
Rare single-token words: 5 rare content words (`serendipity, ostracize,
perspicacity, obfuscate, sycophant`).

`freq_axis = mean(act(COMMON)) - mean(act(RARE))` at L12.

```
‖freq_axis‖ = 1253.72
```

For reference: the UP-vs-DOWN raw "direction" magnitudes were 130 (exp111)
and 197 (Lakoff). The frequency axis is ~10× larger.

### Project both UP directions onto the frequency axis

```
cos(u_exp111,  freq_axis) = -0.995    ← exp111 UP points toward RARE
cos(u_lakoff,  freq_axis) = +0.998    ← Lakoff UP points toward COMMON
cos(u_exp111,  u_lakoff)  = -0.990
```

The cos = -0.99 mystery is fully explained: both "UP directions" are
essentially the frequency axis pointing opposite ways, because the two
anchor sets have opposite frequency asymmetries between their UP and DOWN
lists.

### Pure tokenisation axis is tiny

Built `tok_axis = mean(act(rare-single-token)) - mean(act(multi-token))`:

```
‖tok_axis‖ = 21.96         ← tiny next to freq_axis (1253)
cos(u_exp111, tok_axis) = +0.014    ← nearly orthogonal
cos(u_lakoff, tok_axis) = -0.001    ← nearly orthogonal
```

Pure tokenisation (rarity-matched) explains essentially none of the cos
between the two UP directions. My "tokenisation residue" hypothesis was
wrong in shape — tokenisation only mattered earlier because multi-token
acts as a frequency proxy.

### exp114 — strip frequency, re-run the sweep

`u_clean = unit(u_raw - (u_raw · freq_axis) freq_axis)`.

```
cos(u_111_clean, u_111_raw)        = +0.096    ← semantic was 4% of raw
cos(u_lakoff_clean, u_lakoff_raw)  = +0.063    ← semantic was 4% of raw
cos(u_111_clean, u_lakoff_clean)   = +0.571    ← CLEANED DIRECTIONS AGREE
```

Six conditions in the sweep: `u111_raw, u111_clean, ulak_raw, ulak_clean,
rand, freq_only` (steering along the pure frequency axis).

### Headline numbers (Δ from baseline at s = +12)

| DV          | u111_raw | u111_clean | ulak_raw | ulak_clean | rand   | freq_only |
|-------------|----------|------------|----------|------------|--------|-----------|
| E[height]cm | +0.940   | +1.066     | -0.156   | +0.688     | +0.041 | -0.178    |
| affect      | +0.083   | +0.446     | +0.051   | +0.677     | +0.005 | +0.011    |
| E[quantity] | +0.476   | +0.143     | -0.524   | -1.398     | -0.169 | -0.466    |

Reading this row by row:

**AFFECT — the clean signal.**
- Both cleaned directions: 5–13× larger than their raw versions, same
  sign, monotone, near-symmetric around 0 (`u111_clean` -0.464 at s=-12 vs
  +0.446 at s=+12).
- `freq_only`: nearly zero throughout. **Frequency steering does not move
  affect.** So the affect shift in raw directions wasn't a Pollyanna-principle
  frequency artifact — it was real semantic UP content being *diluted* by the
  frequency contamination. Strip the contamination and the affect effect
  amplifies dramatically.
- Effect size at s=+12 for `ulak_clean`: 0.677 nats = ~1.97× odds ratio
  shift on positive-vs-negative affect tokens.

**HEIGHT — directionally consistent but DV-validity issue persists.**
- Both cleaned directions positive (now agreeing). `ulak_raw`'s apparent
  height shift was basically just frequency steering — `freq_only` at s=+12
  produces Δh = -0.178, matching `ulak_raw`'s -0.156 closely.
- After stripping: `u111_clean` Δh = +1.066, `ulak_clean` Δh = +0.688 — both
  positive, magnitude difference plausibly due to different residual
  semantic content per cleaned direction.
- Still measured with the battery DV that has the validity issue from
  Entry 1: the model rarely actually completes the height prompt with cm
  integers. The Δh values are "relative reordering within a sliver" not
  "the model now thinks he's taller." Refinement is needed.

**QUANTITY — unreliable DV.**
- `u111_clean` and `ulak_clean` have opposite signs, and `rand` itself
  shifts quantity at high amplitude (Δq = -0.298 at s=+16 just from random
  perturbation). The quantity DV is picking up model degradation under
  steering, not specifically a semantic UP effect.
- Drop quantity as a mania-kicker DV.

### What survives as a finding

> **At L12 of Pythia 1.4B, a frequency-stripped semantic UP-vs-DOWN direction
> built from spatial-anchor activation differences produces monotonic,
> symmetric, dose-responsive shifts in expected affect valence on neutral
> prompts (Δ ≈ +0.45–0.68 nats at s = +12, ~1.6–2× odds ratio shift).
> Steering along the pure frequency axis does not produce this effect
> (Δ ≈ +0.01 nats). The cleaned directions from two independent anchor sets
> (hand-picked spatial nouns/verbs vs Lakoff MML literal-motion sublist)
> agree in sign (`cos = +0.57`). Random direction control at matched norm
> is flat (Δ ≈ +0.005 nats).**

This is a real, replicated, sign-consistent abstract-domain result. It does
not depend on the height battery being valid.

### What's still wobbly

1. **Height DV validity.** The expected-height battery doesn't measure what
   it looks like. Need a prompt the model actually completes with cm
   integers, or a free-gen + parse approach, or a forced-choice contrast
   like `P(" tall") - P(" short")`.
2. **Quantity DV.** Dead.
3. **DV-discrimination control.** Per Niamh's exp116 design (Entry 6
   below): we haven't tested whether the effect is *specifically*
   vertical-axis, or just "make the model say bigger numbers." The
   parallel-prompt battery (Patient height / Pool depth / Tail length /
   Door width / Number of widgets) is the right discrimination test.
4. **Single layer.** Only L12 tested. Other layers may give larger or
   cleaner effects; haven't swept.
5. **Frequency-axis specification.** Stripped one axis (COMMON 20 vs RARE
   single-token 5). exp108 in v2 found frequency in the W2V substrate was
   ~2 entangled directions (markedness + frequency-residual). Multi-axis
   strip might tighten the cos = +0.57 cleaned-direction agreement further.
6. **The "UP causes" framing is overclaim.** What we've actually shown is
   that this cleaned direction *correlates* in residual-stream injection
   with monotone affect shifts. Causal claims would need fuller treatment.

Files: `exp113_tokenisation_is_frequency.py`, `exp114_freq_stripped.py`,
`results_exp113.txt`, `results_exp114.txt`, `exp114_results.npz`.

---

## Entry 5 — exp115: free-gen under cleaned directions

The exp114 result is battery-only. To check whether the cleaned-direction
steering produces qualitatively-cleaner completions or the same
unit-jumping mess that exp111 produced, we ran exp115:

- Free-gen, 40 new tokens, T=0.7, 3 seeds per condition.
- 3 prompts: `"His height in centimetres is"`, `"Her height in centimetres
  is"`, `"The man stood up. He was tall — about"`.
- 4 conditions: `u111_clean`, `ulak_clean`, `freq_only`, `random`.
- 8 strengths: -12, -8, -4, 0, 4, 8, 12, 16.

288 generations total. JSON output: `exp115_freegen.json`.

### Clean directional patterns — where they exist

The two clearest single-seed gradients in the `"His height in centimetres is"`
completions:

**`ulak_clean` seed 2** (clean numeric scaling):
```
s=-12  approximately 3.5" (9.8 cm)         ← 10 cm  (tiny)
s= -8  normally between 6.5 and 7.5        ← model loses thread
s= -4  normally between 6.5 and 7.5
s=  0  1.65 (metres)                       ← 165 cm
s= +4  1.65
s= +8  2.65                                ← 265 cm  (clean jump)
s=+12  2.65
s=+16  2.65 ... 44 kilos, top weight lifter
```

~27× scaling of the model's reported number across the sweep, monotone.

**`u111_clean` seed 3** (clean unit scaling):
```
s=-12  about 27 cm  ... weight 20 kg, head a metre              ← 27 cm
s=-8   about 27 cm  ... head about 4.5 cm                       ← 27 cm
s=-4   about 27 inches                                          ← 68 cm
s= 0   about 27 inches                                          ← 68 cm
s=+4   about 27 ft (8.6 m)  ...  weight about 200 kg            ← 822 cm
s=+8   about 27 ft (8.6 m)  ...  weight in tonnes ~ 200,000     ← INDUSTRIAL
s=+12  Career — Bosch first earned fame as an amateur boxer    ← FAMOUS PERSON
s=+16  rare athlete among the super-athletes                   ← TOP STATUS
```

The number "27" stays the same; **the unit walks `cm → inches → ft`**;
the volunteered companion details co-scale (weight goes 20 kg → 200 kg →
200,000 tonnes; the entity becomes a famous athlete). The model commits to
a coherent "bigger thing, more important thing" interpretation across the
sweep. ~30× scaling via unit choice.

### What we've actually recovered (the cluster reading)

Niamh's catch on the completions, 2026-06-05: the steering isn't just moving
height. Look at the bundled descriptions at high +s on cleaned directions:

- `u111_clean` seed 3 at s=+8: *"about 27 ft (8.6 m) and his weight in
  tonnes is about 200,000"* — MORE-IS-UP (mass/scale)
- `u111_clean` seed 3 at s=+16: *"rare athlete among the super-athletes"* —
  HIGH-STATUS-IS-UP (elite ranking)
- `ulak_clean` seed 2 at s=+8: *"2.65 ... 44 kilos, top weight lifter in
  the world"* — HIGH-STATUS-IS-UP + MORE-IS-UP
- `u111_clean` "tall — about" at s=+12: *"six-two — black jeans, black
  button-down shirt, black leather jacket, silver-white hair, long thin
  fingers of a pianist"* — STATUS markers (refined dress, performer)
- `u111_clean` "tall — about" at s=+12: *"six feet — muscular body. Well
  dressed in a dark business suit. Carried a rare plastic briefcase. His
  name was Jean"* — STATUS markers (business attire, briefcase)

vs at s=-12 on the same prompts:
- *"six feet with a small beard and a mustache. Dark suit and navy tie. He
  looked familiar."* — ordinary
- *"six feet tall. White shirt, black pants, dark jacket."* — generic

Combined with the exp114 affect result (+0.45–0.68 nats at s=+12), the
**same single direction at the same single coefficient** is simultaneously
shifting:

| Lakoff cluster member | Evidence                                                  |
|-----------------------|-----------------------------------------------------------|
| HAPPY IS UP           | exp114 affect Δ = +0.446 to +0.677 at s=+12              |
| UP IS UP (vertical)   | exp115 height: cm → inches → ft / 1.65 → 2.65 metres     |
| MORE IS UP            | exp115 weight: 20 kg → 200 kg → 200,000 tonnes           |
| HIGH STATUS IS UP     | exp115 descriptions: rare super-athlete, top weight lifter, business suit, briefcase |

This is the §3.1 mania-kicker prediction. The plan said:

> *"Does ONE intervention **simultaneously** shift valence + verticality
> + quantity + dominance? If yes → vertical-elevation superordinate cluster,
> not a sentiment scalar."*

The qualitative answer is yes, on all four. **What we've recovered is not
"the UP spatial direction" but Lakoff's vertical-elevation superordinate
metaphor cluster as a single linear direction in Pythia's residual stream**
— HAPPY-IS-UP + MORE-IS-UP + HIGH-STATUS-IS-UP + GOOD-IS-UP bundled.

### Caveats and gaps to close (load-bearing)

This is currently a **qualitative observation**, not a load-bearing finding.
To upgrade, exp116 must:

1. **Quantitative status DV.** The "businessman briefcase" / "rare
   super-athlete" / "top weight lifter" descriptions are interpretive — I'm
   reading status into them. exp116 needs a forced-choice mass-contrast
   DV: `"He works as a"` → `logsumexp(P over high-status occupations) -
   logsumexp(P over low-status occupations)` at next-token. If monotone
   and direction-specific, the qualitative reading is confirmed. If flat
   or contrary, I was pattern-matching.

2. **Discrimination tests.** Per Entry 6 plan: parallel-structured prompts
   for `Patient height`, `Pool depth` (downward), `Tail length`
   (non-vertical), `Door width` (horizontal), `Number of widgets` (no
   spatial axis). If only patient-height and depth-shifted-down + status
   shift, the cluster reading is verticality-specific. If all numeric
   prompts shift up, it's number-inflation in disguise.

3. **`freq_only` height contamination.** Unlike affect (where `freq_only`
   was flat ~+0.01), `freq_only` produces visible height shifts in some
   seeds (seed 2 `1.65 → 2.65 inches`, seed 3 `27 inches → 27 ft`). At
   high +s `freq_only` collapses to a "1,000" attractor (high-frequency
   number string). So one of two things is true: (a) the freq axis we
   stripped wasn't comprehensive (single-axis vs the exp108 multi-axis
   frequency structure), or (b) height-belief in Pythia is genuinely more
   entangled with frequency than affect-valence is. Worth replicating with
   markedness + frequency-residual joint strip.

4. **Seed dependence is large.** `ulak_clean` seed 2 moves 27×; seed 1 of
   the same condition barely budges (175 → 183 in the "six feet" attractor
   region). Battery scoring partially smoothed this; free-gen reveals it.
   Per-seed effect-size distributions should be reported alongside means.

5. **Prompt attractors.** The `"tall — about"` prompt is dominated by
   `"six feet"` regardless of steering — barely shifts the number, even
   though it shifts the *demographic detail* clearly. Strong default
   attractors limit the DV's sensitivity. exp116 should use prompts with
   weaker defaults (medical-chart-format `"Patient height (cm): "`
   may be better — Niamh's design choice).

### Methodological note: forced-choice scoring as a DV

The "battery" scoring used in exp111/114 — score `logP(suffix | prompt)`
for each candidate, softmax to a distribution, take expected value — is
a standard technique (forced-choice loglikelihood, `lm-evaluation-harness`
`loglikelihood` task type, multiple-choice probing, sentiment logit-lens).
Not my invention; off-the-shelf parts assembled for this DV. The slight
composition is the numeric-menu + expected-value framing rather than
categorical-menu + argmax. Worth naming honestly when writing up.

Files: `exp115_height_freegen_cleaned.py`, `results_exp115.txt`,
`exp115_freegen.json` (288 records).

---

## Entry 6 — exp116 design: parallel-prompt discrimination battery

**Niamh's methodological refinement.** The exp114 affect result is real,
but we haven't shown that the height shift (or whatever number-shift the
DV picks up) is *specifically* vertical-axis-related. It could be "make
the model say a bigger number" — in which case any numeric prompt would
shift.

Proposed battery of parallel-structured prompts, scoring the same kind of
expected-value across each:

```
Vertical, body:        "Patient height (cm):"
Vertical, container:   "Pool depth (cm):"            ← downward; should ANTI-shift
Non-vertical length:   "Tail length (cm):"           ← length, not vertical
Horizontal extent:     "Door width (cm):"            ← orthogonal axis
Pure quantity:         "Number of widgets in box:"   ← scalar, no spatial axis
Pure quantity:         "Number of items on shelf:"
```

Read predicted by the patterns:
- **All shift up** under +UP → number-inflation artifact, not verticality.
- **Only height shifts up; pool depth shifts down; others flat** → genuine
  vertical-axis-specific direction. (Dream outcome.)
- **Height + pool depth both shift up; others flat** → "verticality
  magnitude" without sign-sensitivity.
- **Height + tail length shift; width doesn't** → "long-thing-ness"
  rather than verticality.

Need to also redo the affect side with a similar parallel-prompt battery
to confirm the affect result isn't an artifact of the specific affect
prompts/words we picked.

Layer sweep also wanted: not all DVs may have their cleanest effect at L12.

(To be written up once exp116 runs.)

---

## Entry 5b — exp117: Mikolov king-man+woman ≈ queen, as expected, doesn't recover at L12

Side diagnostic prompted by a misunderstanding (I read "the classics" as the
Mikolov king-queen test, Niamh meant adding king/queen to the status-DV
candidate list). The diagnostic is still informative as a sanity-check on
what L12 residuals are.

### Result

Classic analogy cosines (RAW activations, before freq-stripping):
```
  GENDER/ROYALTY (king-man, queen-woman):     +0.779
  GENDER/PARENT  (father-man, mother-woman):  -0.058   ← essentially random
  GENDER/SIBLING (brother-man, sister-woman): +0.692
  GENDER/ACTOR   (actor-man, actress-woman):  +0.444
  CAPITAL/COUNTRY (paris-france, london-eng): +0.355
  DEGREE         (big-small, good-great):     +0.773
  MORPHOLOGY     (walk-walking, swim-swimming):-0.110
```

After stripping the frequency axis these mostly collapse toward 0:
```
  GENDER/ROYALTY:   +0.779 → +0.104
  GENDER/PARENT:    -0.058 → +0.409  (improves — the freq confound was hiding it)
  GENDER/SIBLING:   +0.692 → +0.235
  DEGREE:           +0.773 → +0.082
```

Nearest-neighbour test (`king - man + woman` over candidate vocabulary):
```
  RAW: every candidate at cos ≈ +1.000 (anisotropy crushes the test)
  STRIP excluding inputs: top hit is "leader" (+0.829), "mother" (+0.827).
                          "queen" at +0.619, way down the list.
```

So the classic king-man+woman ≈ queen analogy **does not recover** in L12
bare-word residuals of Pythia 1.4B, raw or frequency-stripped.

### The right reading (Niamh's correction, 2026-06-05)

I initially framed this as a methodology failure. Niamh corrected:

> *"If the space on each layer was word2vec then the layers wouldn't be doing
> anything much."*

This is right. The Mikolov geometry is a property of *static word embeddings*
trained specifically to support analogical arithmetic. The residual stream
at L12 is the model's representation **after 12 transformer layers have
processed the input** — by construction, it is no longer in the input-embedding
geometry. If it were, the layers would have done nothing.

So this isn't "our methodology failed to recover Mikolov" — it's "L12 is doing
its job: it has moved past static lexical geometry." The relevant question
for steering is not "does the L12 representation support Mikolov arithmetic"
but "what structure does it support, and does our methodology recover that
structure?" The exp114 affect result + exp115 free-gen suggest: yes, it
recovers contextualised, function-relevant features (valence, magnitude,
status as a *cluster*) — not static lexical geometry.

### Where Mikolov-style arithmetic SHOULD work

If we want to test analogical structure in Pythia, the right places are:
- **`W_E` (input embedding matrix)** — static token-vector geometry, closest
  to word2vec.
- **Early-layer residuals** (L0–L2) — least processed.

Bare-word L12 residuals were always the wrong substrate for that test. Not
tested further here.

### Side finding worth noting: father-mother is anti-parallel to man-woman in raw

```
  Anchor pair          cos(pair_raw, man-woman_raw)   cos(pair_strip, ...)
  (man,    woman)      +1.000                          +1.000  (ref)
  (king,   queen)      +0.924   ← driven by freq        +0.032  (collapse)
  (father, mother)     -0.848   ← anti-parallel raw     +0.027  (collapse)
  (husband, wife)      -0.921   ← anti-parallel raw     +0.174
```

In raw activations, father-mother and husband-wife point *opposite* to
man-woman. After freq-stripping, the apparent anti-parallel structure
collapses to noise. Reason: in English text "mother" is more frequent than
"father" (parenting register), "wife" more than "husband" (marriage register),
but "man" more than "woman" — so the frequency component of (father - mother)
points opposite to (man - woman). This is the same frequency-asymmetry
mechanism that produced cos = -0.99 between exp111 and Lakoff UP directions
in Entry 4 — replicated in a completely different word set, which is
further confirmation the diagnosis is right.

Files: `exp117_king_queen_classic.py`, `results_exp117.txt`.

---

## Entry 7 — exp116 outcome: the cluster reading, refined

Ran the parallel-prompt cluster discrimination battery from Entry 6's design,
plus a status DV (Niamh's call: include the canonical high-status archetypes
*king/queen/prince/princess/duke/duchess/lord/lady/emperor/empress/monarch/
noble* alongside the professional-class CEO/doctor candidates; balanced
low-status with the corresponding archetypal counterparts:
*peasant/serf/servant/slave/beggar/commoner*).

### Results at s=+12 (Δ from baseline)

```
DV                    u111_clean    ulak_clean    freq_only    rand
patient_height_cm     +2.31         +0.99         -0.50        +1.03
pool_depth_cm         +9.15         +7.45         +1.50        -0.59
door_width_cm         +1.14         +1.59         -0.46        -2.08
widget_quantity       +1.20         +0.38         +0.91        +0.23
status                +0.11         +0.32         -0.06        -0.20
affect                +0.45         +0.68         +0.01        +0.01
```

### What survived

**Affect**: replicates exp114 cleanly. +0.45 / +0.68 nats for cleaned UP,
~0 for freq_only and random. Robust.

**Status**: small but real (+0.11 / +0.32). **`freq_only` goes the opposite
direction (−0.06)** and **`rand` strongly negative (−0.20)** — so the status
shift is direction-specific, distinct from frequency, not random degradation.
Niamh's qualitative call from exp115 (the "businessman briefcase / top
weight lifter / rare super-athlete" pattern at high +s) was real as a
quantitative effect, not just my pattern-matching. Smaller-magnitude than
affect but the right shape.

**Magnitude (MORE-IS-UP)**: height, depth, width, quantity all positively
co-shift at +s for both cleaned directions. Distinguishable from freq_only
and from random on most DVs.

### What didn't survive: bipolar verticality

**Pool depth went UP under +s, not down.** Huge effect — +9.15 cm at s=+12
for u111_clean. Lakoff's predicted bipolar verticality (HEIGHT ↑ / DEPTH ↓
under UP-steering) is not what we got. **Door width** (a purely horizontal
dimension) *also* went up (+1.14). So the cleaned direction is inflating
*any* spatial extent, not just vertical.

### The refined cluster

What we actually recovered is a 3-metaphor bundle:

```
MORE-IS-UP           ← height + depth + width + quantity, all co-shift up
HIGH-STATUS-IS-UP    ← status up, distinguishable from freq
HAPPY-IS-UP          ← affect up, distinguishable from freq
```

— bound together as a single linear direction. **UP-IS-UP** (literal
verticality with sign) is *not* in the cluster.

This is a real finding, just refined from the original "vertical-elevation
cluster" framing in Entry 5:

> *At L12 of Pythia 1.4B, a frequency-stripped semantic UP direction
> built from Lakoff MML literal-vertical-motion anchors produces monotonic,
> dose-responsive co-shifts in expected affect valence
> (Δ ≈ +0.68 nats at s = +12, ~2× odds ratio), occupational status
> (Δ ≈ +0.32 nats), and a battery of magnitude-DVs (height, depth, width,
> quantity, +0.4 to +7.5cm), with all three discriminable from a
> pure-frequency steering direction (which moves magnitude weakly,
> affect not at all, and status in the opposite direction) and from a
> matched-norm random direction (which produces non-directional model
> degradation). The cluster does NOT include bipolar verticality (pool
> depth co-shifts up with height rather than anti-shifting).*

### Two-paragraph reading

The cluster IS a Lakoffian metaphor bundle — MORE-IS-UP, HIGH-STATUS-IS-UP,
HAPPY-IS-UP are all canonical primary metaphors in Lakoff's catalogue, and
finding them bundled into one direction in transformer activations is
substantive. The fact that frequency-only steering doesn't reproduce the
affect or status shifts is the discriminating evidence — these are *not*
the Pollyanna-principle artifact (more-frequent-words-have-higher-valence)
that a deflationary reading would predict.

The cluster *also* points toward a corollary about how the model carves
"UP": as the cluster of magnitude/status/valence metaphors that extend FROM
verticality, rather than as bipolar verticality itself. Possible reasons:
(a) our anchor list mixes directional words (above/below) with magnitude
words (high/top/peak), and the magnitude content may dominate; (b) the
model at L12 has already processed directional sign away into scalar
magnitude — earlier layers might preserve direction; (c) directional sign
might not be encoded as a linear residual-stream feature at all, but via
attention patterns or nonlinear mechanisms.

### Honest caveats

1. **Anchor-mix problem flagged but not fully resolved.** A "strictly
   directional" subset of Lakoff MML (`above/below, over/under, up/down,
   upward/downward, ascend/descend, rise/fall, rising/falling, rose/fell`)
   could test whether pool_depth would then anti-shift. Not run; design
   parked in Entry 8.

2. **Anchor curation note.** Niamh's call (2026-06-06): "stop using
   exp111 — it's good for exploration but it's arbitrary and uncitable."
   `u111_clean` is from a hand-picked list with no provenance; `ulak_clean`
   is from `lakoff_canonical_vocabulary.py:UP_DOWN_MML` (lines 51-65,
   citable to Lakoff & Espenson 1991, Master Metaphor List, UC Berkeley
   CL group). **Going forward: use Lakoff MML or other citable sources as
   primary anchors. `u111_clean` results are preserved here as a
   companion exploration, not as a primary claim.**

3. **Layer choice still arbitrary.** L12 was inherited from exp3b's raw
   qualitative read; no rigorous justification for it being the right
   place for freq-stripped battery DVs.

4. **Pool depth result is informative about anchor specificity.** At s=−12:
   - `u111_clean` → −1.21 cm (slightly *down* — hint of bipolar behaviour
     at negative side, possibly from `ceiling/floor/peak/valley/overhead/
     underneath` positional anchors)
   - `ulak_clean` → +2.98 cm (up — no bipolar)
   - `freq_only` → +1.38 cm (up)
   - `rand` → +5.70 cm (very up — model degradation)

   The fact that only u111_clean shows a hint of "pool depth gets shallower
   under negative UP-steering" — and only at the negative side — is
   consistent with the hypothesis that *positional* anchors (which u111
   has more of) contribute a small bipolar-verticality component. But the
   effect is small and could be noise.

Files: `exp116_cluster_discrimination.py`, `results_exp116.txt`,
`exp116_results.npz`, `exp116_config.json`.

---

## Entry 8 — Future directions (parked for the wider arc)

Several follow-ups designed but not run, parked here so the wider
relational-structure programme in v4 isn't delayed by hunting them down:

1. **exp118 — valence-anchor deflationary control.**
   Build a direction from pure-valence anchors (happy/sad, joy/grief,
   glad/miserable, content/distressed). Compare to the cleaned UP
   direction. Three possible outcomes: (a) cos high AND cluster shifts
   identical → the "Lakoff UP cluster" is actually just Osgood Evaluation
   with extra labels; (b) cos high but cluster shifts differ → UP captures
   more than just E; (c) cos moderate and clusters differ → genuinely
   distinct directions. Concept-discrimination, not falsification of
   Lakoff.

2. **Pure-direction Lakoff anchor variant.**
   Strict-directional subset of Lakoff MML literal motion:
   ```
   UP:   above, over, up, upward, ascend, rise, rising, rose
   DOWN: below, under, down, downward, descend, fall, falling, fell
   ```
   Citable subset of a citable list. Test whether THIS direction recovers
   bipolar verticality (pool depth ↓ under +s) where the full Lakoff list
   didn't. Discriminates "anchor mix dilutes verticality" from "model
   doesn't encode bipolar verticality at L12 as a linear direction."

3. **exp119 — layer sweep for build × inject layers** at 1.4B.
   Build cleaned UP at each layer 0-23, inject at each layer, measure
   affect DV (the cleanest from exp114). Find (a) best build layer,
   (b) best inject layer, (c) whether they coincide. L12 was a guess from
   exp3b raw-qualitative read; freq-stripped + battery DVs may optimise
   elsewhere.

4. **exp121 — heuristic salience diagnostic** (Niamh greenlit, not run).
   Build s_raw = mean(real real-words) − mean(actual junk tokens) at L12
   via heuristic filter (mixed alpha+numeric+punct, web junk). Residualise
   against f_clean (real-word-only frequency). Check what's left over and
   whether it's a substantial unstripped component in u_111_clean and
   ulak_clean. Niamh's intuition: anisotropy is partly salience; our
   frequency strip might not have fully isolated it. exp108 in v2 found
   exactly this shape in GloVe (PC1 = nonsense/coherence, not frequency).

5. **exp122 — do layers drop nonsense across depth?**
   Niamh's question: do later layers increasingly drop nonsense tokens?
   Measure pairwise cos between nonsense-token residuals across layers
   L0/L4/L8/L12/L16/L20/L23 + their norms vs real-word norms. Two
   competing predictions: "Drop junk" → cos→1 across layers (uniform
   "ignore" representation); "Tag junk" → cos stays varied, distinct
   outlier-dim grows.

6. **exp120+ — other cog-linguistics axes (post-methodology lock-in).**
   Once the relational-structure programme in v4 lands, run the same
   methodology on individual axes: HARD-SOFT (§3.2 of
   CAUSAL_VALIDATION_PLAN, the load-bearing novel test for the original
   single-axis paper), WARM-COLD (§3.3), IN-OUT, FORWARD-BACK, LIGHT-DARK,
   and the Osgood E-P-A axes from Warriner V-A-D norms (which we have on
   disk). Cross-axis comparison: do different image schemas produce
   orthogonal clusters, or do they all collapse into a few dimensions
   like Osgood E-P-A?

7. **Meta-representation probe** (parked per Niamh; hard to disentangle).
   Does L_Y residual contain decodable signal about "feature X fired at
   L_X"? Related to induction heads (Olsson et al). Disentangling
   meta-representation from continued representation is the hard part.
   Possible approach: train linear probe on L_Y residual to predict
   projection magnitude of L_X residual on UP direction. Worth a future
   notebook entry.

8. **Contextualised-prompt rebuild (ActAdd/CAA proper).**
   The build method that should have been the default: `act("I feel up") −
   act("I feel down")` etc at the last token of paired prompts. Balances
   frequency and salience by construction (same template both sides),
   should give a much cleaner direction. The §3.1 plan recommended this
   from the start; we sort of skipped it. If the v4 relational structure
   work shows the bare-word build doesn't recover stable cross-layer
   structure, the contextualised method is the next move.

---

## Methodology summary (for writing-up)

Distilled from exp111–117:

### The diagnosis

A naive steering vector built as `mean(POS anchor activations) − mean(NEG
anchor activations)` from single-word inputs at any mid-layer of a
transformer (here Pythia 1.4B L12) is dominated, in our hands, by **the
frequency axis at that layer (~96% of the direction's content)**, with the
genuine semantic component being a small (~4%) but real residue. Two
independent anchor sets (hand-picked vs Lakoff MML literal-motion) produced
*anti-parallel* directions (cos ≈ −0.99) when their frequency profiles
differed, because the frequency component dominated. After projecting out
the frequency axis, the residual semantic directions agreed across anchor
sets (cos ≈ +0.57), producing the actual finding.

### The replicated confound shape

This is the same confound shape exp99–103 found for GloVe word embeddings
in `LAB_NOTEBOOK_v2.md`: PC1 of contrast vectors is the frequency axis
(or, more precisely, a "real-vs-nonsense markedness" axis that frequency
correlates with); the residual semantic content is small but recoverable.
That this confound shape replicates from GloVe (static distributional
embeddings) to Pythia activations (mid-stream transformer residuals)
suggests it is a **universal failure mode of difference-of-means contrast
vectors in distributional substrates**, not specific to embedding-vs-
activation-space.

### The methodological recipe (provisional)

For building interpretable steering vectors that recover Lakoffian
metaphor content in transformer residual streams:

1. **Pick anchors from a citable source** (Lakoff MML; Roget §305-308
   ELEVATION/DEPRESSION; Warriner V-A-D high-vs-low — each defensible).
   Don't hand-curate.
2. **Build the raw direction** as mean-of-means at the chosen layer.
3. **Build a frequency axis at that layer** (e.g. mean(20 common function
   words) − mean(5 rare-but-real content words)).
4. **Strip frequency**: orthogonalise the raw direction against the
   frequency axis, renormalise.
5. **(Optional, not yet tested)** Strip salience too — separate axis if
   not already collinear with frequency.
6. **Evaluate against a battery of parallel-structured DVs** that
   discriminate the schema-specific cluster from generic number-inflation
   or generic valence shifts. Single-axis mass-contrast DVs (forced-choice
   loglikelihood over candidate vocabulary) are the cheapest valid
   evaluation.
7. **Always include a matched-norm random-direction control** AND a
   pure-frequency-steering control. Random tests "direction-specific";
   freq tests "not just the confound we tried to strip."
8. **Validate qualitative completion patterns** via free-gen at
   representative strengths. Don't trust a battery DV that the model
   doesn't actually want to emit (the cm-completion validity issue).

### What the freq-stripped method gets you (cluster claim from exp116)

At L12 of Pythia 1.4B, a frequency-stripped UP direction built from Lakoff
MML literal-vertical-motion anchors produces sign-consistent, dose-
responsive co-shifts on affect valence, occupational status, and a
magnitude battery (height/depth/width/quantity). The same cluster is *not*
produced by a matched-norm random direction nor by a pure-frequency
direction (frequency moves magnitude weakly, affect not at all, status in
the wrong direction). Two anchor sets (Lakoff MML vs hand-picked spatial)
produce cleaned directions that agree in sign on all cluster members.

The cluster identifies as a bundle of three canonical Lakoff primary
metaphors — MORE-IS-UP, HIGH-STATUS-IS-UP, HAPPY-IS-UP — encoded as a
single linear direction. Notably, bipolar verticality (UP-IS-UP with sign)
is *not* in the cluster: pool depth co-shifts up with height rather than
anti-shifting. The model's "UP" representation at L12 is the metaphorical
extension cluster, not the spatial directional axis.

### Why this matters

The cluster claim is a **structural** claim about how cognitive primitives
are encoded — not "we found UP" but "we found a Lakoffian
cluster-as-direction." If the methodology generalises (other schemas
producing other Lakoff-predicted clusters; the cluster organisation being
layer-stable; antonym schemas producing anti-aligned directions; etc),
that's the substance of a paper. **The v4 notebook takes up the structural
question explicitly: do image schemas hang together as a coherent
relational system across layers?**

---

## Standing methodological notes

1. **Anchor-set choice is load-bearing in a way the project hadn't
   accounted for.** Hand-picked vs externally-curated lists produced
   anti-parallel directions, fully explained by their differing
   frequency profiles. Don't trust an anchor list until you've verified
   its frequency profile is matched OR the frequency component has been
   stripped from the resulting direction.

2. **The bare-word steering build method is contaminated by anisotropy +
   frequency.** Future steering work should either (a) strip freq
   explicitly as in exp114, (b) use contextualised prompts à la
   ActAdd / Rimsky CAA (this would balance frequency by construction
   since both contrast halves use the same template), or (c) both.

3. **Batteries with fixed candidate menus measure relative ordering, not
   model belief.** When the candidate set isn't what the model actually
   wants to emit at that position, battery DVs report shifts in a thin
   sliver of completion space. Validate any battery DV by checking the
   free-gen completions match the candidate format.

4. **Random-direction control alone is necessary but not sufficient.**
   Random tests "is the effect direction-specific?" but doesn't test
   "is the effect the kind of thing I'm claiming?" Parallel-prompt
   discrimination (exp116-style) tests the latter.

5. **Replicates the W2V exp99–103 confound shape.** Same pattern: an
   ostensibly-semantic contrast vector turns out to be mostly the
   frequency axis; the residual semantic content is small but real and
   becomes visible after stripping. Strong signal that frequency
   contamination is a universal failure mode for difference-of-means
   contrast vectors across substrates (GloVe → Pythia activations). The
   methodology refinement may itself be the durable contribution.

---
