# PRE-REGISTRATION — exp166: BALANCE ↔ attention-entropy, THIRD prompt set
# (confirmatory, all three lineages)
Written 2026-06-13, BEFORE any code or model run on the third prompt set.
(Claude; Niamh to review before execution.)

## Provenance / why this needs a prereg
exp161 (pre-registered) confirmed GPT-2-medium grounds BALANCE in
attention entropy on a FRESH (second) prompt set, and reported Pythia's
BALANCE↔entropy ≈ 0 under identical norm controls — read at the time as
the multiple-realizability fingerprint (entropy in GPT-2, norm in
Pythia).

exp164 (DIAGNOSTIC depth map, all layers × 3 models, quadratic control
stack) then surfaced TWO post-hoc bands that the exp161 decision layers
had straddled:
  - PYTHIA L11–12: C2q = −0.162 [−0.264,−0.051] (L11),
                   −0.211 [−0.304,−0.117] (L12). Survived the quadratic
                   norm-leak control stack (NOT explained by curved norm
                   coupling). Contiguous, both CIs exclude 0.
  - LLAMA L5–7:   C2q = −0.118 [−0.218,−0.023] (L5),
                  −0.171 [−0.296,−0.051] (L6),
                  −0.240 [−0.330,−0.157] (L7). Contiguous run of 3, all
                  CIs exclude 0, carriers 0.65/0.73/0.75.

These are POST-HOC: found by scanning a 24/24/16-layer depth map after
looking, on exp161's prompts. Forking paths / winner's curse apply.
exp164 explicitly logged them as leads "NOT a result," to be confirmed
on fresh prompts. exp166 is that confirmation. exp161's prompts are now
"used" for Pythia and Llama (the depth map ran on them), so exp166 MUST
use a third, never-run prompt set.

## The claim at stake (held as a strong lead until this runs)
If BOTH bands replicate on fresh prompts, BALANCE↔entropy grounding is
present in ALL THREE lineages (GPT-2 already triple-established:
exp160 post-hoc, exp161 confirmatory, exp164 18-layer backbone). That
INVERTS last night's summary:
  - OLD: entropy-grounding is GPT-2-idiosyncratic; Pythia grounds
    BALANCE in norm; grounding is the idiosyncratic part.
  - NEW (if confirmed): entropy-grounding is the UNIVERSAL embodiment
    (every attention transformer thermostats entropy via query
    magnitude ‖q‖², a readout of residual DIRECTION); Pythia's
    norm-coupling is the idiosyncratic EXTRA carrier, not the grounding.
Mechanistic licence (not under test here, motivates taking it
seriously): ‖q‖² = LN(x)ᵀWqᵀWq LN(x) sets attention peakedness from
residual direction in any attention transformer regardless of
normaliser. This prereg tests only WHETHER the coupling replicates, not
the mechanism.

## Hypothesis under test
H: BALANCE↔entropy coupling is real (not depth-map forking-path noise)
in Pythia (L11–12) and Llama (L5–7), reproducing on a fresh prompt set
under the standard quadratic control stack.

## Design
1. FRESH DATA: 40 new generic prompts (third set), frozen in Appendix A
   AFTER an automated overlap check passes (see Integrity below). Same
   genre as exp160/exp161 (everyday scenes, varied register), NOT
   stuffed with balance/stability vocabulary, NO token matching any of
   the 8 schemas' axis-defining words, ZERO overlap with the exp160 (40)
   or exp161 (40) prompt sets. The BALANCE projection must vary
   naturally.
2. Models + GATED decision layers:
   - pythia-410m  → BAND {11, 12}              (committed gate)
   - Llama-3.2-1B → BAND {5, 6, 7}             (committed gate)
   - gpt2-medium  → {8, 12, 16}                (POSITIVE CONTROL, below)
   Non-gated layers recorded for context (esp. Pythia L5, L18 — handoff
   lead #6 — and Llama's sign-flip layers L0/L12), no bet attached.
3. Per-layer machinery EXACTLY as exp161/exp164 (import from
   attn_entropy_lib + exp161.build_layer_dirs — do NOT re-type frozen
   maths or prompts): isolated-word residuals → aniso+freq strip → 8
   schema axes; per-query normalised attention entropy; per-token
   unit-residual projections; C5 axis-word token exclusion.
4. d_norm: HELD-OUT estimation (est on half the vocab, carrier checked
   out-of-sample). Full-set d_norm is circular and is used nowhere in
   the verdict.

## Control stack (standard for entropy work as of exp164)
- C2 (HEADLINE, linear): partial r(BALANCE_proj, entropy | position,
  scalar norm, token d_norm_ho projection); axis = BALANCE projected
  ⊥ d_norm_ho. Both sides norm-orthogonalised.
- C2q (HEADLINE under quadratic stack — THE statistic the gates read):
  C2 additionally controlling z_norm², z_dnorm², z_norm·z_dnorm, and
  rank-norm. exp164 validated this stack synthetically (kills a planted
  curved norm leak −0.59→−0.02, spares genuine coupling −0.61→−0.61).
  The bands being confirmed were defined under C2q, so the gate reads
  C2q.
- C3 (schema-orth, NON-GATING): axis additionally ⊥ the other 7 schema
  axes. Reported for context (BALANCE-unique vs shared schema variance);
  per exp161, C3 is explicitly non-gating because schema axes are known
  to be linearly non-orthogonal and nonlinearly entangled (exp161 S1).
- C4 (specificity): all-8-schema C2q partial table, BALANCE rank by |r|
  at each gated layer. Reported, not gated (these are smaller couplings
  than GPT-2's; rank is context).
- Inference honesty: tokens cluster within prompts. Pooled n (~470–500)
  overstates evidence. Gate uses the prompt-level cluster bootstrap 95%
  CI (resample prompts, 1000 reps) — NOT the pooled p. Pooled point
  estimate reported alongside.
- C5 axis-word exclusion: any prompt token whose lowercased string
  matches an axis-defining word for ANY of the 8 schemas is excluded
  from all correlations.

## Precondition (gate validity, exp158 lesson)
Held-out d_norm carrier (out-of-sample corr) ≥ 0.50 at every GATED layer
of a model. If a model's gated layers fail this, that model's verdict =
INVALID (norm-orthogonalisation meaningless), NOT a null. (exp164
carriers at the bands: Pythia 0.92/0.91, Llama 0.65/0.73/0.75 — expected
to pass, re-checked on fresh data.) Selftest must include a
"precondition-fails + headline-null" discrimination case.

## POSITIVE CONTROL (prompt-set validity)
GPT-2's BALANCE↔entropy coupling is triple-established. On the third
prompt set it MUST reproduce: C2q negative with bootstrap CI excluding 0
AND |C2q| ≥ 0.10 at ≥2 of {8, 12, 16}. If GPT-2 FAILS this, the third
prompt set is broken (it killed a triple-confirmed effect) → whole run
INVALID, re-author prompts. This guards against a dud prompt set
masquerading as a Pythia/Llama null.

## Gate definition (per model, applied to Pythia and Llama)
A model's band GATE FIRES iff ALL hold on the fresh prompts under C2q:
  (a) every band layer is NEGATIVE in sign, AND
  (b) a CONTIGUOUS run of ≥2 band layers each has bootstrap 95% CI
      excluding 0, AND
  (c) pooled |C2q| ≥ 0.10 at each layer of that ≥2-layer run, AND
  (d) precondition (carrier ≥ 0.50) holds at those layers.

JUDGMENT LEVER (flagged for Niamh): floor is 0.10, NOT exp161's 0.15.
Rationale — these bands are genuinely smaller than GPT-2's 0.25–0.40,
and fresh-prompt winner's-curse attenuation is expected (exp161 P1
predicted attenuation to 0.10–0.30). The HARD requirement is sign +
CI-excludes-0 on a contiguous ≥2-layer run; the 0.10 floor is a
secondary magnitude guard. Demanding 0.15 risks a false-null on a real
small effect. If Niamh prefers 0.15, change DECISION_HI here before any
run and the predictions below stand.

## Decision rule
- BOTH gates fire → UNIVERSALITY INVERSION CONFIRMED. BALANCE↔entropy
  grounding present in all three lineages. Entropy = candidate universal
  embodiment; Pythia norm-coupling = idiosyncratic extra. → unblocks
  Claim-3 rewrite AND the causal test exp165 (BALANCE-steer → entropy).
- EXACTLY ONE fires → PARTIAL. That lineage joins GPT-2 (now 2 of 3);
  the other model's exp164 band was depth-map forking-path noise.
  Diagnose the failing model before any universality claim; no causal
  work on the failing lineage.
- NEITHER fires → INVERSION DOES NOT HOLD. exp164's Pythia/Llama bands
  were post-hoc noise. Revert to the exp161 reading: GPT-2 entropy +
  Pythia norm, each idiosyncratic. Causal exp165 still licensed for
  GPT-2 alone.
- POSITIVE CONTROL fails / precondition fails → INVALID (see above);
  fix before reading any band number.

## Committed predictions (calibration: this is REPLICATION-WITH-CONTROLS
of an OBSERVED effect, not a mechanism story — the rule earned on
2026-06-13 says bet such replications STRONG; mechanism bets remain
0-for-8 and none is made here)
- P1 (Pythia L11–12 GATE FIRES): BET STRONG. The band already survived
  the quadratic stack with both CIs excluding 0; this is a clean
  replication target. Expect L12 |C2q| ≈ 0.12–0.21, L11 ≈ 0.10–0.16,
  contiguous, both CIs < 0.
- P2 (Llama L5–7 GATE FIRES): BET MODERATE-STRONG. Llama was exploratory
  in exp164 and is the noisiest substrate (sign-flips at L0 +0.25, L12
  +0.20), but the L5–7 run was clean and monotone-deepening
  (−0.12 → −0.17 → −0.24, 3/3 CIs < 0). Expect ≥2 of {5,6,7} to fire,
  most likely the deeper L6–L7; L5 (−0.118 in exp164) is the marginal
  one that may dip under the 0.10 floor on fresh prompts.
- P3 (GPT-2 positive control fires): BET VERY STRONG (triple-confirmed).
- P4 (overall): BOTH gates fire → universality inversion confirmed.
  BET STRONG-ish but this is the conjunction P1∧P2, so it inherits P2's
  Llama uncertainty. If forced to a single number I expect "both fire"
  more likely than not, with the Llama gate the live risk.
- Non-gated context, NO bet: Pythia L18's isolated −0.221 (lead #6) —
  record whether it reappears on fresh prompts; Pythia L5 (−0.144);
  Llama L13 (−0.183) and the L0/L12 positive sign-flips.

## Integrity checks (run BEFORE any model; conventions from 2026-06-13)
- FROZEN-PROMPT CHECKSUM ASSERT: exp166 module asserts a checksum over
  its 40 prompts that must equal the value frozen in this prereg.
- AUTOMATED OVERLAP CHECK (must pass before prompts are frozen here):
    (i)  no prompt token matches any axis-defining word of ANY of the 8
         active schemas (UP-DOWN, IN-OUT_CLEAN, FORWARD-BACK,
         PATH-MOTION, LIGHT-DARK, FORCE, BALANCE, DIFFICULTY-BURDEN);
    (ii) zero verbatim-prompt overlap and zero high-overlap near-dupes
         with exp160's 40 or exp161's 40 (Jaccard on token sets < 0.5);
    (iii) exactly 40 prompts, all unique.
  This check's own logic is itself tested (exp161 v1.1 lesson: the first
  overlap checker was buggy and checked the wrong text region).
- IMPORT, don't re-type: frozen maths from attn_entropy_lib +
  exp161.build_layer_dirs; quadratic stack from exp164.
- SYNTHETIC discrimination test BEFORE the real run: must include a
  precondition-fails + headline-null case AND a planted-coupling case
  the stack should preserve (exp164 lesson: check the synthetic test
  tests something).

## What I will NOT do
- No swapping prompts after seeing a null.
- No relaxing the 0.10 floor or the contiguity requirement post-hoc
  (Niamh may set it to 0.15 or 0.10 BEFORE the run; not after).
- No promoting Pythia L5/L18 or Llama L13 to "confirmed" if the
  committed bands fail — they are non-committed context and any follow
  is a new prereg.
- No quoting C1/C2 (linear) as the result if C2q (quadratic stack) is
  weaker. C2q is the headline; the bands were defined under it.

## Appendix A — frozen fresh prompt set (40)
FROZEN 2026-06-13 after exp166_prompt_verify.py passed all three checks
(no axis words for any of the 8 schemas; zero verbatim/near-dupe overlap
with exp160 or exp161; 40 unique). The exp166 run module must import
these or assert this checksum before any model loads.

  prompt-set checksum (sha256[:16] of "\n".join): 4d54ff4297bd7e2c

1.  The barista frothed the milk and dusted cocoa across the foam.
2.  A magpie landed on the fence and eyed the picnic table.
3.  The mechanic drained the oil and wiped his hands on a rag.
4.  Her phone buzzed twice and then went silent in her pocket.
5.  The chef plated the fish and garnished it with dill.
6.  Children traced chalk animals on the playground tarmac.
7.  The cobbler stitched the sole and tapped in fresh nails.
8.  Rain freckled the windscreen as the wipers squeaked.
9.  The cashier counted the float and zipped the bag shut.
10. A pot of soup simmered while she chopped parsley.
11. The electrician labelled each wire and taped the panel shut.
12. Geese crossed the meadow in a ragged honking row.
13. The potter wet the clay and the wheel began to spin.
14. She knitted two rows and miscounted the stitches.
15. The fishmonger packed the crab in crushed ice.
16. A drone skimmed the wheat and snapped photographs.
17. The waiter recited the specials and refilled their water.
18. Frost coated the lawn and the engine grumbled to life.
19. The tailor pinned the hem and chalked a curve at the waist.
20. Bees worked the lavender while the afternoon hummed.
21. The plumber bled the radiator and the knocking finally quit.
22. She sorted the laundry into colours and folded the towels.
23. The butcher boned the lamb and tied it with kitchen twine.
24. A toddler smeared yogurt across the tray and giggled.
25. The florist clipped the stems and bound them with raffia.
26. Thunder rolled somewhere distant and the dog whined.
27. He greased the baking tin and cracked four eggs into the bowl.
28. The usher tore the tickets and gestured toward the seats.
29. Snow muffled the street and the ploughs began their rounds.
30. The vet listened to the cat's chest and frowned a little.
31. A busker strummed a chord and nodded at the gathering crowd.
32. The cleaner mopped the lobby and roped off the wet tiles.
33. Wasps had built a grey nest against the garden shed.
34. The barman polished the tumblers and stacked them by size.
35. A courier left the parcel with the neighbour at number nine.
36. The hikers boiled snowmelt and shared a bar of chocolate.
37. She defrosted the freezer and sponged the puddle dry.
38. The referee blew the whistle and waved play on.
39. Pigeons squabbled around a crust on the pavement.
40. The seamstress threaded the machine and ran a test seam.

---

## RESULT — exp166 (2026-06-13, ran AFTER freezing the above)
Output: exp166_output.txt. Harness validated first (selftests + synthetic
quad-stack discrimination [leak −0.66→−0.04 killed, genuine −0.63→−0.64
spared] + prompt checksum 4d54ff4297bd7e2c). C5 dropped 1–2 axis tokens
per model.

VERDICT: **PARTIAL — Llama confirmed; Pythia NULL; GPT-2 control PASS.**
The universality inversion does NOT hold. Entropy-grounding of BALANCE is
confirmed on fresh prompts in 2 of 3 lineages (GPT-2, Llama); Pythia's
exp164 band was post-hoc noise.

- **Pythia L11–12 GATE = NULL.** L11 C2q −0.007 [−0.109,+0.098], L12
  −0.048 [−0.149,+0.051]. The exp164 band (−0.162/−0.211, "survived the
  quadratic stack, real second-carrier candidate") DID NOT REPLICATE —
  it collapsed to ≈0 with both CIs straddling zero. Not a near-miss; the
  coupling is simply absent on fresh prompts. Context layers confirm:
  Pythia L5 −0.047 (exp164 −0.144, gone), L18 +0.007 (exp164 −0.221
  isolated cell / handoff lead #6, GONE). **exp161's P5 (Pythia
  BALANCE↔entropy ≈ 0 under norm controls) is vindicated on a third
  prompt set; exp164's Pythia entropy bands were forking-path artifacts
  of scanning 24 layers.** Pythia's BALANCE grounding remains NORM
  (exp167 orientation cluster) — unaffected.
- **Llama L5–7 GATE = FIRES.** L5 −0.139 [−0.245,−0.043], L6 −0.168
  [−0.264,−0.081], L7 −0.202 [−0.322,−0.102]; all negative, all CIs
  exclude 0, contiguous good run (5,7), carriers 0.65/0.73/0.75. Close
  replication of exp164 (−0.118/−0.171/−0.240). L5 — flagged in P2 as the
  marginal layer — cleared the 0.10 floor (−0.139). **Llama joins GPT-2:
  entropy-grounding of BALANCE confirmed in a second, unrelated lineage.**
- **GPT-2 positive control = PASS** (prompt set valid). L8 −0.266
  [−0.372,−0.164], L12 −0.320 [−0.403,−0.247], L16 −0.145 [−0.238,−0.051].
  L3 shallow peak −0.322 again (replicates the exp164 shallow-peak shape).
  A prompt set that reproduces GPT-2's triple-confirmed coupling but kills
  Pythia's band is doing its job: the Pythia null is real, not a dud set.

### Prediction scorecard (calibration)
- P1 (Pythia L11–12 fires) — bet STRONG — **MISS.** The strong bet lost.
- P2 (Llama L5–7 fires) — bet MODERATE-STRONG — **HIT** (incl. L5).
- P3 (GPT-2 control) — bet VERY STRONG — **HIT.**
- P4 (both fire → inversion) — **MISS** (only Llama).

### Calibration update (important)
The 2026-06-13 rule "bet REPLICATION-WITH-CONTROLS strong" needs a
boundary: it holds for replications of a PRE-REGISTERED confirmatory
effect (exp161 GPT-2 → replicated three ways), but NOT for replication of
a POST-HOC band lifted from a depth-map SCAN. exp164's Pythia L11–12
*looked* robust — survived the quadratic stack, CI excluded 0 — yet it
was selected by eye from 24 layers, so winner's curse fully applied, and
it died on fresh prompts. **A post-hoc depth-map cell keeps the
deflationary prior even when framed as "replication."** Had we built
exp165 steering on the Pythia band (the tempting next step), we'd have
steered noise. Confirmation-first was the right call.

### Revised picture (replaces last night's kindling)
NOT "entropy-grounding universal across all three." Instead: **partial
multiple realizability** — GPT-2 and Llama ground BALANCE in attention
entropy; Pythia grounds it in residual norm (exp167). 2-of-3 share the
entropy carrier; 1 uses a different body. Mechanistically still licensed
(‖q‖²-thermostat is present in all three), but the coupling only
materializes in two of the families at the layers tested.

### Live leads after exp166
- **exp165 (CAUSAL steering) is now licensed for GPT-2 (primary) AND
  Llama — NOT Pythia** (no entropy coupling to steer there).
- **NEW LEAD: Llama L13.** Non-gated context cell C2q −0.268
  [−0.352,−0.196] — STRONGER than the gated band and replicates exp164
  (−0.183). Llama's entropy coupling may extend deeper than L5–7; its own
  prereg (don't promote a context cell to a result).
- **DEAD:** Pythia L18 and L5 isolated cells (handoff lead #6) — did not
  replicate; drop from the lead list.
- Pythia's NORM grounding (exp167) is untouched and remains its carrier.
