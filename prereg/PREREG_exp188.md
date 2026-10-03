# PRE-REGISTRATION — exp188: DOES STEERING UP MAKE THE LAYERS AGREE?

Written 2026-10-02 20:38 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 20:35:
"i did have other ideas about how UP could be grounded in coherence
between layers. you can decide for yourself if any of its worth
testing", with her August notes pasted (verbatim in
HYPOTHESIS_UP_grounding_2026-10-02.md). Frozen at the commit carrying
this text.

## Plain summary

Her August design, never run: add the UP direction at ONE early layer
and see whether the layers after it agree with each other more. Her
sign: UP is agreement ("legs, hips and shoulders not agreeing means
you're on the floor"). Her instrument: how far a token's state ends
up, divided by how far it travelled getting there. If the layers'
contributions add up, that ratio is high; if they fight, it is low.

Her rules, kept: steer at one site only (steering at many layers
raises agreement by construction); steer early, so there is depth
left to measure over. Her worry, addressed: agreement and norm might
be one thing, so the endpoint norm is reported beside the path
measure.

## Design

Pythia 410M, real start token. Text: the lab's 120 everyday sentences
(as exp181).

Directions, all built the clean way (single-token words inside
neutral sentences with a start token: exp175's cached states at the
steering layer; anisotropy and frequency directions projected out;
unit length):
- **UP**: exp175's 35 spatial UP words minus its 39 DOWN words.
- VALENCE: exp175's 22 positive minus 19 negative words (reported).
- **RANDOM-WORD NULL**: N_NULL directions, each 35 random words minus
  39 other random words from exp175's other analysis words.

Steering: at layer SITE only, add strength x direction to the state
at every non-start position. Strength = C times the mean norm of the
unsteered state at that layer over all sentence tokens, C in
{0.5, 1.0}, both signs.

Measured on every non-start token, over the layers AFTER the site
(blocks SITE+1 .. 23; each block's write is its output minus its
input):
- **STRAIGHT (PRIMARY)**: norm of the sum of those writes, divided by
  the sum of their norms.
- WRITE_COS (secondary): mean cosine between consecutive writes.
- reported: path length, net displacement, and the norm of the final
  state (the endpoint quantity).
Per sentence: the mean over its tokens.

Statistic: for a direction d and strength C,
A(d, C) = STRAIGHT with +C*d minus STRAIGHT with −C*d, per sentence,
averaged over sentences. Anything steering does regardless of sign
cancels; what is left is what pointing UP does that pointing DOWN
does not.

## RULE PARAMETERS (frozen)

  SITE = 4   N_NULL = 40   SEED_NULL = 188   ALPHA = 0.05   N_PERM = 10000

## Decision rule

At each of the two strengths, A(UP) is tested against zero across
sentences (sign-flip p) and placed in the distribution of A over the
RANDOM-WORD NULL.
- **UP_RAISES_AGREEMENT**: at BOTH strengths, A(UP) > 0 with
  p < ALPHA and above the null's 95th percentile. Her prediction.
- **WRONG_SIGN**: at both strengths, A(UP) < 0 with p < ALPHA and
  below the null's 5th percentile.
- **NOT_BEYOND_RANDOM**: anything else.
The same is reported for WRITE_COS and for the endpoint norm. If
STRAIGHT and the endpoint norm give the same picture for UP against
the null, the two have not come apart.

## Committed predictions

- P1 UP_RAISES_AGREEMENT: **8%**
- P2 WRONG_SIGN: **8%**
- P3 NOT_BEYOND_RANDOM: **84%**
(Today's record for ideas of this kind: nine nulls, and one
replicated effect that turned out to be word order.)

## What I will NOT do

- No moving the site, the strengths or the measure after seeing
  numbers.
- No reading A(UP) against zero without the random-word null: a
  random direction's A is not expected to be zero either.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 21:02 IST, 2026-10-02)

Raw output: exp188_output.txt. Numbers: exp188_results.json; every
cell in exp188_cells.npz. Mac GPU. No deviations. Unsteered levels:
STRAIGHT 0.2220, WRITE_COS 0.0401, final-state norm 50.69.

A(UP) = measure with +UP minus measure with −UP, mean over the 120
sentences; the null is the same quantity for 40 random-word
directions.

| measure | strength | A(UP) | p | sentences positive | null 5%..95% | UP's percentile |
|---|---|---|---|---|---|---|
| **STRAIGHT** | 0.5 | −0.0003 | 0.62 | 60 | −0.0090..+0.0100 | 45th |
| **STRAIGHT** | 1.0 | +0.0258 | 0.0001 | 120 | −0.0216..+0.0213 | 98th |
| WRITE_COS | 0.5 | +0.0134 | 0.0001 | 103 | −0.0227..+0.0110 | 95th |
| WRITE_COS | 1.0 | +0.0274 | 0.0001 | 120 | −0.0444..+0.0282 | 92nd |
| final-state norm | 0.5 | −0.92 | 0.0001 | 43 | −2.92..+4.11 | 35th |
| final-state norm | 1.0 | +3.20 | 0.0001 | 104 | −8.97..+8.97 | 60th |

**VERDICT (STRAIGHT, both strengths required): NOT_BEYOND_RANDOM.**
The same rule on WRITE_COS and on the final-state norm:
NOT_BEYOND_RANDOM.

### What the table shows, beyond the label
- At the stronger strength, pointing UP leaves the later layers
  agreeing more than pointing DOWN does, in every one of the 120
  sentences, by more than 39 of the 40 random-word directions manage.
  At the weaker strength there is nothing on STRAIGHT.
- WRITE_COS leans the same way at both strengths (just above the
  null's 95th percentile at 0.5, just below it at 1.0).
- The final-state norm does NOT stand out at either strength (35th
  and 60th percentile). So the path measure and the endpoint measure
  did come apart here, which is the separation Niamh's August notes
  asked for.
- Steering either way lowers agreement relative to no steering
  (STRAIGHT −0.025 with +UP, −0.051 with −UP at strength 1.0).
  Pointing DOWN disrupts it about twice as much as pointing UP.
- The VALENCE direction shows a similar asymmetry at strength 1.0
  (+0.019, just under the null's 95th percentile), so whatever this
  is may belong to "positive pole" directions more generally.
- Forty null directions is a coarse yardstick for a 95th percentile.

### Standing
A null under the registered rule, with a lean in the predicted
direction at one of two strengths. Given how the day's other lean
ended (exp175's, which turned out to be word order), it should be
treated as a lead and nothing more until it is rerun with more null
directions, a second steering site and a second model.

### Grades
- P1 UP_RAISES_AGREEMENT (8%): did not occur.
- P2 WRONG_SIGN (8%): did not occur.
- P3 NOT_BEYOND_RANDOM (84%): HIT.
