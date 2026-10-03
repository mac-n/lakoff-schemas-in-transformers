# PRE-REGISTRATION — exp190: FOLLOW-UP TO exp188's LEAN (UP STEERING AND LAYER AGREEMENT)

Written 2026-10-02 21:06 IST, before any code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 21:05:
"yes freeze and launch it now pls". Frozen at the commit carrying
this text.

Seen before freezing: exp188's full result (a lean at strength 1.0,
layer 4, Pythia: +UP leaves later layers agreeing more than −UP, 98th
percentile of 40 random-word directions; nothing at strength 0.5).

## Plain summary

exp188 tested Niamh's August design once and got a lean at one of two
strengths. This asks whether the lean holds up: more random
directions to compare against, a second steering layer, a second
model, and three strengths.

## Design

Models: Pythia 410M and GPT-2-medium. Steering sites: layer 4 and
layer 8. That makes four cells. Real start token; the lab's 120
everyday sentences.

Directions, built the clean way at the steering layer (single-token
words inside exp180's eight neutral sentences; anisotropy and
frequency directions projected out; unit length), the same word
lists in both models:
- **UP**: exp180's 31 UP words minus their 31 DOWN opposites. (exp188
  used a different list and different sentences, so Pythia at layer
  4 is a replication with a rebuilt direction, not a rerun.)
- VALENCE: exp180's 20 positive words minus their 20 opposites.
- **RANDOM-WORD NULL**: N_NULL directions, 31 random words minus 31
  others, from the pool of all exp180 and exp185 pair words.

Steering, measures and statistic: exp188's, unchanged. Add strength x
direction at the site only, every non-start position; strength =
C times the mean unsteered norm at the site. STRAIGHT (PRIMARY),
WRITE_COS and the final-state norm, over the blocks after the site.
A(d, C) = measure with +C*d minus measure with −C*d, per sentence.
UP and VALENCE are run at C in {0.5, 1.0, 1.5}; the null at C = 1.0.

## RULE PARAMETERS (frozen)

  N_NULL = 100   SEED_NULL = 190   ALPHA = 0.05   N_PERM = 10000

## Decision rule

A cell HOLDS if, at C = 1.0 on STRAIGHT, A(UP) > 0 with sign-flip
p < ALPHA across sentences AND A(UP) is above the null's 95th
percentile.
- **LEAD_HOLDS**: at least 3 of the 4 cells hold.
- **LEAD_FAILS**: at most 1 holds.
- **MIXED**: exactly 2 hold.
Reported: the same for WRITE_COS and the final-state norm (if the
norm stands out wherever STRAIGHT does, the two have not come apart);
A(UP) at the three strengths (does it grow with the dose);
A(VALENCE) beside A(UP) throughout.

## Committed predictions

- P1 Pythia, layer 4 holds: **55%**
- P2 each of the other three cells holds: **25%** each
- P3 LEAD_HOLDS: **12%**; LEAD_FAILS: **58%**; MIXED: **30%**

## What I will NOT do

- No changing sites, strengths, lists or the rule after seeing
  numbers.
- No reading a HOLDS in which VALENCE does the same as evidence about
  UP in particular; it would be about positive-pole directions.

## Deviations and stubs

None at freeze.

## RESULT + GRADES (graded 00:35 IST, 03 Oct 2026)

Raw output: exp190_output.txt. Numbers: exp190_results.json; cells in
exp190_cells_<model>_L<site>.npz. Mac GPU. One launch failure before
any data (a shell quoting mistake; relaunched; no cell was affected).

### STRAIGHT at strength 1.0: A(UP) = +UP minus −UP

| cell | A(UP) | sentences positive | null 5th..95th (100 random-word directions) | percentile | holds? |
|---|---|---|---|---|---|
| Pythia, layer 4 | +0.0275 | 120 | −0.0293..+0.0258 | 95th | yes |
| Pythia, layer 8 | +0.0174 | 119 | −0.0228..+0.0223 | 89th | no |
| GPT-2, layer 4 | +0.0242 | 104 | −0.0877..+0.1199 | 72nd | no |
| GPT-2, layer 8 | −0.0151 | 13 | −0.0790..+0.0800 | 38th | no |

**VERDICT: LEAD_FAILS** (1 of 4 cells).

### Notes
- Pythia: the same sign at both layers, narrowly above the random
  directions at layer 4, below the line at layer 8.
- GPT-2: the random-word directions themselves spread four times as
  widely (±0.08..0.12 against ±0.02..0.03 in Pythia), and UP sits
  inside that spread at layer 4 and goes the other way at layer 8.
  So in GPT-2 random-word pushes do move the later layers'
  agreement, which they barely did in Pythia; the "mush is weak"
  description from exp188 is a Pythia fact, not a general one.
- The dose does not rise with strength in any cell.
- The final-state norm never stands out (30th–78th percentile), so
  the one thing that was consistent is that the path measure and the
  endpoint measure are not the same quantity.

### Standing
exp188's lean does not replicate. As a claim about UP it is closed.
exp193 (running) asks a different question with a different
statistic: how much any push lowers agreement, real axes against
concentrated and diffuse controls.

### Grades
- P1 Pythia, layer 4 holds (55%): HIT.
- P2 each other cell (25%): none occurred.
- P3 LEAD_FAILS (58%): HIT.
