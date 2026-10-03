# PRE-REGISTRATION — exp193: WHICH PUSHES MAKE THE LATER LAYERS DISAGREE?

Written 00:23 IST, 03 Oct 2026, before any code or model run for this experiment, by
Claude (session labelled Fable 5.1). Niamh at 00:23: "do q up the
experiments just in case this is anything." Frozen at the commit
carrying this text.

Seen before freezing: exp188 and the first two cells of exp190 (the
up/down axis lowers later-layer agreement far more than random-word
averages do; DOWN more than UP; valence less than UP).

## Plain summary

Three questions about the "agreement of the later layers" measure:
1. Is the up/down axis special among real concept axes, or does every
   real axis disrupt agreement when pushed?
2. Is it meaning that matters, or just concentration? Averages of
   random words are weak by construction. Single-word directions and
   random word-pair directions are concentrated without being
   meaningful.
3. Is the agreement drop just another way of saying "the output
   moved", or does it carry something of its own? (And Niamh's
   deflationary candidate: does pushing simply raise the output's
   entropy?)

## Design

Pythia 410M, start token, the lab's 120 everyday sentences, steering
at layer SITE at every non-start position, strength 1.0 times the
mean unsteered norm at the site, both signs. Measures on the blocks
after the site: STRAIGHT (primary), WRITE_COS, final-state norm, as
exp188. Added in the same pass: the mean KL from the unsteered
next-token distribution to the steered one over all positions, and
the mean change in the next-token entropy.

Directions, all unit length, anisotropy and frequency directions
projected out, built at the site from exp175's cached clean word
states:
- **SCHEMA (9)**: the eight schema axes (the lab's lists as filtered
  to single tokens in exp175_stimuli.json, UP-DOWN from the spatial
  lists) and VALENCE.
- **SINGLE (N_CTRL)**: one random word's state minus the mean state.
- **PAIR (N_CTRL)**: one random word's state minus another's.
- **MUSH (N_CTRL)**: 31 random words minus 31 others (exp188's null).
- **GAUSS (N_CTRL)**: random unit vectors.
Random words come from exp175's analysis words not on the UP / DOWN
or BALANCE lists.

Per direction: drop = unsteered STRAIGHT minus the mean of STRAIGHT
under +d and −d (how much agreement falls, sign-averaged); asym =
STRAIGHT under +d minus under −d; and the same for KL and entropy.

## RULE PARAMETERS (frozen)

  SITE = 4   N_CTRL = 30   SEED = 193

## Decision rule

- Q1 **UP_STANDS_OUT**: UP-DOWN has the largest drop of the nine
  SCHEMA directions AND its drop is above the 95th percentile of the
  SINGLE and PAIR drops pooled. Otherwise **UP_NOT_SPECIAL**.
- Q2 **MEANING_MATTERS**: the mean drop over the nine SCHEMA
  directions is above the 95th percentile of the pooled SINGLE and
  PAIR drops. **CONCENTRATION_SUFFICIENT**: it is inside their range
  (between the 5th and 95th percentiles). **SCHEMA_WEAKER**: below
  the 5th.
- Q3 reported, not labelled: the correlation across all directions
  between drop and output KL, and between drop and entropy change.
  If the drop–KL correlation is above 0.8, the agreement drop is
  read as a proxy for output change. Niamh's entropy candidate
  predicts that directions which raise entropy more disrupt more;
  the drop–entropy correlation tests it.

## Committed predictions

- Q1 UP_STANDS_OUT: **15%**
- Q2 MEANING_MATTERS **35%**; CONCENTRATION_SUFFICIENT **50%**;
  SCHEMA_WEAKER **15%**
- Q3 drop–KL correlation above 0.8: **50%**

## What I will NOT do

- No changing the site, strength, families or thresholds after
  seeing numbers.
- No calling anything new before a literature check.

## RESULT + GRADES (graded 01:04 IST, 03 Oct 2026)

Raw output: exp193_output.txt. Numbers: exp193_results.json; cells in
exp193_cells.npz. Pythia 410M, layer 4, strength 1.0 x site norm,
129 directions, 120 sentences. Unsteered agreement 0.2220.

### Drop in later-layer agreement (sign-averaged), with output change

| direction | drop | output KL | entropy change |
|---|---|---|---|
| **UP-DOWN** | **+0.0377** | 2.79 | +0.71 |
| VALENCE | +0.0176 | 1.62 | +1.22 |
| FORWARD-BACK | +0.0158 | 2.00 | +1.15 |
| IN-OUT | +0.0140 | 2.14 | +1.01 |
| PATH-MOTION | +0.0069 | 2.42 | +1.10 |
| DIFFICULTY-BURDEN | +0.0062 | 2.02 | +1.23 |
| LIGHT-DARK | +0.0055 | 1.82 | +1.19 |
| FORCE | +0.0053 | 2.14 | +1.27 |
| BALANCE | +0.0050 | 1.67 | +1.11 |
| SINGLE (30), median [5%, 95%] | −0.0012 [−0.0115, +0.0117] | 2.33 | |
| PAIR (30) | −0.0016 [−0.0076, +0.0114] | 2.24 | |
| MUSH (30) | −0.0012 [−0.0069, +0.0105] | 1.97 | |
| GAUSS (30) | −0.0008 [−0.0092, +0.0069] | 1.25 | |

- **Q1: UP_STANDS_OUT.** UP-DOWN has the largest drop of the nine real
  axes, by a factor of two over the next (VALENCE), and sits above
  the 95th percentile of the concentrated controls (+0.0115) by
  three times.
- **Q2: MEANING_MATTERS, narrowly.** The mean drop over the nine
  real axes (+0.0127) is just above the concentrated controls' 95th
  percentile (+0.0115), and that mean is carried by UP-DOWN; the
  median real axis (+0.007) would be inside the control range.
- **Q3: the drop is not output change in disguise.** Across all 129
  directions, corr(drop, output KL) = +0.11. Single-word pushes move
  the output as much as the real axes do (KL 2.3 against 1.6–2.8) and
  leave the later layers' agreement untouched. corr(drop, entropy
  change) = −0.41: pushes that raise the output's entropy more
  disrupt agreement LESS, the reverse of Niamh's deflationary
  candidate.

### Standing
- By the frozen rules this is a positive result, on the question
  "does the measure separate real directions from concentrated fake
  ones": in Pythia 410M at layer 4, pushing along the UP axis makes
  the later layers disagree far more than any control push of the
  same size, while moving the output no more than they do.
- It is one model and one layer. exp190 showed that in GPT-2
  random-word (MUSH) pushes lower agreement as much as UP does, so
  this picture is NOT expected to hold there as it stands. The next
  test is this exact design in GPT-2 (with SINGLE and PAIR controls),
  and at a second Pythia layer. Until then: a lead that has survived
  one proper set of controls, not a finding.
- Why UP in particular would be the direction Pythia's later layers
  respond to most is unknown.

### Grades
- Q1 UP_STANDS_OUT (15%): occurred.
- Q2 MEANING_MATTERS (35%): occurred, narrowly.
- Q3 drop–KL correlation above 0.8 (50%): did not occur (+0.11).
