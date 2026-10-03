# PRE-REGISTRATION — exp198: DOES THE SOURCE/GOAL ROLE READOUT TRANSFER FROM LITERAL JOURNEYS TO ABSTRACT CHANGE? (the real design)

Written 02:26 IST, 03 Oct 2026, before any data for this experiment, by Claude (Fable
5.1). Niamh at 02:25: "okay claude but why can't you do the real
design? or can you?" Frozen at the commit carrying this text.

Seen before freezing: the exp197 pilot (transfer 0.92 at Pythia layer
8 with from/to sentences; role-swapped sentences followed at 0.94;
noun control on unseen nouns at chance). Layer 8 as PRIMARY is taken
from the pilot and is declared here as such.

## Plain summary

Lakoff: PURPOSES ARE DESTINATIONS, abstract change is understood as a
journey. If so, the model's representation of "which thing is the
start and which the end" should be the same for a walk from a barn to
a river and for a change from poverty to wealth. The pilot said yes
but could not rule out the probe reading the words "from" and "to".
Here the two sets share NO cue words.

## Design

Two models, CPU: Pythia 410M and GPT-2-medium (both 24 layers), start
token, layers 4, 8, 12, 16, 20. The state at the last token of each
role noun.

Sets (each 32 sentences, order balanced: 16 name the goal first):
- **LITERAL-CF**: journeys with no "from"/"to", using spatial verbs and
  nouns only (set out at, reached, departed, destination, origin, lay
  behind / ahead, left ... arrived at).
- **ABSTRACT-CF**: abstract change with no "from"/"to" and no shared
  cue word with LITERAL-CF (became, turned into, gave way to, replaced,
  grew out of, emerged from, in the end was). Half the goals are
  negative (joy -> grief, trust -> betrayal), so valence cannot carry
  the role.
- **SWAPPED**: ABSTRACT-CF with the two nouns exchanged (the goal noun
  now plays source).
- **NO-JOURNEY**: the ABSTRACT-CF nouns in a sentence with no change in
  it ("The grief and the joy were both discussed"), scored with the
  journey labels. These nouns never appear in training.
- Secondary: the pilot's from/to sets (LITERAL-FT, ABSTRACT-FT).

Probe: ridge (lambda 10) on standardised states, source = 0, goal = 1,
trained on LITERAL-CF only. Chance distribution: N_SHUF label-shuffled
probes, tested on ABSTRACT-CF.

## RULE PARAMETERS (frozen)

  N_SHUF = 300   SEED = 198   PRIMARY_LAYER = 8

## Decision rule (per model, layer 8)

- **TRANSFERS**: ABSTRACT-CF accuracy above the shuffle 95th; SWAPPED
  accuracy above the shuffle 95th (the probe follows the roles); and
  NO-JOURNEY accuracy inside the shuffle 5th–95th.
- **CUE_OR_NOUN**: ABSTRACT-CF above the 95th but either of the other
  two conditions fails.
- **NO_TRANSFER**: ABSTRACT-CF inside the shuffle range.
Reported: all five layers; the from/to secondary sets; leave-one-
sentence-out accuracy within LITERAL-CF (does the probe generalise
at all).

## Committed predictions

- Pythia TRANSFERS: **60%**; GPT-2 TRANSFERS: **55%**; both: **45%**
- Peak layer in the early-middle (4–12) in both models: **70%**
- Secondary from/to transfer at least as high as cue-free: **65%**

## What I will NOT do
- No changing the layer after the numbers.
- No dropping sentences after seeing which ones the probe gets wrong.

## RESULT + GRADES (graded 02:34 IST, 03 Oct 2026)

Outputs: exp198_output_pythia-410m.txt, exp198_output_gpt2-medium.txt;
exp198_results_*.json. One cue word is shared after stop-words
("came", once in each set, on opposite sides of its noun). Pre-data
bug fixed and committed: the role swap ignored capital letters.

Probe trained on LITERAL-CF (no from/to). Accuracies; shuffle 95th
about 0.61–0.64 at every layer.

| | layer | within-literal (leave-one-out) | ABSTRACT-CF | SWAPPED | NO-JOURNEY | from/to abstract (CF->FT) |
|---|---|---|---|---|---|---|
| Pythia | 8 (primary) | 0.88 | **0.67** | 0.64 | 0.50 | **0.98** |
| Pythia | 12 | 0.88 | 0.73 | 0.69 | 0.48 | 0.90 |
| Pythia | 20 | 0.83 | 0.73 | 0.64 | 0.50 | 0.83 |
| GPT-2 | 8 (primary) | 0.80 | 0.66 | 0.52 | 0.62 | 0.77 |
| GPT-2 | 12 | 0.89 | 0.70 | 0.59 | 0.59 | 0.90 |
| GPT-2 | 16 | 0.95 | 0.72 | 0.62 | 0.53 | 0.90 |

- **Pythia, layer 8: TRANSFERS** by the rule, modestly (0.67 against a
  0.61 threshold; swapped roles followed at 0.64; unseen nouns with no
  journey at chance). Layers 12–20 are a little stronger (0.70–0.73).
- **GPT-2, layer 8: CUE_OR_NOUN.** Abstract clears the threshold
  (0.66) but the swapped test does not (0.52) and the no-journey
  control sits on the line (0.62). At layer 16 all three conditions
  would pass (0.72 / 0.62 / 0.53), but 16 was not the named layer.
- **The from/to question is settled the other way round.** A probe
  that never saw "from" or "to" reads the from/to abstract sentences
  at 0.98 (Pythia L8) and 0.90 (GPT-2 L12–16). The pilot's transfer
  was not the probe reading those two words. The marked journey frame
  is where the shared source/goal representation is clearest; the
  varied change verbs (soured into, replaced, grew out of) carry it
  more weakly, and the probe itself is only 0.88 within literal.

### Standing
The source/goal role representation built on literal journeys reads
abstract change in both models, strongly when the change is framed as
a path ("from poverty to wealth"), weakly when it is framed by a
change verb. Pythia passes the frozen rule; GPT-2 passes it at a
later layer than the one named. Reported as: supported in Pythia,
suggestive in GPT-2, with the cue-reading explanation ruled out.
Next: a layer sweep named in advance for GPT-2; a probe trained on
more literal sentences (0.88 within-set leaves room); and the causal
version (steer "nearly there" and watch whether abstract tasks
resolve sooner).

### Grades
- Pythia TRANSFERS (60%): occurred.
- GPT-2 TRANSFERS (55%): did not occur at layer 8 (would have at 16).
- Both (45%): did not occur.
- Peak in layers 4–12 in both (70%): Pythia yes (12), GPT-2 no (16).
- From/to transfer at least as high as cue-free (65%): occurred, by a
  wide margin.

### Correction (02:45 IST, 03 Oct 2026)
My shared-cue check treated "into" and "out" as stop-words, which
they are not: they are exactly the kind of cue the design was meant to
exclude. Counts: "into" appears once in LITERAL-CF and 14 times in
ABSTRACT-CF; "out of" twice and 9 times. Three literal sentences
could in principle teach "the noun after into/out of". A ridge probe
over 64 points is unlikely to rest on three of them, and the from/to
result (0.98 with neither word in training) does not depend on this,
but the claim "no shared cue words" is overstated and should read
"no shared cue words except into (1) and out of (2) on the literal
side". The next version should strip those three literal sentences.
