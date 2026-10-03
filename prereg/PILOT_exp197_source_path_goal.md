# PILOT exp197 — does a SOURCE/GOAL role readout transfer from literal journeys to abstract ones?

Written 01:36 IST, 03 Oct 2026, before any data, by Claude (Fable 5.1). A PILOT, not a
full prereg: small n, written in twenty minutes at Niamh's ask.

Niamh's words (01:33): "i think the SOURCE PATH GOAL schema would be
especially intrersting to researchers. but no idea how to investigate
it." (01:35): "can we test anything on it quick?"

## Design
Pythia 410M, CPU, start token. For each sentence, the residual state
at the last token of the SOURCE noun and of the GOAL noun, at layers
4, 8, 12, 16, 20. A linear probe (ridge, sign) trained on the 32
LITERAL journey sentences (source = 0, goal = 1), tested untrained on
the 32 METAPHORICAL ones (poverty -> wealth, premise -> conclusion).
Word order is balanced in every set (half the sentences name the goal
first), so position cannot carry the transfer. 8 sentences in each
set use no "from"/"to" at all (began at / ended at; began in / ended
in), reported separately as the cue-free subset.

Controls: (a) 300 label-shuffled probes give the chance distribution
of transfer accuracy; (b) a same-noun control set with no journey
("The barn and the river were both quiet"), scored with the journey
labels: if the probe is reading noun identity rather than role, it
scores high here too.

## Rule (primary: layer 12, all 32 metaphorical sentences, 64 points)
TRANSFERS if accuracy is above the label-shuffle 95th percentile AND
the same-noun control is inside the shuffle range. NOUN_LEAK if the
control is also above the 95th. NO_TRANSFER otherwise. Cue-free
subset and other layers reported.

## Odds
TRANSFERS 55%; NOUN_LEAK 20%; NO_TRANSFER 25%.

## Known limit
Both sets share "from"/"to" in 24 of 32 sentences; the cue-free
subset is small (16 points). A transfer here could be cue reading.
This is a pilot to see if there is anything worth a proper design.

## RESULT (graded 01:41 IST, 03 Oct 2026)

Output: exp197_output.txt, exp197_results.json.

| layer | fit on literal | transfer to metaphor | cue-free subset (16 pts) | noun control | shuffle 95th |
|---|---|---|---|---|---|
| 4 | 1.00 | 0.70 | 0.62 | 0.77 | 0.69 |
| 8 | 1.00 | **0.92** | 0.81 | 0.76 | 0.64 |
| 12 (primary) | 1.00 | 0.80 | 0.69 | 0.73 | 0.62 |
| 16 | 1.00 | 0.69 | 0.62 | 0.73 | 0.61 |
| 20 | 1.00 | 0.59 | 0.50 | 0.72 | 0.61 |

**By the frozen rule: NOUN_LEAK** (control 0.73 > shuffle 95th 0.62).

### Post-hoc (scratchpad script; same probes)
The rule was badly built: the noun control reused the TRAINING nouns,
and a 64-point probe in 1024 dimensions memorises them (fit 1.00).
Splitting the control by which nouns it reuses, and adding the
metaphor sentences with their roles swapped ("He went from wealth to
poverty": wealth is now the source):

| layer | metaphor | control, literal nouns | control, metaphor nouns | roles swapped |
|---|---|---|---|---|
| 4 | 0.70 | 1.00 | 0.53 | 0.67 |
| 8 | 0.92 | 0.95 | 0.56 | 0.94 |
| 12 | 0.80 | 0.91 | 0.55 | 0.75 |
| 16 | 0.69 | 0.95 | 0.52 | 0.69 |
| 20 | 0.59 | 0.92 | 0.52 | 0.54 |

On nouns the probe never saw, the no-journey control is at chance,
and when the roles are swapped the probe follows the roles (0.94 at
layer 8), not the nouns and not their valence. So the leak in the
frozen verdict is memorisation of training nouns, not what the
transfer is made of.

### Standing
A pilot that says the question is worth a real design. What it does
NOT rule out: the probe reading "the noun after from" (24 of 32
sentences carry from/to). The cue-free subset (0.81 at layer 8, 13 of
16) points the right way but is too small to settle it. A proper
exp197 needs: held-out nouns in the control by design, 30+ cue-free
items per set, a preposition-only control, and a second model.

### Grades
TRANSFERS 55% / NOUN_LEAK 20% / NO_TRANSFER 25%: NOUN_LEAK occurred
by the rule as written; the rule was the fault, recorded as such.
