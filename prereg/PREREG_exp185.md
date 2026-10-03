# PRE-REGISTRATION — exp185: THE UP BUNDLE, OR THE DEFAULT MEMBER OF ANY PAIR?

Written 2026-10-02 19:55 IST, before any new code or model run for this
experiment, by Claude (session labelled Fable 5.1). Niamh at 19:52:
"2 is important. because almost everything maps onto UP but ur right
in/out doesn't." Frozen at the commit carrying this text.

Seen before freezing: exp180's results for the 30 polar pairs AS A
SET (mean, p). Their values pair by pair have not been looked at by
anyone. Tokenizer-only checks of the new pairs.

## Plain summary

exp180 found the first member's advantage in up / down pairs and
equally in other opposite pairs. Many of those other pairs are ones
Lakoff ties to UP (MORE IS UP, CONTROL IS UP, and so on). So "it is
general" might mean "it is the whole UP bundle" or "it is any pair
with a default member". This sorts pairs into the two kinds, before
looking at them one by one, and compares.

## The sorting (Claude's, by a stated rule; Niamh has not reviewed it)

Rule: a pair is BUNDLE if its contrast is a scale of quantity,
intensity, power, status, health or life, with the first member at
the "more" end. It is NO-LINK if its contrast is place, direction,
order or a two-way relation with no "more" end, and English has a
fixed order for the pair ("in and out", "give and take") that sets
which member comes first. Pairs that could go either way are left
out.

exp180's polar pairs:
- BUNDLE (16): big/small large/little long/short wide/narrow
  fast/slow hot/cold thick/thin strong/weak full/empty rich/poor
  loud/quiet bright/dark many/few more/less major/minor plus/minus
- NO-LINK (9): near/far front/back forward/backward inside/outside
  before/after in/out early/late open/closed wet/dry
- left out as ambiguous (5): old/young heavy/light hard/soft
  first/last on/off

New pairs (each word one token in both models, none used before):
- BUNDLE (16): huge/tiny maximum/minimum increase/decrease gain/loss
  grow/shrink add/subtract winner/loser superior/inferior
  senior/junior awake/asleep alive/dead healthy/sick active/passive
  master/slave majority/minority abundant/scarce
- NO-LINK (20): here/there this/that now/then come/go give/take
  push/pull buy/sell left/right north/south east/west male/female
  day/night land/sea start/finish begin/end send/receive read/write
  question/answer cause/effect input/output

Totals: BUNDLE 32 pairs, NO-LINK 29 pairs.

## Design

Measure, frames and analysis: exp180's, unchanged. The old pairs use
exp180's saved values. The new words are run in exp180's eight frames
with exp180's saved layer averages, so old and new values are on the
same footing. Pythia 410M and GPT-2-medium, each analysed separately.

Per set: mean first-minus-second difference, sign-flip p, and the
difference at equal word frequency with its bootstrap CI (exp180's
T_vert and T_freq). Between sets: BUNDLE mean minus NO-LINK mean,
label-shuffle p.

## RULE PARAMETERS (frozen)

  ALPHA = 0.05   N_PERM = 10000   N_BOOT = 2000

## Decision rule (per model)

"Holds" for a set: mean > 0 with p < ALPHA and the equal-frequency
CI entirely above 0.
- **UP_BUNDLE**: holds for BUNDLE, does not hold for NO-LINK, AND
  BUNDLE minus NO-LINK > 0 with p < ALPHA.
- **ANY_DEFAULT**: holds for both sets.
- **UNCLEAR**: anything else.
Across models: the label counts only if both models return it.

## Committed predictions

- P1 BUNDLE holds in both models: **70%**
- P2 UP_BUNDLE in both: **15%**; ANY_DEFAULT in both: **30%**;
  otherwise UNCLEAR or split: **55%**

## What I will NOT do

- No re-sorting pairs after seeing values. If Niamh sorts them
  differently, that is a second, labelled analysis.

## AMENDMENT (20:09 IST, 2026-10-02): Niamh's sorting — post-freeze, PRE-DATA

The run is in progress; no exp185 output has been read by anyone.
Nothing above is changed. Niamh, on seeing the lists, verbatim (20:09):

> "north/south i would expect to be up/down mapped in english. i
> think there may be something with before/after in lakoff on the
> up/down axis but i cant think what direction its in so maybe not"

Registered now as a second analysis, labelled NIAMH'S SORTING:
- north/south moves from NO-LINK to BUNDLE ("up north", "down
  south"; north first, so the first member is the UP one).
- before/after is left out as ambiguous. Lakoff and Johnson do list
  FORESEEABLE FUTURE EVENTS ARE UP ("what's coming up this week"),
  which would put AFTER on the UP side, the reverse of this pair's
  order; she was unsure of the direction. (The citation is from
  Claude's memory of Metaphors We Live By and has not been checked
  against the text.)
- Everything else as sorted above. Totals: BUNDLE 33, NO-LINK 27.

Same statistics and the same decision rule, computed from the saved
values after the run. Both sortings are reported. If they disagree,
that is the result.

## AMENDMENT 2 (20:12 IST, 2026-10-02): more of Niamh's sorting — still no output read

Niamh, verbatim (20:11):

> "forward is up. male is up because status is up. input/output are
> probably not up/down mapped in normal speech byt may well be for
> llms"

NIAMH'S SORTING now reads, relative to Claude's:
- to BUNDLE: north/south, forward/backward, male/female (first member
  UP in each);
- left out as ambiguous: before/after, input/output;
- everything else unchanged (front/back stays NO-LINK: she named
  "forward", not "front").
Totals: BUNDLE 35, NO-LINK 24. Computed from the saved values after
the run, same statistics and rule, reported beside Claude's sorting.

## AMENDMENT 3 (20:13 IST, 2026-10-02): front/back — still no output read

Niamh, verbatim (20:13):

> "not sure about front/back but it may well be that front is up too.
> upfront. front and centre. i think it's a little bit up."

In NIAMH'S SORTING front/back is left out as ambiguous ("a little bit
up" is neither a clear bundle member nor a clear no-link one).
Totals: BUNDLE 35, NO-LINK 23; left out: before/after, input/output,
front/back, plus Claude's five.

## RESULT + GRADES (graded 20:26 IST, 2026-10-02)

Raw output: exp185_output_<model>.txt. Numbers:
exp185_results_<model>.json (every pair's difference is saved there).
Mac GPU. No deviations. Mechanics checks PASS; the old word "big"
recomputes to exp180's saved value exactly, so old and new values are
on the same footing.

### Claude's sorting (the registered primary)

| model | set | pairs | first minus second | p | positive | at equal frequency [CI] | holds? |
|---|---|---|---|---|---|---|---|
| Pythia | BUNDLE | 32 | +0.132 | 0.018 | 22 | +0.195 [+0.071, +0.291] | yes |
| Pythia | NO-LINK | 29 | +0.224 | 0.0095 | 20 | +0.253 [+0.102, +0.386] | yes |
| GPT-2 | BUNDLE | 32 | +0.066 | 0.146 | 21 | +0.111 [+0.032, +0.176] | no |
| GPT-2 | NO-LINK | 29 | +0.299 | 0.0001 | 25 | +0.313 [+0.191, +0.425] | yes |

BUNDLE minus NO-LINK: Pythia −0.092 (p 0.34); GPT-2 −0.233 (p 0.003).
Verdicts: Pythia **ANY_DEFAULT**; GPT-2 **UNCLEAR** (the rule had no
label for "no-link pairs show it and bundle pairs do not"). No label
is shared by both models.

### Niamh's sorting (registered pre-data in amendments 1–3)

| model | set | pairs | first minus second | p | positive | holds? |
|---|---|---|---|---|---|---|
| Pythia | BUNDLE | 35 | +0.135 | 0.014 | 24 | yes |
| Pythia | NO-LINK | 23 | +0.191 | 0.060 | 15 | no (p just over) |
| GPT-2 | BUNDLE | 35 | +0.070 | 0.086 | 24 | no |
| GPT-2 | NO-LINK | 23 | +0.303 | 0.0007 | 19 | yes |

BUNDLE minus NO-LINK: Pythia −0.056 (p 0.58); GPT-2 −0.233 (p 0.004).
Verdicts: **UNCLEAR** in both. The two sortings do not disagree in
substance.

### What this says
- **It is not the UP bundle.** Under either sorting, in either model,
  the bundle pairs do not show the first-member advantage more than
  the no-link pairs. In GPT-2 the no-link pairs show it clearly MORE.
- By source (descriptive): the old bundle pairs (big / small, hot /
  cold) show it (+0.23 Pythia, +0.15 GPT-2); the NEW bundle pairs
  (huge / tiny, increase / decrease, winner / loser, superior /
  inferior) do not (+0.04, −0.02); the new no-link pairs (here /
  there, this / that, come / go, give / take) show it most (+0.27,
  +0.36).
- A reading of that pattern, untested: the advantage belongs to
  short, very common, default words that come first in a fixed
  everyday pairing, whatever the dimension. It does not belong to the
  "more" end of a scale as such.
- The pairs Niamh moved: north/south +0.31 / +0.14, male/female
  +0.52 / +0.19, forward/backward −0.33 / +0.02 (Pythia / GPT-2).

### Grades
- P1 BUNDLE holds in both models (70%): MISS (not in GPT-2).
- P2 UP_BUNDLE in both (15%): did not occur. ANY_DEFAULT in both
  (30%): did not occur. Otherwise (55%): HIT.
