# The start-token bug (found 2026-10-02, 17:20 IST)

Found by Claude (session labelled Fable 5.1) while debugging exp175
with Niamh. Everything under "Checked" was run today and is on disk.
Everything under "Not checked" is exactly that.

## What it is

In this project's TransformerLens build, Pythia has
`default_prepend_bos = False`. `model.to_tokens("the")` returns one
token, with no start token in front. 82 experiment scripts call
`to_tokens` and none of them sets `prepend_bos`.

Two consequences in the bare-word protocol (exp154, exp170, exp174 and
anything built the same way):

1. **A word that is a single token sits at sequence position 0.** In
   Pythia 410M the state at position 0 has a norm of about 990 at
   layers 8–16, against about 22 for a token at any later position.
2. **"Token count = shape[1] - 1" is off by one.** The group those
   scripts call "single-token" is really the two-token words. The true
   one-token words are counted as zero.

## Checked today

`diag_bos_position0.py` (cached states, no model) and
`diag_bos_rerun.py` (fresh run, with and without a start token).
Neither is preregistered. Both are diagnostics.

- Norm is a position-0 flag. corr(norm, word-is-one-token) = +0.93 at
  L4 and +1.000 at L8–L20, over exp170's 489 words (222 one-token,
  219 two-token, 48 longer).
- exp170's norm direction is that flag: cos(d_norm_orig, d_flag) =
  +0.994..+1.000.
- BALANCE's poles split on it more than any other schema: 73% of its
  positive-pole words are one token, 7% of its negative-pole words
  (unbalanced, unstable and the like are multi-token). The next
  largest gap among the eight schemas is 21 points.
- **Finding 5, Pythia norm arm.** Without a start token the rerun
  reproduces exp170 exactly: cos(BALANCE, d_norm_orig) = +0.784,
  +0.729, +0.701, +0.718, +0.729. With a start token: +0.395, +0.332,
  −0.049, +0.030, −0.117. Gone at layers 12–20. The +0.33..+0.40 left
  at layers 4–8 has not been controlled for token count.
- This also explains exp170's two disagreeing controls. Norm is a
  step (one token against the rest), not a slope, so regressing token
  count out linearly left most of the step in. The other control
  (constant token count) removed it, and read about zero.
- **The inflectional sink on BALANCE** (exp170's "infl mean sink",
  the Finding 4 quantity). Without a start token: −0.414, −0.320,
  −0.285, −0.288, −0.280 (exp170 exactly). With one: −0.181, −0.209,
  −0.177, −0.153, −0.137. About half the size, still negative at
  every layer. No null was computed for the with-start-token values.
  Base forms are 87–100% one-token; inflected forms 0–47% (plurals
  75%), so the old protocol mostly compared a position-0 state with a
  later-position state.

## Not checked

- Whether the sink that remains with a start token beats a null, and
  whether the transformer-versus-static dissociation (exp138 / 150 /
  153, replicated in exp171) survives. exp171's held-out replication
  used the same protocol, so it would replicate an artifact as
  faithfully as a real effect.
- Finding 3 (exp123, the layer-stable relational system). Same
  protocol; schema axes mix position-0 and later-position states.
- Findings 1 and 2 (steering). Contrast vectors were built from bare
  words; the behavioural results are what they are, but the vectors
  were built from mixed-position states.
- The GPT-2 / Llama entropy arm (exp161 / 164 / 166 / 168). Those
  models may prepend a start token by default where Pythia does not.
  If so, "Pythia grounds BALANCE in norm, the others in entropy" is
  partly a comparison of with and without a start token. Not looked
  at.
- Larger Pythia sizes and other layers.

## exp174 and exp175

- exp174 ran with the bug (bare words without a start token; every
  sentence prefixed with the token "a"). ERRATUM in PREREG_exp174.md.
- exp175's script had the same call and was fixed before any data
  (PREREG_exp175.md, POST-FREEZE LOG). Its words sit mid-sentence, so
  its word states were never at position 0.

## What would settle it

One preregistered rerun of the Finding 3, 4 and 5 keystones with a
start token (or with every word inside a neutral frame, as exp175
does), with nulls. The blog and glassnest.ai/embodied state the norm
grounding and r = +0.86..+0.97; those should not be repeated until
that rerun is in.

## Update 17:24 IST: the GPT-2 / Llama entropy arm

Checked from the library config and by reading the code. Not rerun.

- `default_prepend_bos`: pythia-410m False, gpt2-medium True,
  meta-llama/Llama-3.2-1B True. So GPT-2 and Llama DID get a start
  token in every experiment; only Pythia did not.
- The entropy experiments (exp161 / 164 / 166 / 168) work on tokens
  inside sentences, and `attn_entropy_per_query` and `collect_cloud`
  both skip sequence position 0. With a start token present, position
  0 is the start token, so no word state sits there.
- Their schema axes come from bare words through
  `collect_residuals`, whose docstring says "BOS-safe (BOS, if any,
  sits at position 0)". True for GPT-2 and Llama. False for Pythia.

So the start-token bug does not reach the GPT-2 / Llama entropy
results. What it does reach is the cross-model comparison: Pythia's
side of exp166 (entropy-null) and its norm result were both measured
without a start token, the other two models with one. Pythia's
carrier is therefore unknown, not "norm". Rerunning Pythia's entropy
test with a start token would say whether it joins the other two.
