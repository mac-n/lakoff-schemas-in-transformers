# PRE-REGISTRATION — exp164: depth map + nonlinear control stack
Written 2026-06-12, before any code. (Claude; follows exp161's V1a.)

## Status of this experiment
DIAGNOSTIC / MAPPING, not confirmatory. No V-style gate. Any NEW
positive found here is a LEAD requiring its own fresh-prompt prereg —
it is never a result. Two committed predictions only; the rest is
declared exploration.

## Questions
Q1 Is Pythia's lone L12 entropy-coupling survivor (exp161: C2 = −0.21,
   CI [−0.29,−0.11], between nulls) nonlinear norm leakage?
   Mechanistic motivation, written BEFORE the test: attention entropy
   is thermostatted by query magnitude, ‖q‖² = LN(x)ᵀ W_QᵀW_Q LN(x) —
   a QUADRATIC readout of residual direction. exp161's control stack
   removes only LINEAR dependence on norm/d_norm-proj; a curved
   norm→entropy channel passes straight-line controls, and Pythia L12
   is exactly where cos(BALANCE, d_norm) peaks (+0.64).
Q2 (Niamh) Does Llama ground BALANCE in entropy SOMEWHERE — i.e. did
   exp161's decision layers (5/8/11) simply miss a peak, the way
   GPT-2's peak turned out to be shallow (L3: −0.44)? Llama's edge
   layers L2 (−0.156) and L13 (−0.187) both excluded zero.
Q3 Does GPT-2's confirmed coupling survive the upgraded (nonlinear)
   control stack — i.e. is exp161's V1a robust to the quadratic
   leakage channel?

## Design
- Models: pythia-410m (24 layers), gpt2-medium (24), Llama-3.2-1B (16).
  EVERY layer, not five.
- Data: exp161's frozen fresh prompts, imported from the exp161 module
  (no transcription — exp161's catch #4 lesson). DECLARED: this reuses
  exp161's prompt set; fine for mapping, disqualifying for confirmation.
- Per layer, axis = BALANCE ⊥ d_norm_ho (exp161 C2 axis, held-out
  d_norm per exp154c):
    C2  = partial r(axis-proj, entropy | pos, z_norm, z_dnorm)   [linear]
    C2q = as C2 plus z_norm², z_dnorm², z_norm·z_dnorm, rank(norm)
          [quadratic/nonlinear stack; covariates z-scored before
          squaring to limit collinearity]
- Prompt-cluster bootstrap 95% CI (1000) for C2 and C2q at every layer.
- Held-out d_norm carrier reported per layer; any layer with carrier
  < 0.50 is FLAGGED and excluded from lead-claims (not from the map).

## Committed predictions
- P1: Pythia L12 collapses under the nonlinear stack: |C2q| < 0.10.
  (If P1 hits → the quadratic channel is real → the nonlinear stack
  becomes STANDARD for all future entropy work, and exp161's published
  numbers get re-quoted under it.)
- P2: GPT-2's coupling is robust: C2q remains ≤ −0.15 wherever exp161
  found C2 ≤ −0.15 (L8, L12, L16), and the shallow peak (L3) survives
  at C2q ≤ −0.30. (Calibration lesson applied: this is a
  replication-with-controls bet, not a mechanism story — bet strong.)
- Llama (Q2): NO committed prediction — mapping is exploratory there.
  Soft note, on record: GPT-2's shallow peak makes a shallow Llama
  peak (L1–L4) plausible; I am not betting on it.

## Interpretation rules (written before results)
- D1 (P1 hits): L12 was leakage. Control-stack upgrade adopted.
- D2 (P1 misses, L12 survives C2q with CI excluding 0): real
  second-carrier candidate in Pythia — Pythia-only fresh-prompt prereg
  before it is believed.
- Llama lead criterion: a contiguous run of ≥ 2 layers with C2q
  negative AND 95% CI excluding 0 AND carrier ≥ 0.50. Anything less is
  noise-shaped and gets written down but not chased.
- GPT-2 flag: if C2q attenuates the exp161 decision layers below
  −0.15, V1a is REQUALIFIED (not revoked — requalified pending a
  fresh-prompt run under the new stack) and exp165 waits.

## What I will not do
- No promoting any single-layer cell anywhere to "finding".
- No treating Llama-lead layers found here as confirmed — they go to a
  new prereg with NEW prompts.
- No quietly preferring whichever stack flatters the story; both C2
  and C2q are reported for every layer of every model.
