# exp24 — are IN and OUT opposite directions WITHIN UD-space?

Test: project each IN_offset and OUT_offset (from matched IO triples) onto
the constructed A_updown axis. Compare mean projections.

## Per-layer projections (Pythia 70m residual-stream SAE)

  layer    UP_proj   DOWN_proj    IN_proj    OUT_proj                                   verdict
      0    +0.6727     -1.3000    -0.2796     -0.3836                     both→DOWN (NOT polar)
      1    +0.4249     -1.6657    -0.9055     -0.7559                     both→DOWN (NOT polar)
      2    +0.2697     -1.7671    -1.2957     -1.0652                     both→DOWN (NOT polar)
      3    +0.1765     -2.0755    -1.8478     -1.4852                     both→DOWN (NOT polar)
      4    +0.3786     -2.1787    -1.5387     -1.0991                     both→DOWN (NOT polar)
      5    +0.3235     -2.6010    -1.6178     -1.4219                     both→DOWN (NOT polar)

## Interpretation

- If IN > 0 and OUT < 0 across layers: IN ↔ UP, OUT ↔ DOWN. The model has a
  shared polarity convention between containment and verticality.
- If both > 0 or both < 0: IN and OUT both lean the same way on UD-axis.
  No individual polarity, the IO axis is rotated but not polar in UD-space.
- If both ≈ 0: IO and UD are genuinely orthogonal at all levels.
