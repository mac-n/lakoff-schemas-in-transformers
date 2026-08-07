# exp14 — Schemas as directions: offset-vector alignment in SAE feature space

**The Lakoffian claim restated:** image schemas are invariant relational
transformations across domains. UP isn't a feature; UP is the offset that
takes a baseline state to its upward-shifted counterpart, and that offset
is approximately the same vector across temperature, mood, prices, status,
health. (Like king−man+woman=queen: 'royal' is a direction, not a feature.)

Pairs: 25 UP (5 domains × 5 pairs), 25 DOWN (5 domains × 5 pairs),
15 BEVERAGE sham (3 categories × 5 pairs, taxonomic), 10 NULL.
Aggregation: MAX SAE activation per feature across token positions.
Offset Δ = activation(transformed) − activation(baseline) per pair.

## Summary across layers

  layer  UP_within DOWN_within BEV_within NULL_within UP_cross_domain  UP_PC1  NULL_PC1  BEV_PC1  UP_vs_DOWN
      0     0.0407      0.0409     0.1570     -0.0261          0.0432   0.133     0.195    0.227     +0.0602
      1     0.0450      0.0549     0.1830     -0.0315          0.0438   0.149     0.305    0.195     +0.0655
      2     0.0446      0.0721     0.1621     -0.0191          0.0378   0.249     0.492    0.755     +0.0698
      3     0.0352      0.0768     0.1340     -0.0287          0.0246   0.322     0.553    0.846     +0.0618
      4     0.0361      0.0678     0.1028     -0.0307          0.0260   0.249     0.519    0.692     +0.0559
      5     0.0416      0.0647     0.1012     -0.0076          0.0375   0.132     0.244    0.366     +0.0647

## Interpretation guide

- **UP_within > NULL_within**: UP pairs align more than random pairs → schema-as-direction exists.
- **UP_within > BEV_within**: UP (embodied schema) aligns more than BEVERAGE (taxonomy) → schemas ≠ taxonomic categories.
- **UP_cross_domain ≈ UP_within**: alignment is *cross-domain* (the Lakoffian invariance), not just within-domain.
- **UP_PC1 >> NULL_PC1**: UP offsets have a dominant principal component (a 'UP direction') that random offsets lack.
- **UP_vs_DOWN < 0**: UP and DOWN are antialigned (DOWN ≈ −UP in feature space).
- **Layer pattern**: where do schemas-as-directions live?
