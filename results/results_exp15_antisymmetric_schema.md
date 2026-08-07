# exp15 — Antisymmetric decomposition: isolating the schema axis

**Niamh's insight:** exp14 found UP and DOWN positively correlated (+0.06).
If `UP = A·schema_axis + B·common_axis` and `DOWN = −A·schema_axis + B·common_axis`,
then `UP − DOWN = 2A·schema_axis` (the antisymmetric part isolates the schema)
and `UP + DOWN = 2B·common_axis` (the symmetric part isolates the confound).

Same matched-pair design as exp14 but with shared baseline per (domain, pair_idx).

## Summary across layers

  layer  UP_xdomain  DOWN_xdomain  SCHEMA_xdomain  COMMON_xdomain  NULL_SCHEMA_x
      0     +0.0432       +0.0433         +0.0045         +0.0512        -0.0449
      1     +0.0438       +0.0565         +0.0113         +0.0583        -0.0524
      2     +0.0378       +0.0732         +0.0241         +0.0625        -0.0452
      3     +0.0246       +0.0758         +0.0190         +0.0579        -0.0355
      4     +0.0260       +0.0658         +0.0212         +0.0536        -0.0519
      5     +0.0375       +0.0595         +0.0184         +0.0592        -0.0449

PC1 variance ratio (a single dominant direction would be ~0.5+):
  layer   UP_PC1   DOWN_PC1  SCHEMA_PC1  COMMON_PC1  NULL_SCHEMA_PC1
      0    0.133      0.132       0.114       0.152            0.280
      1    0.149      0.133       0.118       0.163            0.282
      2    0.249      0.146       0.234       0.193            0.413
      3    0.322      0.167       0.294       0.227            0.548
      4    0.249      0.140       0.211       0.176            0.530
      5    0.132      0.133       0.114       0.127            0.401

## Verdicts

- **SCHEMA_xdomain >> UP_xdomain and >> DOWN_xdomain**: the schema axis exists, was just hidden by common-axis confound. THE LOAD-BEARING TEST.
- **SCHEMA_xdomain ≈ UP_xdomain ≈ DOWN_xdomain**: no improvement from decomposition; schemas are weak everywhere.
- **COMMON_xdomain > SCHEMA_xdomain**: the dominant shared structure across UP/DOWN pairs is the common axis (valence/change), not the schema.
- **SCHEMA_PC1 >> NULL_SCHEMA_PC1**: a single dominant schema direction exists.
- **NULL_SCHEMA_xdomain ≈ 0**: antisymmetrization of arbitrary triples gives noise (sanity check).
