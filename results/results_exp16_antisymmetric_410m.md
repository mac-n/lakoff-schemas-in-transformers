# exp16 — Antisymmetric decomposition on Pythia 410m

Same test as exp15 (which ran on 70m) on Pythia 410m's MLP-output SAEs.
If schemas-as-polar-opposites emerge with scale, SCHEMA_xdomain should be larger
relative to UP/DOWN/COMMON than at 70m.

**For comparison, exp15 at Pythia 70m (residual-stream SAE):**
  UP_xdomain ≈ 0.04, DOWN_xdomain ≈ 0.06, SCHEMA_xdomain ≈ 0.02 (SMALLER), COMMON_xdomain ≈ 0.06 (LARGEST)
  i.e. antisymmetric decomposition made things WORSE at 70m. UP and DOWN treated as separate change-dirs.

## Summary across layers (Pythia 410m)

  layer   UP_xdom  DOWN_xdom  SCHEMA_xdom  COMMON_xdom  SCHEMA_PC1  COMMON_PC1
      0   +0.0406    +0.0360      +0.0031      +0.0494       0.067       0.123
      1   +0.0292    +0.0365      +0.0049      +0.0407       0.086       0.124
      2   +0.0358    +0.0239      +0.0015      +0.0358       0.090       0.151
      3   +0.0382    +0.0298      +0.0032      +0.0416       0.090       0.130
      4   +0.0360    +0.0395      +0.0139      +0.0444       0.100       0.125
      5   +0.0355    +0.0367      +0.0048      +0.0444       0.093       0.118
      6   +0.0384    +0.0609      +0.0232      +0.0567       0.119       0.100
      7   +0.0371    +0.0451      +0.0144      +0.0477       0.113       0.088
      8   +0.0390    +0.0568      +0.0261      +0.0537       0.101       0.094
      9   +0.0448    +0.0539      +0.0182      +0.0573       0.103       0.096
     10   +0.0390    +0.0518      +0.0170      +0.0531       0.101       0.093
     11   +0.0447    +0.0533      +0.0186      +0.0570       0.093       0.096
     12   +0.0440    +0.0601      +0.0182      +0.0622       0.100       0.099
     13   +0.0438    +0.0639      +0.0247      +0.0657       0.134       0.089
     14   +0.0368    +0.0498      +0.0215      +0.0501       0.093       0.087
     15   +0.0364    +0.0458      +0.0116      +0.0496       0.110       0.091
     16   +0.0309    +0.0508      +0.0196      +0.0472       0.093       0.109
     17   +0.0475    +0.0920      +0.0359      +0.0858       0.128       0.103
     18   +0.0485    +0.1130      +0.0473      +0.1004       0.177       0.148
     19   +0.0326    +0.0563      +0.0185      +0.0572       0.125       0.093
     20   +0.0271    +0.0359      +0.0126      +0.0380       0.088       0.098
     21   +0.0251    +0.0652      +0.0178      +0.0586       0.113       0.145
     22   +0.0254    +0.0698      +0.0222      +0.0650       0.143       0.136
     23   +0.0310    +0.0880      +0.0232      +0.0814       0.165       0.123

## Verdicts

- If at any layer **SCHEMA_xdomain > UP_xdomain and > DOWN_xdomain**: schemas-as-polar-opposites EMERGED with scale. This is a real positive result for the scale hypothesis.
- If SCHEMA_xdomain still < UP/DOWN/COMMON at every layer: 410m doesn't have the polar structure either. Pattern persists.
- Watch the per-layer pattern: maybe schemas emerge at specific depths.
