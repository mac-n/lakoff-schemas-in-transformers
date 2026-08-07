# exp22 — multiple schema polar axes: orthogonality matrix

Building on exp21 (cos(UD, IO) ≈ −0.1). Now testing whether 4 Lakoff schemas
have mutually orthogonal constructed polar axes — strong evidence for
multiple separable embodied dimensions.

Triples per schema: UP-DOWN=25, IN-OUT=25, 
FORWARD-BACK=25, LIGHT-DARK=25, 
TUE-WED sham=5.

## Pairwise cosine matrices per layer (Pythia 70m)

### Layer 0

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | +0.032 | +0.020 | +0.115 | +0.001 |
| **IN-OUT** | +0.032 | +1.000 | +0.083 | -0.020 | +0.000 |
| **FORWARD-BACK** | +0.020 | +0.083 | +1.000 | +0.011 | +0.000 |
| **LIGHT-DARK** | +0.115 | -0.020 | +0.011 | +1.000 | -0.000 |
| **TUE-WED_sham** | +0.001 | +0.000 | +0.000 | -0.000 | +1.000 |

### Layer 1

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | -0.055 | +0.116 | +0.329 | +0.113 |
| **IN-OUT** | -0.055 | +1.000 | +0.026 | -0.117 | -0.065 |
| **FORWARD-BACK** | +0.116 | +0.026 | +1.000 | +0.119 | +0.043 |
| **LIGHT-DARK** | +0.329 | -0.117 | +0.119 | +1.000 | +0.157 |
| **TUE-WED_sham** | +0.113 | -0.065 | +0.043 | +0.157 | +1.000 |

### Layer 2

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | -0.090 | +0.299 | +0.517 | +0.494 |
| **IN-OUT** | -0.090 | +1.000 | -0.029 | -0.164 | -0.197 |
| **FORWARD-BACK** | +0.299 | -0.029 | +1.000 | +0.357 | +0.383 |
| **LIGHT-DARK** | +0.517 | -0.164 | +0.357 | +1.000 | +0.683 |
| **TUE-WED_sham** | +0.494 | -0.197 | +0.383 | +0.683 | +1.000 |

### Layer 3

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | -0.140 | +0.397 | +0.594 | +0.583 |
| **IN-OUT** | -0.140 | +1.000 | -0.068 | -0.218 | -0.257 |
| **FORWARD-BACK** | +0.397 | -0.068 | +1.000 | +0.439 | +0.445 |
| **LIGHT-DARK** | +0.594 | -0.218 | +0.439 | +1.000 | +0.805 |
| **TUE-WED_sham** | +0.583 | -0.257 | +0.445 | +0.805 | +1.000 |

### Layer 4

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | -0.151 | +0.346 | +0.519 | +0.509 |
| **IN-OUT** | -0.151 | +1.000 | -0.064 | -0.232 | -0.268 |
| **FORWARD-BACK** | +0.346 | -0.064 | +1.000 | +0.437 | +0.444 |
| **LIGHT-DARK** | +0.519 | -0.232 | +0.437 | +1.000 | +0.807 |
| **TUE-WED_sham** | +0.509 | -0.268 | +0.444 | +0.807 | +1.000 |

### Layer 5

| | UP-DOWN | IN-OUT | FORWARD-BACK | LIGHT-DARK | TUE-WED_sham |
|---|---|---|---|---|---|
| **UP-DOWN** | +1.000 | -0.055 | +0.314 | +0.331 | +0.136 |
| **IN-OUT** | -0.055 | +1.000 | +0.028 | -0.055 | -0.026 |
| **FORWARD-BACK** | +0.314 | +0.028 | +1.000 | +0.241 | -0.020 |
| **LIGHT-DARK** | +0.331 | -0.055 | +0.241 | +1.000 | +0.342 |
| **TUE-WED_sham** | +0.136 | -0.026 | -0.020 | +0.342 | +1.000 |

## Interpretation

Look at the off-diagonal entries of the 4-schema sub-matrix (UD, IO, FB, LD).

- All |cos| < 0.2: 4 mutually orthogonal embodied dimensions ✓ (Lakoff-supported)
- Some pairs at |cos| > 0.4: those schemas partially collapse into shared axis
- All cos > 0.5: schemas collapse to one generic polarity ruler

Tue-Wed sham row/column: should all be ≈ 0 if methodology working.
If sham aligns with any schema, sham still has hidden content.
