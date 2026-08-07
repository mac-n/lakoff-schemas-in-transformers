# exp21 — are schema polar axes orthogonal?

Constructed polar axes by fiat (mean of within-pair UP-DOWN offsets across pairs)
and checked whether different schemas' axes align or are orthogonal.

**Predictions:**
- cos(A_updown, A_inout) ≈ +1: schemas collapse to one generic polarity ruler
- cos(A_updown, A_inout) ≈  0: schemas are separable polar dimensions (Lakoff-consistent)
- cos(A_x, A_beverage) ≈ 0: sham (coffee vs tea) shouldn't align with real polar axes

## Summary across layers (Pythia 70m)

  layer    cos(UD,IO)    cos(UD,BEV)    cos(IO,BEV)    mean cos(UDpair,IOpair)
      0       +0.0323        -0.0190        +0.0054                    +0.0015
      1       -0.0549        +0.0318        -0.0212                    -0.0040
      2       -0.0904        +0.2553        -0.0917                    -0.0032
      3       -0.1404        +0.3530        -0.1301                    -0.0026
      4       -0.1509        +0.2114        -0.1062                    -0.0062
      5       -0.0548        +0.0247        +0.0061                    -0.0047

## Interpretation

Look at the cos(UD, IO) column. If consistently:
- > 0.7: schemas have collapsed to one polarity. The model treats UP↔DOWN and IN↔OUT as the same dimension.
- 0.2-0.6: partially aligned. Some shared valence component, some distinctness.
- < 0.2: schemas are separable polar dimensions. UP-DOWN and IN-OUT live on different axes.
- negative: even more separated — they have inverted polar structure.

The sham columns should hover near 0 if BEVERAGE is doing its job as a non-polar control.
