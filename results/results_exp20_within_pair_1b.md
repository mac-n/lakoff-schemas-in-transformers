# exp20 — within-pair cos(UP, DOWN) at Pythia 1B layer 11

Single layer of SAE coverage at 1B (timhua/pythia1b_deduped_saes covers only L11).
L11 of 16 is ~69% through model — comparable depth to 70m L4 and 410m L17.

**Within-pair cos(UP, DOWN) at Pythia 1B layer 11:**
- mean   = +0.5320
- median = +0.5183
- range  = [+0.2649, +0.8959]

## Cross-scale comparison

  70m  L4 (67% through): +0.4915  (exp19)
  410m L17-18 (71-75%):  +0.5358 to +0.5554  (exp19)
  1B   L11 (69% through): +0.5320  (this experiment)

  70m  all 6 layers:  +0.5546 (exp19)
  410m all 24 layers: +0.5409 (exp19)
  1B   layer 11 only: +0.5320 (this experiment)

## Per-triple cosines

        domain pair   cos(UP, DOWN)
   temperature    0         +0.8391
   temperature    1         +0.6337
   temperature    2         +0.6121
   temperature    3         +0.6678
   temperature    4         +0.2766
          mood    0         +0.3988
          mood    1         +0.5402
          mood    2         +0.4339
          mood    3         +0.4784
          mood    4         +0.5183
      quantity    0         +0.7587
      quantity    1         +0.8959
      quantity    2         +0.4407
      quantity    3         +0.5317
      quantity    4         +0.5549
        status    0         +0.4854
        status    1         +0.4280
        status    2         +0.2649
        status    3         +0.4660
        status    4         +0.4707
        health    0         +0.4883
        health    1         +0.7585
        health    2         +0.5237
        health    3         +0.5312
        health    4         +0.3037
