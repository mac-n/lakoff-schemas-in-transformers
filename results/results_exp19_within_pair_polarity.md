# exp19 — within-pair polarity of UP and DOWN, across scale

Per-pair cosine similarity between UP_offset and DOWN_offset, taken from
the SAME baseline within the same (domain, pair_idx). Measures whether the
model represents UP and DOWN as polar opposites within a single domain.

- cos ≈ -1: perfect antipolarity (the 'ruler' finding)
- cos ≈  0: orthogonal (unrelated transformations)
- cos ≈ +1: same direction (shared 'change' component dominates)

## Pythia 70m (6 layers, residual-stream SAE, 32k features)

  layer      mean    median       min       max  n=25
      0   +0.5674   +0.6006   +0.1549   +0.9311
      1   +0.5942   +0.6004   +0.0929   +0.9394
      2   +0.6009   +0.6041   +0.1273   +0.9427
      3   +0.5429   +0.5765   +0.1109   +0.9179
      4   +0.4915   +0.4723   +0.0601   +0.9177
      5   +0.5308   +0.5282   +0.2152   +0.9492

Mean across all layers (70m): **+0.5546**

## Pythia 410m (24 layers, MLP-output SAE, 65k features)

  layer      mean    median       min       max  n=25
      0   +0.4774   +0.5049   +0.2475   +0.9318
      1   +0.5644   +0.5440   +0.2853   +0.8881
      2   +0.4673   +0.4412   +0.0851   +0.9306
      3   +0.4832   +0.4743   +0.1508   +0.9362
      4   +0.5204   +0.5023   +0.2081   +0.8615
      5   +0.5381   +0.5214   +0.2867   +0.8453
      6   +0.5831   +0.6253   +0.3233   +0.8935
      7   +0.5838   +0.6199   +0.3447   +0.8553
      8   +0.5808   +0.6093   +0.3656   +0.8626
      9   +0.5902   +0.6096   +0.3955   +0.8593
     10   +0.5843   +0.5817   +0.3708   +0.7995
     11   +0.5661   +0.5400   +0.3455   +0.8313
     12   +0.5541   +0.5357   +0.3044   +0.8228
     13   +0.5331   +0.5354   +0.3040   +0.8248
     14   +0.5572   +0.5797   +0.2581   +0.8532
     15   +0.5546   +0.5316   +0.2746   +0.8806
     16   +0.5603   +0.5800   +0.2419   +0.8602
     17   +0.5358   +0.5434   +0.2292   +0.9103
     18   +0.5554   +0.5272   +0.1852   +0.8817
     19   +0.5263   +0.5460   +0.2167   +0.9140
     20   +0.5272   +0.5441   +0.1945   +0.8810
     21   +0.5305   +0.5047   +0.2513   +0.8664
     22   +0.4884   +0.4607   +0.2116   +0.8523
     23   +0.5188   +0.4766   +0.2356   +0.8860

Mean across all layers (410m): **+0.5409**

## Scale comparison

  Pythia 70m:  mean within-pair cos(UP, DOWN) across all layers = +0.5546
  Pythia 410m: mean within-pair cos(UP, DOWN) across all layers = +0.5409
  Δ (410m − 70m) = -0.0138

**Interpretation:**
- If 410m mean is MORE NEGATIVE than 70m: ruler is emerging with scale.
- If 410m mean is MORE POSITIVE: polar opposite structure regresses (or noise).
- If similar: no scale effect at this range; would need bigger models.
