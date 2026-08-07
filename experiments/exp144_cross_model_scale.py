"""
exp144_cross_model_scale.py — do the substrate findings (exp140
extent-on-MAG, exp141 BALANCE-as-norm-deviation) hold across Pythia
model sizes?

Models tested:
- pythia-70m   (6 layers, d_model=512)
- pythia-160m  (12 layers, d_model=768)
- pythia-410m  (24 layers, d_model=1024)  ← baseline (already known)
- pythia-1.4b  (24 layers, d_model=2048)

For each model size, compute:

TEST A (extent-on-MAG, from exp140):
  At a middle layer (~half-depth), build clean MAG and DIR axes,
  project test words {deep, deeper, deepest, abyss, tall, taller,
  tallest}. Compute mean |projection| on each. Winner = MAG or DIR.
  Substrate-native claim holds if MAG dominates across model sizes.

TEST B (BALANCE-as-norm, from exp141 TEST 2):
  At several layers, build clean BALANCE axis. For each suffix family
  (-ER, -EST, -ING, -ED, un-, re-), compute per-pair correlation
  r(Δ‖residual‖, BALANCE projection). Substrate-architectural claim
  holds if correlations are high (>0.7) across model sizes.

Outputs: comparison table across model sizes for both tests.
"""

import numpy as np
import torch
import gc
from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML

device = "mps"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b"]


# ============================================================================
# Word lists — same as exp140 / exp141
# ============================================================================

MAGNITUDE_BIG   = ["huge","big","large","enormous","vast","massive","gigantic"]
MAGNITUDE_SMALL = ["tiny","small","little","minute","microscopic","miniature"]

DIRECTIONAL_UP   = ["above","over","atop","upward","overhead","upper"]
DIRECTIONAL_DOWN = ["below","under","underneath","downward","beneath","lower"]

TEST_WORDS_EXTENT = ["deep","deeper","deepest","abyss","tall","taller","tallest"]

SUFFIX_PAIRS = {
    "ER_comparative": [("big","bigger"),("small","smaller"),("tall","taller"),
        ("high","higher"),("low","lower"),("deep","deeper"),("wide","wider"),
        ("fast","faster"),("slow","slower"),("old","older"),("new","newer"),
        ("hot","hotter"),("cold","colder")],
    "EST_superlative": [("big","biggest"),("small","smallest"),("tall","tallest"),
        ("high","highest"),("low","lowest"),("deep","deepest"),("old","oldest"),
        ("hot","hottest"),("cold","coldest")],
    "ING_progressive": [("walk","walking"),("run","running"),("jump","jumping"),
        ("sit","sitting"),("stand","standing"),("swim","swimming"),
        ("talk","talking"),("sing","singing"),("dance","dancing"),
        ("play","playing"),("work","working"),("read","reading")],
    "ED_past": [("walk","walked"),("jump","jumped"),("look","looked"),
        ("talk","talked"),("play","played"),("work","worked"),("ask","asked"),
        ("call","called"),("move","moved"),("stop","stopped"),("start","started")],
    "UN_negation": [("happy","unhappy"),("kind","unkind"),("healthy","unhealthy"),
        ("safe","unsafe"),("clear","unclear"),("clean","unclean"),("fair","unfair"),
        ("certain","uncertain"),("known","unknown"),("seen","unseen")],
    "RE_repetition": [("do","redo"),("make","remake"),("build","rebuild"),
        ("write","rewrite"),("read","reread"),("start","restart"),
        ("create","recreate"),("paint","repaint")],
}

COMMON = ["the","of","and","to","in","is","it","you","that","he","was","for",
          "on","are","with","as","his","they","at","be"]
RARE = ["serendipity","ostracize","perspicacity","obfuscate","sycophant"]


# ============================================================================
# Per-model test
# ============================================================================

def run_for_model(model_name):
    print(f"\n{'='*78}\nMODEL: {model_name}\n{'='*78}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    N_LAYERS = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  n_layers={N_LAYERS}, d_model={d_model}")

    # Pick "middle layer" for TEST A and a range for TEST B
    L_mid = N_LAYERS // 2
    BALANCE_LAYERS = sorted({
        max(1, N_LAYERS // 6),
        N_LAYERS // 3,
        L_mid,
        2 * N_LAYERS // 3,
        max(1, N_LAYERS - 2),
    })
    LAYERS_USED = sorted(set([L_mid] + BALANCE_LAYERS))
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS_USED]

    # Collect words needed
    all_words = set(COMMON + RARE + TEST_WORDS_EXTENT)
    for words in [MAGNITUDE_BIG, MAGNITUDE_SMALL, DIRECTIONAL_UP, DIRECTIONAL_DOWN]:
        all_words.update(words)
    for pairs in SUFFIX_PAIRS.values():
        for b, i in pairs:
            all_words.add(b); all_words.add(i)
    # BALANCE schema anchors
    for p, n in LAKOFF_SCHEMAS_MML["BALANCE"]:
        all_words.add(p); all_words.add(n)
    all_words = sorted(all_words)
    print(f"  collecting residuals for {len(all_words)} words at {LAYERS_USED}...")

    residuals = {}
    for k, w in enumerate(all_words):
        toks = model.to_tokens(w)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_names)
        residuals[w] = {L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].cpu().numpy()
                        for L in LAYERS_USED}

    # Anisotropy per layer
    anisotropy_dirs = {}
    for L in LAYERS_USED:
        all_r = np.stack([residuals[w][L] for w in all_words], axis=0)
        m = all_r.mean(axis=0)
        anisotropy_dirs[L] = m / np.linalg.norm(m)

    def mean_acts(words, layer):
        return np.mean([residuals[w][layer] for w in words], axis=0)

    def strip_aniso_freq(direction, layer):
        freq_raw = mean_acts(COMMON, layer) - mean_acts(RARE, layer)
        freq = freq_raw / np.linalg.norm(freq_raw)
        aniso = anisotropy_dirs[layer]
        direction = direction - (direction @ aniso) * aniso
        freq_orth = freq - (freq @ aniso) * aniso
        freq_orth = freq_orth / np.linalg.norm(freq_orth)
        direction = direction - (direction @ freq_orth) * freq_orth
        return direction / np.linalg.norm(direction)

    def build_axis_clean(pos, neg, layer):
        raw = mean_acts(pos, layer) - mean_acts(neg, layer)
        raw = raw / np.linalg.norm(raw)
        return strip_aniso_freq(raw, layer)

    def build_schema_clean(name, layer):
        pairs = LAKOFF_SCHEMAS_MML[name]
        pos = sorted(set(p[0] for p in pairs))
        neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos, layer) - mean_acts(neg, layer)
        raw = raw / np.linalg.norm(raw)
        return strip_aniso_freq(raw, layer)

    # ----- TEST A: extent-on-MAG at L_mid -----
    print(f"\n  TEST A — extent words on MAG vs DIR axis at L{L_mid}:")
    mag_axis = build_axis_clean(MAGNITUDE_BIG, MAGNITUDE_SMALL, L_mid)
    dir_axis = build_axis_clean(DIRECTIONAL_UP, DIRECTIONAL_DOWN, L_mid)
    test_a_results = []
    print(f"  {'word':<12}  {'on MAG':>10}  {'on DIR':>10}  {'|MAG|>|DIR|':>12}")
    for w in TEST_WORDS_EXTENT:
        rw = residuals[w][L_mid]
        rw_unit = rw / np.linalg.norm(rw)
        pm = float(rw_unit @ mag_axis); pd = float(rw_unit @ dir_axis)
        winner = "MAG" if abs(pm) > abs(pd) else "DIR"
        test_a_results.append({"word": w, "MAG": pm, "DIR": pd, "winner": winner})
        print(f"  {w:<12}  {pm:>+10.3f}  {pd:>+10.3f}  {winner:>12}")
    n_mag_wins = sum(1 for r in test_a_results if r["winner"] == "MAG")
    mean_abs_mag = np.mean([abs(r["MAG"]) for r in test_a_results])
    mean_abs_dir = np.mean([abs(r["DIR"]) for r in test_a_results])
    test_a_summary = {
        "n_mag_wins": n_mag_wins, "n_total": len(test_a_results),
        "mean_abs_MAG": mean_abs_mag, "mean_abs_DIR": mean_abs_dir,
        "ratio_MAG_over_DIR": mean_abs_mag / max(mean_abs_dir, 1e-6),
    }
    print(f"  TEST A summary: MAG wins {n_mag_wins}/{len(test_a_results)}, "
          f"mean |MAG|/|DIR| = {test_a_summary['ratio_MAG_over_DIR']:.2f}")

    # ----- TEST B: BALANCE-as-norm correlations -----
    print(f"\n  TEST B — r(Δ‖residual‖, BALANCE projection) per suffix family:")
    print(f"  {'layer':<6}  {'suffix family':<22}  r")
    test_b_results = []
    for L in BALANCE_LAYERS:
        bal = build_schema_clean("BALANCE", L)
        for sn, pairs in SUFFIX_PAIRS.items():
            deltas = []
            bal_projs = []
            for b, i in pairs:
                rb = residuals[b][L]; ri = residuals[i][L]
                deltas.append(np.linalg.norm(ri) - np.linalg.norm(rb))
                diff = (ri / np.linalg.norm(ri)) - (rb / np.linalg.norm(rb))
                bal_projs.append(float(diff @ bal))
            if len(deltas) >= 4 and np.std(deltas) > 1e-8 and np.std(bal_projs) > 1e-8:
                r = float(np.corrcoef(deltas, bal_projs)[0, 1])
            else:
                r = float("nan")
            test_b_results.append({"layer": L, "suffix": sn, "r": r})
            print(f"  L{L:<5}  {sn:<22}  {r:+.3f}")

    # Aggregate: mean r per layer
    print(f"\n  TEST B aggregated by layer (mean r across suffix families):")
    bal_layer_means = {}
    for L in BALANCE_LAYERS:
        rs = [r["r"] for r in test_b_results if r["layer"] == L and not np.isnan(r["r"])]
        if rs:
            bal_layer_means[L] = np.mean(rs)
            print(f"  L{L}: mean r = {bal_layer_means[L]:+.3f}")

    del model
    torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None
    gc.collect()
    return {
        "model": model_name,
        "n_layers": N_LAYERS,
        "d_model": d_model,
        "L_mid": L_mid,
        "test_a": test_a_summary,
        "test_a_results": test_a_results,
        "test_b_layer_means": bal_layer_means,
        "test_b_results": test_b_results,
    }


# ============================================================================
# Run all models
# ============================================================================

all_results = []
for m in MODELS:
    try:
        res = run_for_model(m)
        all_results.append(res)
    except Exception as e:
        print(f"\n!! FAILED for {m}: {e}")
        import traceback; traceback.print_exc()


# ============================================================================
# Cross-model summary
# ============================================================================

print("\n" + "=" * 78)
print("CROSS-MODEL SUMMARY")
print("=" * 78)

print("\nTEST A — extent words on MAG vs DIR (at middle layer)")
print(f"  {'model':<14}  {'L_mid':>6}  {'MAG wins':>10}  {'mean|MAG|':>10}  "
      f"{'mean|DIR|':>10}  {'ratio':>8}")
for r in all_results:
    a = r["test_a"]
    print(f"  {r['model']:<14}  L{r['L_mid']:<5}  {a['n_mag_wins']}/{a['n_total']:<8}  "
          f"{a['mean_abs_MAG']:>+10.3f}  {a['mean_abs_DIR']:>+10.3f}  "
          f"{a['ratio_MAG_over_DIR']:>8.2f}")

print("\nTEST B — BALANCE-as-norm correlation (mean r across suffix families)")
print(f"  {'model':<14}  layer_means")
for r in all_results:
    print(f"  {r['model']:<14}  " +
          "  ".join(f"L{L}:{v:+.2f}" for L, v in sorted(r['test_b_layer_means'].items())))

# Quick verdict per claim
print("\nVERDICT (rule of thumb):")
print("  - extent-on-MAG holds if ratio > 1.3 across most model sizes")
print("  - BALANCE-as-norm holds if mean r > 0.7 across most layers / sizes")

# extent-on-MAG ratios
ratios = [r["test_a"]["ratio_MAG_over_DIR"] for r in all_results]
print(f"\n  extent-on-MAG ratios across sizes: {[f'{x:.2f}' for x in ratios]}")
print(f"  -> {sum(1 for x in ratios if x > 1.3)}/{len(ratios)} sizes have ratio > 1.3")

# BALANCE-norm correlations
bal_means_all = []
for r in all_results:
    for L, v in r["test_b_layer_means"].items():
        bal_means_all.append(v)
print(f"\n  BALANCE-norm correlations (all layers, all sizes):")
print(f"    n = {len(bal_means_all)}, mean = {np.mean(bal_means_all):+.3f}, "
      f"min = {min(bal_means_all):+.3f}, max = {max(bal_means_all):+.3f}")
print(f"    {sum(1 for x in bal_means_all if x > 0.7)}/{len(bal_means_all)} > 0.7")

print("\nDone.")
