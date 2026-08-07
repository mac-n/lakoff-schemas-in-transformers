"""
exp99_potency_external_validation.py

External validation of the HARDNESS finding against independently-built human norms,
following the web-Claude advisor design (see CAUSAL_VALIDATION_PLAN.md §4).

The worry being tested: is soft-hard "hardness", or is it just one lexical handle on
Osgood's POTENCY dimension (strong-weak / heavy-light / dominant-submissive)?

Checks:
  1. Correlate each physical axis's word-projections against Warriner DOMINANCE
     (Dominance ~ Potency ~ hardness). Strong corr = external validation, kills
     "artifact of my carving". Valence/Arousal reported for context.
  2. Correlate against Brysbaert CONCRETENESS (rule out a concrete/abstract mask).
  3. Four-axis bake-off: pairwise cosines of soft-hard, strong-weak, heavy-light,
     dominant-submissive. High cosines => near-rotations of one subspace => Potency.
  4. Residual test: regress strong-weak + heavy-light out of soft-hard. Does hardness
     retain independent variance? If so, what does the residual look like (epistemic
     "hard facts/hard science" sense)?

Substrate: glove-wiki-gigaword-300. Axes built as unit(mean(w[hard_pole]-w[soft_pole])).
Word position on an axis = cosine(word_vec, axis) (scale-invariant; avoids freq/norm confound).
Anchor words are excluded from each axis's own correlation set.
"""

import numpy as np
import pandas as pd
import gensim.downloader as api
from scipy.stats import pearsonr, spearmanr

RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# Axis anchor pairs: (soft/low-pole word, hard/high-pole word).
# Direction = high_pole - low_pole, so positive projection = hard/strong/heavy/dominant.
# Built from SOURCE-DOMAIN-ONLY (physical) language as much as possible.
# ----------------------------------------------------------------------------
AXES = {
    # exp90's canonical HARDNESS anchors (the axis the +21pp finding was actually about)
    "HARDNESS_exp90 (soft->hard)": [
        ("soft", "hard"), ("mushy", "firm"), ("pliable", "rigid"), ("flimsy", "solid"),
    ],
    # firm/solid removed -- 'firm' (business firm) and 'solid' (solid company/performance)
    # are polysemous; this isolates whether the management/company residual is an artifact
    "HARDNESS_clean (no firm/solid)": [
        ("soft", "hard"), ("mushy", "stiff"), ("pliable", "rigid"), ("squishy", "rigid"),
    ],
    # broader hardness set (note: extra soft anchors pull in a food-texture register)
    "HARDNESS_broad (soft->hard)": [
        ("soft", "hard"), ("squishy", "rigid"), ("pliable", "stiff"),
        ("flexible", "inflexible"), ("supple", "rigid"), ("mushy", "firm"),
        ("tender", "tough"), ("limp", "stiff"), ("springy", "hard"),
    ],
    "STRENGTH (weak->strong)": [
        ("weak", "strong"), ("feeble", "powerful"), ("frail", "mighty"),
        ("flimsy", "sturdy"), ("puny", "formidable"), ("delicate", "robust"),
        ("fragile", "durable"), ("weakling", "powerhouse"),
    ],
    "WEIGHT (light->heavy)": [
        ("lightweight", "heavyweight"), ("weightless", "weighty"),
        ("airy", "dense"), ("buoyant", "ponderous"), ("nimble", "cumbersome"),
        ("light", "heavy"), ("feathery", "leaden"),
    ],
    "DOMINANCE (submissive->dominant)": [
        ("submissive", "dominant"), ("subordinate", "superior"),
        ("meek", "assertive"), ("timid", "authoritative"),
        ("docile", "domineering"), ("deferential", "commanding"),
        ("obedient", "bossy"), ("passive", "controlling"),
    ],
}


def build_axis(kv, pairs):
    """unit(mean(w[hi] - w[lo])) over pairs where both words are in vocab."""
    offs, used, dropped = [], [], []
    for lo, hi in pairs:
        if lo in kv and hi in kv:
            offs.append(kv[hi] - kv[lo])
            used.append((lo, hi))
        else:
            dropped.append((lo, hi))
    v = np.mean(offs, axis=0)
    v = v / np.linalg.norm(v)
    return v, used, dropped


def projections(kv, axis, words):
    """cosine(word, axis) for each word present in vocab; returns (vals, kept_words)."""
    vals, kept = [], []
    for w in words:
        if w in kv:
            v = kv[w]
            vals.append(float(np.dot(v, axis) / np.linalg.norm(v)))
            kept.append(w)
    return np.array(vals), kept


def corr_block(axis_vec, anchor_words, norm_words, norm_vals, kv, label):
    """Correlate axis projections vs a norm vector, excluding anchor words."""
    pairs = [(w, val) for w, val in zip(norm_words, norm_vals)
             if w in kv and w not in anchor_words and np.isfinite(val)]
    ws = [w for w, _ in pairs]
    y = np.array([val for _, val in pairs])
    x, kept = projections(kv, axis_vec, ws)
    # align y to kept
    keptset = {w: i for i, w in enumerate(ws)}
    y = np.array([y[keptset[w]] for w in kept])
    r, rp = pearsonr(x, y)
    rho, rhop = spearmanr(x, y)
    return dict(label=label, n=len(kept), pearson=r, pearson_p=rp,
                spearman=rho, spearman_p=rhop)


def main():
    print("Loading glove-wiki-gigaword-300 ...")
    kv = api.load("glove-wiki-gigaword-300")
    print(f"  vocab: {len(kv.index_to_key):,}\n")

    # ---- Build axes ----
    axes, anchors = {}, {}
    print("=" * 78)
    print("AXIS CONSTRUCTION")
    print("=" * 78)
    for name, pairs in AXES.items():
        v, used, dropped = build_axis(kv, pairs)
        axes[name] = v
        anchors[name] = set([w for p in used for w in p])
        print(f"\n{name}")
        print(f"  pairs used ({len(used)}): {used}")
        if dropped:
            print(f"  DROPPED (oov): {dropped}")

    # ---- Load norms ----
    print("\n" + "=" * 78)
    print("LOADING NORMS")
    print("=" * 78)
    war = pd.read_csv("norms/Warriner_VAD.csv")
    war["Word"] = war["Word"].astype(str).str.lower()
    val = dict(zip(war["Word"], war["V.Mean.Sum"]))
    aro = dict(zip(war["Word"], war["A.Mean.Sum"]))
    dom = dict(zip(war["Word"], war["D.Mean.Sum"]))
    print(f"  Warriner VAD: {len(dom):,} words (valence/arousal/dominance)")

    bry = pd.read_csv("norms/Brysbaert_concreteness.txt", sep="\t")
    bry = bry[bry["Bigram"] == 0]
    bry["Word"] = bry["Word"].astype(str).str.lower()
    conc = dict(zip(bry["Word"], bry["Conc.M"]))
    print(f"  Brysbaert concreteness (single words): {len(conc):,}")

    norm_sets = {
        "Warriner DOMINANCE": (list(dom.keys()), list(dom.values())),
        "Warriner Valence  ": (list(val.keys()), list(val.values())),
        "Warriner Arousal  ": (list(aro.keys()), list(aro.values())),
        "Brysbaert Concrete": (list(conc.keys()), list(conc.values())),
    }

    # ---- Correlations ----
    print("\n" + "=" * 78)
    print("CORRELATIONS  (Pearson r / Spearman rho; anchor words excluded)")
    print("=" * 78)
    print(f"\n{'axis':<34}{'norm':<22}{'n':>7}{'pearson':>10}{'spearman':>10}")
    print("-" * 83)
    for aname, av in axes.items():
        for nname, (nw, nv) in norm_sets.items():
            res = corr_block(av, anchors[aname], nw, nv, kv, nname)
            star = "***" if res["pearson_p"] < 1e-3 else ("*" if res["pearson_p"] < 0.05 else "")
            print(f"{aname:<34}{nname:<22}{res['n']:>7}"
                  f"{res['pearson']:>10.3f}{res['spearman']:>10.3f}  {star}")
        print()

    # ---- Bake-off: pairwise cosines ----
    print("=" * 78)
    print("FOUR-AXIS BAKE-OFF: pairwise cosines")
    print("  (high |cos| => near-rotations of one subspace => Osgood POTENCY)")
    print("=" * 78)
    names = list(axes.keys())
    print(f"\n{'':<34}" + "".join(f"{n.split()[0][:8]:>10}" for n in names))
    for n1 in names:
        row = f"{n1:<34}"
        for n2 in names:
            row += f"{float(np.dot(axes[n1], axes[n2])):>10.3f}"
        print(row)

    # ---- Residual test (run for both hardness variants to expose the firm artifact) ----
    print("\n" + "=" * 78)
    print("RESIDUAL TEST: regress STRENGTH + WEIGHT out of HARDNESS")
    print("  (does soft-hard retain independent variance beyond Potency-proper?")
    print("   and is the management/company residual a 'firm' polysemy artifact?)")
    print("=" * 78)
    basis = [axes["STRENGTH (weak->strong)"], axes["WEIGHT (light->heavy)"]]
    onb = []
    for b in basis:
        w = b.copy()
        for e in onb:
            w = w - np.dot(w, e) * e
        nrm = np.linalg.norm(w)
        if nrm > 1e-8:
            onb.append(w / nrm)
    # precompute a frequent-word matrix once for top-word readouts
    vocab = kv.index_to_key[:40000]
    M = np.stack([kv[w] for w in vocab])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

    for hname in ["HARDNESS_exp90 (soft->hard)", "HARDNESS_clean (no firm/solid)"]:
        h = axes[hname]
        resid = h.copy()
        for e in onb:
            resid = resid - np.dot(resid, e) * e
        resid_frac = float(np.linalg.norm(resid))
        print(f"\n--- {hname} ---")
        print(f"  in span(STRENGTH,WEIGHT): {np.sqrt(max(0,1-resid_frac**2)):.3f}   "
              f"||residual||: {resid_frac:.3f}   "
              f"=> {resid_frac**2*100:.1f}% OUTSIDE potency-proper")
        if resid_frac > 1e-6:
            resid_u = resid / resid_frac
            for nname in ["Warriner DOMINANCE", "Brysbaert Concrete", "Warriner Valence  "]:
                nw, nv = norm_sets[nname]
                res = corr_block(resid_u, anchors[hname], nw, nv, kv, nname)
                print(f"    {nname:<22} n={res['n']:>6}  pearson={res['pearson']:>6.3f}  "
                      f"spearman={res['spearman']:>6.3f}")
            scores = M @ resid_u
            order = np.argsort(scores)
            print("  Most HARD-residual:", [vocab[i] for i in order[::-1][:25]])
            print("  Most SOFT-residual:", [vocab[i] for i in order[:25]])

    print("\n" + "=" * 78)
    print("READING GUIDE")
    print("=" * 78)
    print("""
  - If HARDNESS correlates with DOMINANCE about as strongly as STRENGTH/WEIGHT/
    DOMINANCE-axis do, AND the four axes have high pairwise cosines, the honest
    framing is 'Osgood Potency, measured', not 'I found hardness'.
  - If HARDNESS's concreteness correlation is much higher than its dominance
    correlation, it may be a concrete/abstract axis wearing a hardness mask.
  - If the residual (hardness minus strength+weight) is large AND its top words
    are epistemic (hard facts / hard data / hard science / rigorous), THAT residual
    is the real, defensible discovery.
""")


if __name__ == "__main__":
    main()
