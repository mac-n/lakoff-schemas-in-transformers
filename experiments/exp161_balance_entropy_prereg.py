"""
exp161_balance_entropy_prereg.py — pre-registered confirmatory test of
the GPT-2 BALANCE <-> attention-entropy coupling (exp160's post-hoc
find, exp160b's surviving lead).

PRE-REGISTRATION: PREREG_exp161.md (frozen 2026-06-12 BEFORE this file
was written; v1.1 amendment logged there). Read it first. Summary:
  - FRESH prompts (Appendix A of the prereg; exp160's 40 are not reused).
  - C1 replication: partial r(BAL, ent | pos, norm).
  - C2 HEADLINE: covars = pos, norm, d_norm_ho-projection; axis =
    BALANCE projected perp d_norm_ho (held-out per exp154c protocol).
  - C3 (non-gating): axis additionally perp the other 7 schema axes.
  - C4 specificity: all-8 partial table with C2 covariates.
  - C5: axis-defining tokens (any schema) excluded from analysis.
  - Cluster bootstrap (resample prompts, 1000 reps) 95% CI on C2.
  - S1 (non-gating, declared): pairwise nonlinear dependence of BALANCE
    on each other schema projection — Pearson vs Spearman vs dCor with
    permutation null on the excess dCor^2 - r^2.
  - PRECONDITION: held-out d_norm carrier >= 0.50 at every GPT-2
    decision layer, else INVALID (not null).
  - Verdict (GPT-2 only, decision layers 8/12/16):
      V1a/V1b candidate stands / V2 dead / V3 ambiguous / INVALID.

Conventions honoured: shared machinery from attn_entropy_lib +
markedness_norm_protocol (no copy-paste); import-safe (__main__ guard);
synthetic verdict selftest INCLUDING precondition-fails+headline-null;
HF_TOKEN set explicitly for TL.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import (SCHEMA_NAMES, COMMON, RARE, corrf,
                                      build_word_lists, collect_residuals)
from attn_entropy_lib import MODELS, attn_entropy_per_query, partial_corr

DECISION_HI = 0.15   # V1 threshold on |C2| (negative sign required)
DECISION_LO = 0.10   # V2 threshold
CARRIER_MIN = 0.50   # precondition on held-out d_norm carrier
N_BOOT = 1000
N_PERM = 1000
SEED = 161

FRESH_PROMPTS = [
    "The kettle clicked off and she poured the water over the leaves.",
    "Halfway through the meeting nobody could remember who was chairing it.",
    "The carpenter measured twice and the shelf fit on the first try.",
    "Loose papers slid from the pile every time the door swung open.",
    "The orchestra tuned to a single note and the hall went quiet.",
    "His suitcase burst at the airport and his clothes went everywhere.",
    "She filed the receipts by month and closed the drawer with a click.",
    "The toddler stacked the blocks until the tower leaned and toppled.",
    "The ferry crossed on schedule despite the morning swell.",
    "Three alarms went off at once and he silenced none of them.",
    "The gardener pruned the hedge into a clean straight line.",
    "The spreadsheet totals refused to match no matter who re-added them.",
    "A heron stood motionless at the edge of the reservoir.",
    "The debate drifted off topic within the first five minutes.",
    "The baker weighed the flour and the dough came out the same as always.",
    "Her headphones tangled into a knot at the bottom of the bag.",
    "The surveyor set the tripod and the bubble came to rest in the centre.",
    "Rumours of the merger changed shape with every retelling.",
    "The night train ran quiet and true across the plain.",
    "The committee's minutes contradicted the recording in three places.",
    "He laced his boots, checked the map, and set off at first light.",
    "Paint dripped from the ladder onto the new carpet below.",
    "The accountant reconciled the books before the end of the quarter.",
    "Half the streetlights were out and the road kept changing under him.",
    "The juggler kept four clubs aloft in an unbroken arc.",
    "The printer jammed again and the queue of documents kept growing.",
    "The librarian reshelved the returns before the doors opened.",
    "Wind gusted through the market and the stalls flapped and strained.",
    "The pilot trimmed the aircraft and the nose held its line.",
    "The recipe doubled badly and the sauce split in the pan.",
    "She watered the plants on the same morning every week.",
    "The scaffolding clattered as the storm pulled at its joints.",
    "The clockmaker set the pendulum and the ticking evened out.",
    "Boxes from the move still blocked the hallway a month later.",
    "The rowers found their rhythm and the boat ran flat and fast.",
    "His password expired mid-task and the form deleted everything.",
    "The nurse charted the doses in a neat, unhurried hand.",
    "Gravel spilled across the lane where the wall had given way.",
    "The tide came in exactly as the tables said it would.",
    "The signal cut out each time the speaker reached her point.",
]

AXIS_WORDS = set()
for _sn in SCHEMA_NAMES:
    for _p, _n in LAKOFF_SCHEMAS_MML[_sn]:
        AXIS_WORDS.add(_p.lower()); AXIS_WORDS.add(_n.lower())


def _selfcheck_prompts():
    """Frozen-prompt integrity: no BALANCE axis word may appear (C5
    handles the rest at token level)."""
    import re
    bal = set()
    for p, n in LAKOFF_SCHEMAS_MML["BALANCE"]:
        bal.add(p.lower()); bal.add(n.lower())
    toks = set(re.findall(r"[a-z']+", " ".join(FRESH_PROMPTS).lower()))
    bad = toks & bal
    assert not bad, f"BALANCE axis words in fresh prompts: {bad}"


# ---------------- direction machinery (exp154c maths, exp160 strip) ----

def proj_out(d, axis):
    d = d - (d @ axis) * axis
    return d / np.linalg.norm(d)


def build_layer_dirs(residuals, L, all_words, est_words, test_words):
    """aniso/freq strip, 8 schema dirs, HELD-OUT d_norm + carrier,
    BALANCE orthogonalisation variants. Mirrors markedness_norm_protocol
    .analyze_layer's d_norm maths (est_words excludes BALANCE vocab, so
    cos(BALANCE, d_norm_ho) is non-circular by construction)."""
    aniso = np.stack([residuals[w][L] for w in all_words]).mean(axis=0)
    aniso = aniso / np.linalg.norm(aniso)

    def mean_acts(words):
        return np.mean([residuals[w][L] for w in words], axis=0)

    freq = mean_acts(COMMON) - mean_acts(RARE)
    freq = freq / np.linalg.norm(freq)
    freq_orth = proj_out(freq, aniso)

    def strip(d):
        d = d - (d @ aniso) * aniso
        d = d - (d @ freq_orth) * freq_orth
        return d / np.linalg.norm(d)

    norms = {w: float(np.linalg.norm(residuals[w][L])) for w in all_words}
    units = {w: residuals[w][L] / norms[w] for w in all_words}

    nv = np.array([norms[w] for w in est_words])
    zz = {w: (norms[w] - nv.mean()) / nv.std() for w in est_words}
    d_norm_ho = np.sum([zz[w] * units[w] for w in est_words], axis=0)
    d_norm_ho = strip(d_norm_ho / np.linalg.norm(d_norm_ho))

    z_test = [(norms[w] - nv.mean()) / nv.std() for w in test_words]
    carrier_out = corrf([float(units[w] @ d_norm_ho) for w in test_words],
                        z_test)

    sd = {}
    for sn in SCHEMA_NAMES:
        pairs = LAKOFF_SCHEMAS_MML[sn]
        pos = sorted(set(p[0] for p in pairs))
        neg = sorted(set(p[1] for p in pairs))
        raw = mean_acts(pos) - mean_acts(neg)
        sd[sn] = strip(raw / np.linalg.norm(raw))

    bal = sd["BALANCE"]
    bal_pn = proj_out(bal, d_norm_ho)                       # C2 axis
    others = [sd[sn] for sn in SCHEMA_NAMES if sn != "BALANCE"]
    basis, _ = np.linalg.qr(np.stack(others + [d_norm_ho]).T)  # d x 8
    bal_pall = bal - basis @ (basis.T @ bal)                # C3 axis
    bal_pall = bal_pall / np.linalg.norm(bal_pall)

    return dict(schema=sd, d_norm_ho=d_norm_ho, carrier_out=carrier_out,
                bal_pn=bal_pn, bal_pall=bal_pall,
                cos_bal_dnorm=float(bal @ d_norm_ho))


# ---------------- token-cloud collection (C5 exclusion) ----------------

def collect_cloud(model, cfg, dirs):
    """Per layer: arrays of schema projections, BALANCE-variant
    projections, d_norm projection, entropy, position, norm, prompt id.
    Tokens whose stripped lowercased string is an axis word for ANY
    schema are excluded (C5)."""
    LAYERS = cfg["layers"]
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]
    acc = {L: dict(proj={sn: [] for sn in SCHEMA_NAMES},
                   bal_pn=[], bal_pall=[], dnorm=[],
                   ent=[], pos=[], norm=[], pid=[])
           for L in LAYERS}
    n_excluded = 0
    for pid, prompt in enumerate(FRESH_PROMPTS):
        toks = model.to_tokens(prompt)
        strs = model.to_str_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in LAYERS:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            pat = c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
            ent = attn_entropy_per_query(pat)
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                if strs[q].strip().lower().strip(".,!?'\"") in AXIS_WORDS:
                    if L == LAYERS[0]:
                        n_excluded += 1
                    continue                                   # C5
                r = resid[q]; nrm = np.linalg.norm(r); u = r / nrm
                a = acc[L]
                for sn in SCHEMA_NAMES:
                    a["proj"][sn].append(float(u @ dirs[L]["schema"][sn]))
                a["bal_pn"].append(float(u @ dirs[L]["bal_pn"]))
                a["bal_pall"].append(float(u @ dirs[L]["bal_pall"]))
                a["dnorm"].append(float(u @ dirs[L]["d_norm_ho"]))
                a["ent"].append(ent[q]); a["pos"].append(q)
                a["norm"].append(nrm); a["pid"].append(pid)
    return acc, n_excluded


# ---------------- statistics ----------------

def layer_stats(a, rng):
    pos = np.array(a["pos"], float); nrm = np.array(a["norm"], float)
    ent = np.array(a["ent"], float); dnp = np.array(a["dnorm"], float)
    pid = np.array(a["pid"], int)
    bal = np.array(a["proj"]["BALANCE"], float)
    bal_pn = np.array(a["bal_pn"], float)
    bal_pall = np.array(a["bal_pall"], float)

    c1 = partial_corr(bal, ent, [pos, nrm])
    covs2 = [pos, nrm, dnp]
    c2 = partial_corr(bal_pn, ent, covs2)
    c3 = partial_corr(bal_pall, ent, covs2)
    table = {sn: partial_corr(np.array(a["proj"][sn], float), ent, covs2)
             for sn in SCHEMA_NAMES}
    rank = 1 + sum(1 for sn in SCHEMA_NAMES if sn != "BALANCE"
                   and abs(table[sn]) > abs(table["BALANCE"]))

    boots = []
    upids = np.unique(pid)
    idx_of = {p: np.where(pid == p)[0] for p in upids}
    for _ in range(N_BOOT):
        sel = np.concatenate([idx_of[p] for p in
                              rng.choice(upids, len(upids), replace=True)])
        boots.append(partial_corr(bal_pn[sel], ent[sel],
                                  [pos[sel], nrm[sel], dnp[sel]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return dict(c1=c1, c2=c2, c3=c3, table=table, rank=rank,
                ci=(float(lo), float(hi)), n=len(ent))


def _dcenter(x):
    D = np.abs(x[:, None] - x[None, :])
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return corrf(rx, ry)


def s1_nonlinear(a, rng):
    """S1: BALANCE vs each other schema — Pearson, Spearman, dCor, and a
    permutation p-value for the nonlinear excess dCor^2 - r^2."""
    bal = np.array(a["proj"]["BALANCE"], float)
    A = _dcenter(bal)
    A2m = (A * A).mean()
    out = {}
    for sn in SCHEMA_NAMES:
        if sn == "BALANCE":
            continue
        y = np.array(a["proj"][sn], float)
        B = _dcenter(y)
        B2m = (B * B).mean()
        r = corrf(bal, y)
        rho = _spearman(bal, y)
        dcor2 = (A * B).mean() / np.sqrt(A2m * B2m)
        excess = dcor2 - r * r
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            p = rng.permutation(len(y))
            Bp = B[p][:, p]
            d2 = (A * Bp).mean() / np.sqrt(A2m * B2m)
            null[i] = d2 - corrf(bal, y[p]) ** 2
        pval = float((np.sum(null >= excess) + 1) / (N_PERM + 1))
        out[sn] = dict(r=r, rho=rho, dcor2=float(dcor2),
                       excess=float(excess), p=pval)
    return out


# ---------------- verdict (pure, selftest-able) ----------------

def verdict161(layers_summary, decision_layers,
               hi=DECISION_HI, lo=DECISION_LO, carrier_min=CARRIER_MIN):
    """layers_summary[L]: dict with c2, c3, rank, carrier. GPT-2 only.
    Returns one of INVALID / V1a / V1b / V2 / V3 (prereg decision rule)."""
    if any(layers_summary[L]["carrier"] < carrier_min
           for L in decision_layers):
        return "INVALID"
    c2 = [layers_summary[L]["c2"] for L in decision_layers]
    v1_couple = all((c <= -hi) for c in c2)              # negative AND big
    spec = sum(1 for L in decision_layers if layers_summary[L]["rank"] <= 2)
    v1_spec = spec > len(decision_layers) / 2
    if v1_couple and v1_spec:
        c3_big = sum(1 for L in decision_layers
                     if abs(layers_summary[L]["c3"]) >= hi)
        return "V1a" if c3_big > len(decision_layers) / 2 else "V1b"
    if all(abs(c) < lo for c in c2):
        return "V2"
    return "V3"


def selftest_verdict161(dl=(8, 12, 16)):
    def fake(c2, rank, carrier=0.8, c3=0.0):
        return {L: dict(c2=c2, c3=c3, rank=rank, carrier=carrier) for L in dl}
    # precondition fails + headline null -> INVALID (exp158 lesson, case #5)
    assert verdict161(fake(-0.02, 5, carrier=0.30), dl) == "INVALID"
    # precondition fails + headline strong -> still INVALID
    assert verdict161(fake(-0.30, 1, carrier=0.30), dl) == "INVALID"
    # strong, specific, C3 survives -> V1a
    assert verdict161(fake(-0.25, 1, c3=-0.20), dl) == "V1a"
    # strong, specific, C3 dies -> V1b
    assert verdict161(fake(-0.25, 2, c3=-0.03), dl) == "V1b"
    # null everywhere -> V2
    assert verdict161(fake(-0.05, 4), dl) == "V2"
    # strong but WRONG SIGN -> V3, never V1
    assert verdict161(fake(+0.25, 1), dl) == "V3"
    # strong but non-specific -> V3
    assert verdict161(fake(-0.25, 5), dl) == "V3"
    # intermediate -> V3
    assert verdict161(fake(-0.12, 1), dl) == "V3"
    # mixed carrier: one decision layer below threshold -> INVALID
    s = fake(-0.25, 1, c3=-0.20); s[12]["carrier"] = 0.4
    assert verdict161(s, dl) == "INVALID"
    print("selftest_verdict161: all branches fire correctly "
          "(incl. precondition-fails + headline-null).")


# ---------------- run ----------------

def main():
    print("exp161 — pre-registered BALANCE<->entropy test "
          "(PREREG_exp161.md, frozen before this script)")
    print("Selftests first (synthetic-verdict convention)...")
    selftest_verdict161()
    _selfcheck_prompts()
    print("Prompt integrity check passed (no BALANCE axis words).")

    all_words, est_words, test_words = build_word_lists()
    print(f"vocab: {len(all_words)} words "
          f"(est {len(est_words)} / test {len(test_words)}; "
          f"BALANCE vocab held out of d_norm estimation)")

    results = {}
    for tag, cfg in MODELS.items():
        rng = np.random.default_rng(SEED)
        print(f"\n{'='*72}\n{tag}\n{'='*72}")
        model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
        model.eval()
        LAYERS = cfg["layers"]

        residuals = collect_residuals(model, LAYERS, all_words, log_every=0)
        dirs = {L: build_layer_dirs(residuals, L, all_words,
                                    est_words, test_words) for L in LAYERS}
        acc, n_excl = collect_cloud(model, cfg, dirs)
        print(f"  C5 axis-word exclusion: {n_excl} tokens dropped/layer-set")

        summary = {}
        print(f"  {'L':>3} {'carrier':>8} {'cosBN':>7} {'C1':>7} "
              f"{'C2(headline)':>13} {'C2 95%CI':>16} {'C3':>7} "
              f"{'rk':>3} {'n':>5}")
        for L in LAYERS:
            st = layer_stats(acc[L], rng)
            st["carrier"] = dirs[L]["carrier_out"]
            st["cos_bal_dnorm"] = dirs[L]["cos_bal_dnorm"]
            summary[L] = st
            print(f"  {L:>3} {st['carrier']:>8.2f} "
                  f"{st['cos_bal_dnorm']:>+7.2f} {st['c1']:>+7.3f} "
                  f"{st['c2']:>+13.3f} "
                  f"[{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] "
                  f"{st['c3']:>+7.3f} {st['rank']:>3} {st['n']:>5}")
        for L in cfg["decision"]:
            t = summary[L]["table"]
            ranked = sorted(SCHEMA_NAMES, key=lambda s: -abs(t[s]))
            print(f"  L{L} C4 partial-r: " +
                  ", ".join(f"{s.split('-')[0][:5]} {t[s]:+.2f}"
                            for s in ranked))
        print("  S1 (nonlinear excess dCor^2-r^2 vs BALANCE; perm p):")
        for L in cfg["decision"]:
            s1 = s1_nonlinear(acc[L], rng)
            summary[L]["s1"] = s1
            line = ", ".join(
                f"{sn.split('-')[0][:5]} ex{v['excess']:+.3f}(p{v['p']:.3f})"
                for sn, v in sorted(s1.items(),
                                    key=lambda kv: -kv[1]["excess"])[:4])
            print(f"    L{L}: {line}")
        results[tag] = (summary, cfg["decision"])
        del model, residuals, acc
        gc.collect(); torch.mps.empty_cache()

    print("\n" + "=" * 72)
    print("VERDICT vs PREREG_exp161.md (GPT-2 gates; others reported)")
    print("=" * 72)
    for tag, (summary, decision) in results.items():
        code = verdict161(summary, decision)
        gate = " <-- GATES" if tag == "gpt2-medium" else " (report only)"
        cs = ", ".join(f"L{L}:{summary[L]['c2']:+.2f}(rk{summary[L]['rank']},"
                       f"car{summary[L]['carrier']:.2f})" for L in decision)
        print(f"  {tag:<14} {code:<8} [{cs}]{gate}")
    print("\n  V1a/V1b = candidate stands (unique/shared) -> causal exp165+")
    print("  V2 = dead -> carrier list   V3 = ambiguous   INVALID = fix d_norm")

    # P5 check (cross-substrate fingerprint)
    if "gpt2-medium" in results and "pythia-410m" in results:
        g, gd = results["gpt2-medium"]
        p, pd = results["pythia-410m"]
        gm = np.mean([abs(g[L]["c2"]) for L in gd])
        pm = np.mean([abs(p[L]["c2"]) for L in pd])
        print(f"\n  P5: mean|C2| GPT-2 {gm:.3f} vs Pythia {pm:.3f} -> "
              f"{'HIT' if pm < gm else 'MISS'} "
              f"(predicted Pythia < GPT-2 under full controls)")


if __name__ == "__main__":
    from transformer_lens import HookedTransformer
    main()
