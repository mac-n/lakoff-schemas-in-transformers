"""
exp178_what_produces_coupling.py — what produces the BALANCE<->attention-
entropy coupling? Frozen rules: PREREG_exp178.md (commit 2e0d682,
2026-10-02 18:19 IST). Written AFTER the freeze.

Part 1: random axes with the same token split as the real BALANCE axis
        (original bare-word protocol).
Part 2: a clean BALANCE axis built from single-token words inside neutral
        sentences, longer lists; compared with random clean axes.

exp166's machinery is imported; the token-cloud loop mirrors
exp166.run_model.

Usage:  ./lakoff/bin/python3 exp178_what_produces_coupling.py gpt2-medium
        ./lakoff/bin/python3 exp178_what_produces_coupling.py Llama-3.2-1B
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from attn_entropy_lib import attn_entropy_per_query, partial_corr
from markedness_norm_protocol import build_word_lists, collect_residuals, COMMON, RARE
from exp161_balance_entropy_prereg import build_layer_dirs, proj_out, AXIS_WORDS
from exp164_depth_map_nonlinear_controls import covar_stacks
from exp166_balance_entropy_third_set import (FRESH_PROMPTS, DECISION_LO, CARRIER_MIN,
                                              N_BOOT, SEED, band_status, control_status,
                                              validate_harness)
from exp177_fragment_reading import token_types, strip_tools

REPL_TOL = 0.02
N_RAND = 500
SEED_RAND = 178
DEVICE = "mps"
MODELS = {
    "gpt2-medium": dict(repo="gpt2-medium", layers=[3, 8, 12, 16], judge="control", judged=[8, 12, 16],
                        base={3: -0.322, 8: -0.266, 12: -0.320, 16: -0.145}),
    "Llama-3.2-1B": dict(repo="meta-llama/Llama-3.2-1B", layers=[5, 6, 7, 13], judge="band", judged=[5, 6, 7],
                         base={5: -0.139, 6: -0.168, 7: -0.202, 13: -0.268}),
}
POS_EXT = """balanced stable steady even level equal firm solid settled upright centered aligned poised tight
grounded anchored secure fixed still regular uniform composed sturdy rooted steadfast flat square""".split()
NEG_EXT = """unstable uneven unequal tilted tipping shaky skewed slack fallen rocky precarious loose leaning tipped
stumbling slipping sliding collapsing staggering erratic volatile irregular""".split()
FRAMES = [
    "she wrote the word ___ on the board before the lesson began that morning",
    "the next word on the list is ___ and then we move on to the rest of the page",
    "after the move the old table looked ___ and nobody in the room said anything about it",
    "by the end of the week the whole arrangement seemed ___ to everyone who came to see it",
]
PREREG = open(f"{ROOT}/PREREG_exp178.md").read()
_flat = " ".join(PREREG.split())
for _name, _val in [("DECISION_LO", DECISION_LO), ("CARRIER_MIN", CARRIER_MIN), ("N_BOOT", N_BOOT),
                    ("REPL_TOL", REPL_TOL), ("N_RAND", N_RAND), ("SEED_RAND", SEED_RAND)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
assert len(POS_EXT) == 27 and len(NEG_EXT) == 22
assert " ".join(POS_EXT) in _flat and " ".join(NEG_EXT) in _flat, "word lists drifted from prereg"
for _f in FRAMES:
    assert _f in _flat, "frame drifted from prereg"


def main(tag):
    from transformer_lens import HookedTransformer
    cfg = MODELS[tag]
    LAYERS, JUDGED = cfg["layers"], cfg["judged"]
    print(f"exp178 — what produces the BALANCE<->entropy coupling? — {tag} (prereg frozen at 2e0d682)")
    validate_harness()
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained(cfg["repo"], device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is True and model.to_tokens("the").shape[1] == 2
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]

    # ---- word states: bare (exp166 protocol) and inside the four frames
    res_bare = collect_residuals(model, LAYERS, all_words, log_every=0)
    ntok_bare = {w: int(model.to_tokens(w).shape[1] - 1) for w in all_words}
    words_f = sorted(set(all_words) | set(POS_EXT) | set(NEG_EXT))
    ntok_sp = {w: int(model.to_tokens(" " + w).shape[1] - 1) for w in words_f}
    assert all(ntok_sp[w] == 1 for w in POS_EXT + NEG_EXT), "a list word is not one token here"
    res_frame = {}
    for k, w in enumerate(words_f):
        accf = {L: 0.0 for L in LAYERS}
        for fr in FRAMES:
            head = fr.split("___")[0].rstrip()
            toks = model.to_tokens(fr.replace("___", w))
            ptoks = model.to_tokens(head + " " + w)
            n = ptoks.shape[1]
            assert bool((toks[0, :n] == ptoks[0]).all()), "slot position not prefix-stable"
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=hook_names)
            for L in LAYERS:
                accf[L] = accf[L] + cache[f"blocks.{L}.hook_resid_post"][0, n - 1, :].float().cpu().numpy()
        res_frame[w] = {L: accf[L] / len(FRAMES) for L in LAYERS}
        if (k + 1) % 200 == 0:
            print(f"  in-sentence states {k + 1}/{len(words_f)}", flush=True)

    bal_pairs = LAKOFF_SCHEMAS_MML["BALANCE"]
    pos = sorted(set(p[0] for p in bal_pairs)); neg = sorted(set(p[1] for p in bal_pairs))
    bal_vocab = set(pos) | set(neg)
    pool = [w for w in all_words if w not in bal_vocab and w not in COMMON and w not in RARE]
    singles = [w for w in pool if ntok_bare[w] == 1]; multis = [w for w in pool if ntok_bare[w] > 1]
    s_pos = sum(ntok_bare[w] == 1 for w in pos); m_pos = len(pos) - s_pos
    s_neg = sum(ntok_bare[w] == 1 for w in neg); m_neg = len(neg) - s_neg
    pool_f = [w for w in pool if ntok_sp[w] == 1]
    print(f"\npool {len(pool)} words: {len(singles)} single-token bare, {len(multis)} split; "
          f"{len(pool_f)} single-token with a leading space")
    print(f"real BALANCE token structure: positive {s_pos} single + {m_pos} split, negative {s_neg} single + {m_neg} split")

    dirs = {}
    for L in LAYERS:
        base = build_layer_dirs(res_bare, L, all_words, est_words, test_words)
        strip_b, _ = strip_tools(res_bare, L, all_words)
        fr = build_layer_dirs(res_frame, L, all_words, est_words, test_words)
        strip_f, mean_f = strip_tools(res_frame, L, all_words)
        raw = mean_f(POS_EXT) - mean_f(NEG_EXT)
        clean_full = strip_f(raw / np.linalg.norm(raw))
        dirs[L] = dict(base=base, fr=fr, strip_b=strip_b, strip_f=strip_f,
                       clean=proj_out(clean_full, fr["d_norm_ho"]),
                       cos_orig_clean=float(base["schema"]["BALANCE"] @ clean_full),
                       Rb={w: res_bare[w][L] for w in all_words},
                       Rf={w: res_frame[w][L] for w in words_f})

    # ---- token cloud (mirrors exp166.run_model's loop; unit vectors kept)
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]
    acc = {L: dict(U=[], ent=[], pos=[], norm=[], pid=[], type=[], w=[]) for L in LAYERS}
    n_excl = 0
    for pid, prompt in enumerate(FRESH_PROMPTS):
        toks = model.to_tokens(prompt)
        strs = model.to_str_tokens(prompt)
        types = token_types(strs)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in LAYERS:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            pat = c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy()
            ent = attn_entropy_per_query(pat)
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                wstr = strs[q].strip().lower().strip(".,!?'\"")
                if wstr in AXIS_WORDS:
                    if L == LAYERS[0]:
                        n_excl += 1
                    continue                                   # C5
                r = resid[q]; nrm = np.linalg.norm(r)
                a = acc[L]
                a["U"].append((r / nrm).astype(np.float64)); a["ent"].append(ent[q]); a["pos"].append(q)
                a["norm"].append(nrm); a["pid"].append(pid); a["type"].append(types[q]); a["w"].append(wstr)
        del c
    del model
    print(f"  C5 axis-word exclusion: {n_excl} tokens dropped/layer-set")
    cloud = {}
    list_words = set(POS_EXT) | set(NEG_EXT)
    for L in LAYERS:
        a = acc[L]
        cloud[L] = dict(U=np.stack(a["U"]), ent=np.array(a["ent"], float), pos=np.array(a["pos"], float),
                        norm=np.array(a["norm"], float), pid=np.array(a["pid"], int),
                        whole=np.array([t == "WHOLE" and w not in list_words for t, w in zip(a["type"], a["w"])]),
                        notlist=np.array([w not in list_words for w in a["w"]]))

    def c2q_point(L, axis, dnorm_dir, mask=None):
        c = cloud[L]
        m = np.ones(len(c["ent"]), bool) if mask is None else c[mask]
        bal = c["U"][m] @ axis; dnp = c["U"][m] @ dnorm_dir
        _, quad = covar_stacks(c["pos"][m], c["norm"][m], dnp)
        return partial_corr(bal, c["ent"][m], quad)

    def c2q_boot(axis_of, dn_of, carrier_of, mask=None):
        rng = np.random.default_rng(SEED)
        out = {}
        for L in LAYERS:
            c = cloud[L]
            m = np.ones(len(c["ent"]), bool) if mask is None else c[mask]
            bal = c["U"][m] @ axis_of(L); dnp = c["U"][m] @ dn_of(L)
            pos_, nrm, ent, pid = c["pos"][m], c["norm"][m], c["ent"][m], c["pid"][m]
            _, quad = covar_stacks(pos_, nrm, dnp)
            c2q = partial_corr(bal, ent, quad)
            upids = np.unique(pid); idx_of = {p: np.where(pid == p)[0] for p in upids}
            boots = []
            for _ in range(N_BOOT):
                sel = np.concatenate([idx_of[p] for p in rng.choice(upids, len(upids), replace=True)])
                _, q_sel = covar_stacks(pos_[sel], nrm[sel], dnp[sel])
                boots.append(partial_corr(bal[sel], ent[sel], q_sel))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[L] = dict(c2q=float(c2q), ci=(float(lo), float(hi)), carrier=float(carrier_of(L)), n=int(m.sum()))
        return out

    def classify(out):
        return control_status(out, tuple(JUDGED)) if cfg["judge"] == "control" else band_status(out, JUDGED)[0]

    def show(name, out):
        print(f"  {name:36s} n {out[LAYERS[0]]['n']:>3}  "
              + "  ".join(f"L{L} {out[L]['c2q']:+.3f} [{out[L]['ci'][0]:+.2f},{out[L]['ci'][1]:+.2f}]" for L in LAYERS)
              + "  carriers " + "/".join(f"{out[L]['carrier']:.2f}" for L in LAYERS))

    results = {"model": tag}
    # ---- gate
    print("\n" + "=" * 72 + "\nREPLICATION GATE\n" + "=" * 72)
    p0 = c2q_boot(lambda L: dirs[L]["base"]["bal_pn"], lambda L: dirs[L]["base"]["d_norm_ho"],
                  lambda L: dirs[L]["base"]["carrier_out"])
    show("original analysis", p0)
    gate = all(abs(p0[L]["c2q"] - cfg["base"][L]) <= REPL_TOL for L in LAYERS)
    print(f"  exp166 had {cfg['base']}  ->  {'PASS' if gate else 'FAIL'}")
    results["P0"] = {str(L): p0[L] for L in LAYERS}
    if not gate:
        print("STOP: harness drift. Nothing below was computed.")
        sys.exit(2)

    # ---- Part 1
    print("\n" + "=" * 72 + "\nPART 1 — random axes with BALANCE's token split (bare-word protocol)\n" + "=" * 72)
    rng = np.random.default_rng(SEED_RAND)

    def rand_axis(L, pw, nw, R, strip, dn):
        raw = np.mean([R[w] for w in pw], axis=0) - np.mean([R[w] for w in nw], axis=0)
        return proj_out(strip(raw / np.linalg.norm(raw)), dn)

    matched = {L: [] for L in LAYERS}; single_only = {L: [] for L in LAYERS}
    for _ in range(N_RAND):
        sp = list(rng.choice(singles, s_pos + s_neg, replace=False))
        mp = list(rng.choice(multis, m_pos + m_neg, replace=False))
        pw, nw = sp[:s_pos] + mp[:m_pos], sp[s_pos:] + mp[m_pos:]
        so = list(rng.choice(singles, 30, replace=False))
        for L in LAYERS:
            d = dirs[L]; dn = d["base"]["d_norm_ho"]
            matched[L].append(c2q_point(L, rand_axis(L, pw, nw, d["Rb"], d["strip_b"], dn), dn))
            single_only[L].append(c2q_point(L, rand_axis(L, so[:15], so[15:], d["Rb"], d["strip_b"], dn), dn))
    part1 = {}
    for L in LAYERS:
        mt, so = np.array(matched[L]), np.array(single_only[L])
        real = p0[L]["c2q"]
        part1[L] = dict(real=real, matched_median=float(np.median(mt)), matched_p5=float(np.percentile(mt, 5)),
                        matched_p95=float(np.percentile(mt, 95)), real_percentile=float((mt < real).mean() * 100),
                        single_median=float(np.median(so)), single_p5=float(np.percentile(so, 5)),
                        single_p95=float(np.percentile(so, 95)))
        r = part1[L]
        print(f"  L{L:<2} real BALANCE {real:+.3f} | MATCHED fake axes: median {r['matched_median']:+.3f} "
              f"[5% {r['matched_p5']:+.3f}, 95% {r['matched_p95']:+.3f}]; real sits at the {r['real_percentile']:.0f}th percentile"
              f" | SINGLE-ONLY fake axes: median {r['single_median']:+.3f} [5% {r['single_p5']:+.3f}, 95% {r['single_p95']:+.3f}]"
              f"{'   (judged)' if L in JUDGED else ''}")
    n_spec = sum(part1[L]["real"] < part1[L]["matched_p5"] for L in JUDGED)
    n_suff = sum(part1[L]["matched_median"] <= -DECISION_LO for L in JUDGED)
    v1 = "BALANCE_SPECIFIC" if n_spec >= 2 else ("TOKEN_STRUCTURE_SUFFICIENT" if n_suff >= 2 else "NEITHER")
    print(f"\n  PART 1 VERDICT: {v1}   (real below the matched 5th percentile at {n_spec}/3 judged layers;"
          f" matched median <= -{DECISION_LO} at {n_suff}/3)")
    results["part1"] = {str(L): part1[L] for L in LAYERS}; results["part1_verdict"] = v1

    # ---- Part 2
    print("\n" + "=" * 72 + "\nPART 2 — clean in-sentence BALANCE axis (27 balance words vs 22 imbalance words)\n" + "=" * 72)
    clean = c2q_boot(lambda L: dirs[L]["clean"], lambda L: dirs[L]["fr"]["d_norm_ho"],
                     lambda L: dirs[L]["fr"]["carrier_out"], mask="whole")
    show("clean axis, whole words (PRIMARY)", clean)
    clean_all = c2q_boot(lambda L: dirs[L]["clean"], lambda L: dirs[L]["fr"]["d_norm_ho"],
                         lambda L: dirs[L]["fr"]["carrier_out"], mask="notlist")
    show("clean axis, all tokens", clean_all)
    print("  cos(original BALANCE, clean in-sentence BALANCE): "
          + "  ".join(f"L{L} {dirs[L]['cos_orig_clean']:+.3f}" for L in LAYERS))
    rng2 = np.random.default_rng(SEED_RAND + 1)
    crand = {L: [] for L in LAYERS}
    for _ in range(N_RAND):
        ws = list(rng2.choice(pool_f, len(POS_EXT) + len(NEG_EXT), replace=False))
        pw, nw = ws[:len(POS_EXT)], ws[len(POS_EXT):]
        for L in LAYERS:
            d = dirs[L]; dn = d["fr"]["d_norm_ho"]
            crand[L].append(c2q_point(L, rand_axis(L, pw, nw, d["Rf"], d["strip_f"], dn), dn, mask="whole"))
    part2 = {}
    for L in LAYERS:
        cr = np.array(crand[L]); real = clean[L]["c2q"]
        part2[L] = dict(real=real, ci=clean[L]["ci"], rand_median=float(np.median(cr)),
                        rand_p5=float(np.percentile(cr, 5)), rand_p95=float(np.percentile(cr, 95)),
                        real_percentile=float((cr < real).mean() * 100))
        r = part2[L]
        print(f"  L{L:<2} clean BALANCE {real:+.3f} | CLEAN-RANDOM axes: median {r['rand_median']:+.3f} "
              f"[5% {r['rand_p5']:+.3f}, 95% {r['rand_p95']:+.3f}]; real sits at the {r['real_percentile']:.0f}th percentile"
              f"{'   (judged)' if L in JUDGED else ''}")
    status = classify(clean)
    n_rev = sum(clean[L]["c2q"] >= DECISION_LO and clean[L]["ci"][0] > 0 for L in JUDGED)
    n_out = sum(part2[L]["real"] < part2[L]["rand_p5"] or part2[L]["real"] > part2[L]["rand_p95"] for L in JUDGED)
    if status in ("PASS", "FIRES"):
        v2 = "CLEAN_COUPLING_PRESENT"
    elif n_rev >= 2:
        v2 = "CLEAN_COUPLING_REVERSED"
    else:
        v2 = "CLEAN_COUPLING_ABSENT"
    flag = " + SPECIFIC" if (v2 != "CLEAN_COUPLING_ABSENT" and n_out >= 2) else ""
    print(f"\n  PART 2 VERDICT: {v2}{flag}   (classifier {status}; positive-and-clear at {n_rev}/3 judged layers;"
          f" outside the random 5–95% range at {n_out}/3)")
    results["part2"] = {str(L): part2[L] for L in LAYERS}
    results["part2_all_tokens"] = {str(L): clean_all[L] for L in LAYERS}
    results["part2_verdict"] = v2 + flag
    json.dump(results, open(f"{ROOT}/exp178_results_{tag}.json", "w"), indent=1)
    print(f"\n  >>> {tag}: Part 1 {v1} | Part 2 {v2}{flag} <<<")
    print(f"results saved to exp178_results_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1])
