"""
exp179_causal_matched.py — exp168's causal attention-temperature test,
compared against token-structure-matched random axes and repeated with the
clean in-sentence BALANCE axis. Frozen rules: PREREG_exp179.md (commit
18f9d41, 2026-10-02 18:21 IST). Written AFTER the freeze.

The temperature hook, token selection and slope are exp168's
(_measure_tau / _slope_vs_entropy), generalised to many axes at once.
"""
import json
import re
import sys

import numpy as np
import torch

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from attn_entropy_lib import attn_entropy_per_query
from markedness_norm_protocol import build_word_lists, collect_residuals, COMMON, RARE
from exp161_balance_entropy_prereg import build_layer_dirs, proj_out, AXIS_WORDS
from exp165_balance_entropy_steer import ols_slope, spearman
from exp166_prompt_verify import CANDIDATE as FRESH_PROMPTS
from exp168_arrow_b_entropy_to_geometry import TAU, SEED as SEED168
from exp177_fragment_reading import token_types, strip_tools
from exp178_what_produces_coupling import POS_EXT, NEG_EXT, FRAMES

N_RAND = 300
SEED_RAND = 179
SLOPE_TOL = 0.0005
N_BOOT = 1000
DEVICE = "mps"
LAYERS = [3, 8]
TARGET = {3: -0.0083, 8: -0.0018}
PREREG = open(f"{ROOT}/PREREG_exp179.md").read()
for _name, _val in [("N_RAND", N_RAND), ("SEED_RAND", SEED_RAND), ("SLOPE_TOL", SLOPE_TOL), ("N_BOOT", N_BOOT)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
assert TAU == [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0]


def main():
    from transformer_lens import HookedTransformer
    print("exp179 — causal test against matched controls — gpt2-medium (prereg frozen at 18f9d41)")
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is True
    hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    res_bare = collect_residuals(model, LAYERS, all_words, log_every=0)
    ntok_bare = {w: int(model.to_tokens(w).shape[1] - 1) for w in all_words}
    words_f = sorted(set(all_words) | set(POS_EXT) | set(NEG_EXT))
    ntok_sp = {w: int(model.to_tokens(" " + w).shape[1] - 1) for w in words_f}
    res_frame = {}
    for w in words_f:
        accf = {L: 0.0 for L in LAYERS}
        for fr in FRAMES:
            head = fr.split("___")[0].rstrip()
            toks = model.to_tokens(fr.replace("___", w)); ptoks = model.to_tokens(head + " " + w)
            n = ptoks.shape[1]
            assert bool((toks[0, :n] == ptoks[0]).all())
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=hook_names)
            for L in LAYERS:
                accf[L] = accf[L] + cache[f"blocks.{L}.hook_resid_post"][0, n - 1, :].float().cpu().numpy()
        res_frame[w] = {L: accf[L] / len(FRAMES) for L in LAYERS}

    pairs = LAKOFF_SCHEMAS_MML["BALANCE"]
    pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
    bal_vocab = set(pos) | set(neg)
    pool = [w for w in all_words if w not in bal_vocab and w not in COMMON and w not in RARE]
    singles = [w for w in pool if ntok_bare[w] == 1]; multis = [w for w in pool if ntok_bare[w] > 1]
    s_pos = sum(ntok_bare[w] == 1 for w in pos); m_pos = len(pos) - s_pos
    s_neg = sum(ntok_bare[w] == 1 for w in neg); m_neg = len(neg) - s_neg
    pool_f = [w for w in pool if ntok_sp[w] == 1]
    list_words = set(POS_EXT) | set(NEG_EXT)
    print(f"real BALANCE token structure: positive {s_pos} single + {m_pos} split, negative {s_neg} single + {m_neg} split")

    # the same random word draws at both layers
    rng = np.random.default_rng(SEED_RAND)
    draws_m, draws_s, draws_c = [], [], []
    for _ in range(N_RAND):
        sp = list(rng.choice(singles, s_pos + s_neg, replace=False))
        mp = list(rng.choice(multis, m_pos + m_neg, replace=False))
        draws_m.append((sp[:s_pos] + mp[:m_pos], sp[s_pos:] + mp[m_pos:]))
        so = list(rng.choice(singles, 30, replace=False)); draws_s.append((so[:15], so[15:]))
        ws = list(rng.choice(pool_f, len(POS_EXT) + len(NEG_EXT), replace=False))
        draws_c.append((ws[:len(POS_EXT)], ws[len(POS_EXT):]))

    results = {}
    for L in LAYERS:
        gate_layer = L == 3
        print(f"\n{'=' * 72}\nLAYER {L} {'(PRIMARY)' if gate_layer else '(reported)'}\n{'=' * 72}")
        base = build_layer_dirs(res_bare, L, all_words, est_words, test_words)
        strip_b, _ = strip_tools(res_bare, L, all_words)
        fr = build_layer_dirs(res_frame, L, all_words, est_words, test_words)
        strip_f, mean_f = strip_tools(res_frame, L, all_words)

        def axis(pw, nw, R, strip, dn):
            raw = np.mean([R[w][L] for w in pw], axis=0) - np.mean([R[w][L] for w in nw], axis=0)
            return proj_out(strip(raw / np.linalg.norm(raw)), dn)

        raw = mean_f(POS_EXT) - mean_f(NEG_EXT)
        clean = proj_out(strip_f(raw / np.linalg.norm(raw)), fr["d_norm_ho"])
        bare_axes = np.stack([base["bal_pn"]]
                             + [axis(p, n_, res_bare, strip_b, base["d_norm_ho"]) for p, n_ in draws_m]
                             + [axis(p, n_, res_bare, strip_b, base["d_norm_ho"]) for p, n_ in draws_s])
        clean_axes = np.stack([clean] + [axis(p, n_, res_frame, strip_f, fr["d_norm_ho"]) for p, n_ in draws_c])
        score_hook = f"blocks.{L}.attn.hook_attn_scores"
        resid_hook = f"blocks.{L}.hook_resid_post"; patt_hook = f"blocks.{L}.attn.hook_pattern"
        P_bare = {}; E_all = {}; P_clean = {}; E_whole = {}
        for tau in TAU:
            def temp(scores, hook, tau=tau):
                return scores / tau
            pb, ea, pc, ew = [], [], [], []
            with model.hooks(fwd_hooks=[(score_hook, temp)]):
                for prompt in FRESH_PROMPTS:
                    toks = model.to_tokens(prompt); strs = model.to_str_tokens(prompt)
                    types = token_types(strs)
                    with torch.no_grad():
                        _, c = model.run_with_cache(toks, names_filter=[resid_hook, patt_hook])
                    resid = c[resid_hook][0].float().cpu().numpy().astype(np.float64)
                    ent = attn_entropy_per_query(c[patt_hook][0].float().cpu().numpy())
                    keep, whole = [], []
                    for q in range(1, resid.shape[0]):
                        if np.isnan(ent[q]):
                            continue
                        w = strs[q].strip().lower().strip(".,!?'\"")
                        if w in AXIS_WORDS:
                            continue
                        keep.append(q)
                        whole.append(types[q] == "WHOLE" and w not in list_words)
                    keep = np.array(keep); whole = np.array(whole)
                    U = resid[keep] / np.linalg.norm(resid[keep], axis=1, keepdims=True)
                    pb.append((U @ bare_axes.T).mean(axis=0)); ea.append(ent[keep].mean())
                    pc.append((U[whole] @ clean_axes.T).mean(axis=0)); ew.append(ent[keep][whole].mean())
            P_bare[tau] = np.stack(pb); E_all[tau] = np.array(ea)
            P_clean[tau] = np.stack(pc); E_whole[tau] = np.array(ew)
        ent_means = [float(E_all[t].mean()) for t in TAU]
        valid = all(ent_means[i] <= ent_means[i + 1] + 1e-6 for i in range(len(TAU) - 1))
        print("  achieved entropy by tau: " + "  ".join(f"{t:.2f}:{e:.3f}" for t, e in zip(TAU, ent_means))
              + f"   monotone: {'YES' if valid else 'NO'}")

        def slopes(P, E):
            xs = [E[t].mean() for t in TAU]
            Y = np.stack([P[t].mean(axis=0) for t in TAU])          # [tau, axes]
            return np.array([ols_slope(xs, Y[:, j]) for j in range(Y.shape[1])])

        def boot_ci(P, E, j, seed):
            rng_b = np.random.default_rng(seed)
            n = len(E[TAU[0]]); out = []
            for _ in range(N_BOOT):
                idx = rng_b.choice(n, n, True)
                out.append(ols_slope([E[t][idx].mean() for t in TAU], [P[t][idx, j].mean() for t in TAU]))
            return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

        sb = slopes(P_bare, E_all); sc = slopes(P_clean, E_whole)
        real, matched, single = sb[0], sb[1:1 + N_RAND], sb[1 + N_RAND:]
        clean_s, crand = sc[0], sc[1:]
        ci_real = boot_ci(P_bare, E_all, 0, SEED168 + L); ci_clean = boot_ci(P_clean, E_whole, 0, SEED168 + L)
        xs = [E_all[t].mean() for t in TAU]
        spr = spearman(xs, [P_bare[t][:, 0].mean() for t in TAU])
        gate = abs(real - TARGET[L]) <= SLOPE_TOL and valid
        print(f"  real BALANCE slope {real:+.4f} CI[{ci_real[0]:+.4f},{ci_real[1]:+.4f}] Spearman {spr:+.2f}"
              f"   exp168 had {TARGET[L]:+.4f}  ->  REPLICATION GATE {'PASS' if gate else 'FAIL'}")
        if not gate:
            print("STOP: harness drift. Nothing below was computed.")
            sys.exit(2)
        pct = float((matched < real).mean() * 100)
        print(f"  MATCHED fake axes (same token split): median {np.median(matched):+.4f} "
              f"[5% {np.percentile(matched, 5):+.4f}, 95% {np.percentile(matched, 95):+.4f}], "
              f"{(matched < 0).mean() * 100:.0f}% negative; real BALANCE sits at the {pct:.0f}th percentile")
        print(f"  SINGLE-ONLY fake axes: median {np.median(single):+.4f} "
              f"[5% {np.percentile(single, 5):+.4f}, 95% {np.percentile(single, 95):+.4f}], {(single < 0).mean() * 100:.0f}% negative")
        cpct = float((crand < clean_s).mean() * 100)
        print(f"  CLEAN in-sentence BALANCE slope {clean_s:+.4f} CI[{ci_clean[0]:+.4f},{ci_clean[1]:+.4f}]"
              f" | CLEAN-RANDOM: median {np.median(crand):+.4f} [5% {np.percentile(crand, 5):+.4f}, "
              f"95% {np.percentile(crand, 95):+.4f}]; clean BALANCE sits at the {cpct:.0f}th percentile")
        if real < np.percentile(matched, 5):
            v = "CAUSAL_BALANCE_SPECIFIC"
        elif np.median(matched) < 0 and (matched < 0).mean() >= 0.75:
            v = "CAUSAL_TOKEN_STRUCTURE_SUFFICIENT"
        else:
            v = "CAUSAL_NEITHER"
        outside = clean_s < np.percentile(crand, 5) or clean_s > np.percentile(crand, 95)
        ci_excl = not (ci_clean[0] <= 0 <= ci_clean[1])
        vc = "CLEAN_CAUSAL_SPECIFIC" if (outside and ci_excl) else "CLEAN_CAUSAL_NOT_SPECIFIC"
        print(f"\n  {'VERDICT' if gate_layer else 'at this layer the rule would read'}: {v} | {vc}")
        results[str(L)] = dict(real=float(real), ci_real=ci_real, matched_median=float(np.median(matched)),
                               matched_p5=float(np.percentile(matched, 5)), matched_p95=float(np.percentile(matched, 95)),
                               matched_neg_share=float((matched < 0).mean()), real_percentile=pct,
                               single_median=float(np.median(single)), single_p5=float(np.percentile(single, 5)),
                               single_p95=float(np.percentile(single, 95)), clean=float(clean_s), ci_clean=ci_clean,
                               crand_median=float(np.median(crand)), crand_p5=float(np.percentile(crand, 5)),
                               crand_p95=float(np.percentile(crand, 95)), clean_percentile=cpct,
                               verdict=v, clean_verdict=vc, entropy_by_tau=ent_means)
    json.dump(results, open(f"{ROOT}/exp179_results.json", "w"), indent=1)
    print("\nresults saved to exp179_results.json")


if __name__ == "__main__":
    main()
