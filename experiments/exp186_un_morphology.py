"""
exp186_un_morphology.py — is the surviving causal effect about "un-" words?
Frozen rules: PREREG_exp186.md (commit 0b2c8ad, 2026-10-02 19:55 IST).
Written AFTER the freeze. The temperature loop is exp179's (itself
exp168's), with different axis families. CPU.
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
from exp165_balance_entropy_steer import ols_slope
from exp166_prompt_verify import CANDIDATE as FRESH_PROMPTS
from exp168_arrow_b_entropy_to_geometry import TAU
from exp177_fragment_reading import strip_tools

N_RAND = 300
SEED_RAND = 186
SLOPE_TOL = 0.0005
DEVICE = "cpu"
LAYERS = [3, 8]
TARGET = {3: -0.0083, 8: -0.0018}
BASES = """happy able fair kind clear certain usual likely safe sure true wise clean common pleasant real easy seen
done used paid told changed finished married expected natural official popular healthy friendly aware necessary
important reasonable available acceptable employed""".split()
UN_PAIRS = ["balanced", "equal", "even", "stable"]
PREREG = open(f"{ROOT}/PREREG_exp186.md").read()
_flat = " ".join(PREREG.split())
for _name, _val in [("N_RAND", N_RAND), ("SEED_RAND", SEED_RAND), ("SLOPE_TOL", SLOPE_TOL)]:
    _m = re.search(rf"\b{_name} = ([0-9]+(?:\.[0-9]+)?)", PREREG)
    assert _m and float(_m.group(1)) == float(_val), f"{_name} drifted from prereg"
assert len(BASES) == 38 and all(f"{b} / un{b}" in _flat for b in BASES), "un- pool drifted from prereg"


def main():
    from transformer_lens import HookedTransformer
    print("exp186 — un- morphology control for the causal effect — gpt2-medium (prereg frozen at 0b2c8ad)")
    all_words, est_words, test_words = build_word_lists()
    model = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE); model.eval()
    assert model.cfg.default_prepend_bos is True
    un_words = BASES + ["un" + b for b in BASES]
    res = collect_residuals(model, LAYERS, sorted(set(all_words) | set(un_words)), log_every=0)
    ntok = {w: int(model.to_tokens(w).shape[1] - 1) for w in res}
    assert all(ntok[b] == 1 and ntok["un" + b] > 1 for b in BASES), "a base is split or an un- form is one token"
    pairs = LAKOFF_SCHEMAS_MML["BALANCE"]
    pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
    assert all(b in pos and "un" + b in neg for b in UN_PAIRS)
    pos_nu = [w for w in pos if w not in UN_PAIRS]; neg_nu = [w for w in neg if w not in ["un" + b for b in UN_PAIRS]]
    bal_vocab = set(pos) | set(neg)
    pool = [w for w in all_words if w not in bal_vocab and w not in COMMON and w not in RARE]
    singles = [w for w in pool if ntok[w] == 1]; multis = [w for w in pool if ntok[w] > 1]
    sp_ = sum(ntok[w] == 1 for w in pos_nu); mp_ = len(pos_nu) - sp_
    sn_ = sum(ntok[w] == 1 for w in neg_nu); mn_ = len(neg_nu) - sn_
    print(f"BALANCE without the four un- pairs: {len(pos_nu)} against {len(neg_nu)}; token split"
          f" positive {sp_} single + {mp_} split, negative {sn_} single + {mn_} split")
    rng = np.random.default_rng(SEED_RAND)
    draws_un = [list(rng.choice(BASES, 15, replace=False)) for _ in range(N_RAND)]
    draws_m = []
    for _ in range(N_RAND):
        s = list(rng.choice(singles, sp_ + sn_, replace=False)); m = list(rng.choice(multis, mp_ + mn_, replace=False))
        draws_m.append((s[:sp_] + m[:mp_], s[sp_:] + m[mp_:]))

    results = {}
    for L in LAYERS:
        primary = L == 3
        print(f"\n{'=' * 72}\nLAYER {L} {'(PRIMARY)' if primary else '(reported)'}\n{'=' * 72}")
        base = build_layer_dirs({w: res[w] for w in all_words}, L, all_words, est_words, test_words)
        strip_b, _ = strip_tools({w: res[w] for w in all_words}, L, all_words)
        dn = base["d_norm_ho"]

        def axis(pw, nw):
            raw = np.mean([res[w][L] for w in pw], axis=0) - np.mean([res[w][L] for w in nw], axis=0)
            return proj_out(strip_b(raw / np.linalg.norm(raw)), dn)

        A = np.stack([base["bal_pn"], axis(pos_nu, neg_nu)]
                     + [axis(d, ["un" + b for b in d]) for d in draws_un]
                     + [axis(p, n) for p, n in draws_m])
        score_hook = f"blocks.{L}.attn.hook_attn_scores"
        resid_hook = f"blocks.{L}.hook_resid_post"; patt_hook = f"blocks.{L}.attn.hook_pattern"
        P, E = {}, {}
        for tau in TAU:
            def temp(scores, hook, tau=tau):
                return scores / tau
            pb, ea = [], []
            with model.hooks(fwd_hooks=[(score_hook, temp)]):
                for prompt in FRESH_PROMPTS:
                    toks = model.to_tokens(prompt); strs = model.to_str_tokens(prompt)
                    with torch.no_grad():
                        _, c = model.run_with_cache(toks, names_filter=[resid_hook, patt_hook])
                    resid = c[resid_hook][0].float().cpu().numpy().astype(np.float64)
                    ent = attn_entropy_per_query(c[patt_hook][0].float().cpu().numpy())
                    keep = [q for q in range(1, resid.shape[0]) if not np.isnan(ent[q])
                            and strs[q].strip().lower().strip(".,!?'\"") not in AXIS_WORDS]
                    keep = np.array(keep)
                    U = resid[keep] / np.linalg.norm(resid[keep], axis=1, keepdims=True)
                    pb.append((U @ A.T).mean(axis=0)); ea.append(ent[keep].mean())
            P[tau] = np.stack(pb); E[tau] = np.array(ea)
            print(f"  tau {tau} done", flush=True)
        xs = [E[t].mean() for t in TAU]
        Y = np.stack([P[t].mean(axis=0) for t in TAU])
        sl = np.array([ols_slope(xs, Y[:, j]) for j in range(Y.shape[1])])
        real, no_un = sl[0], sl[1]
        un_ax = sl[2: 2 + N_RAND]; matched = sl[2 + N_RAND:]
        gate = abs(real - TARGET[L]) <= SLOPE_TOL
        print(f"  real BALANCE slope {real:+.4f} (exp168 / exp179 had {TARGET[L]:+.4f}) -> REPLICATION GATE {'PASS' if gate else 'FAIL'}")
        if not gate:
            print("STOP: harness drift."); sys.exit(2)
        p5u, p95u = np.percentile(un_ax, [5, 95]); p5m, p95m = np.percentile(matched, [5, 95])
        print(f"  UN-AXES (15 word / un-word pairs, unrelated concepts): median {np.median(un_ax):+.4f} "
              f"[5% {p5u:+.4f}, 95% {p95u:+.4f}], {(un_ax < 0).mean() * 100:.0f}% negative; "
              f"real BALANCE sits at the {(un_ax < real).mean() * 100:.0f}th percentile")
        print(f"  BALANCE without its four un- pairs: slope {no_un:+.4f} | MATCHED_NO_UN: median {np.median(matched):+.4f} "
              f"[5% {p5m:+.4f}, 95% {p95m:+.4f}]; sits at the {(matched < no_un).mean() * 100:.0f}th percentile")
        if not real < p5u:
            v = "NEGATION_SUFFICIENT"
        elif no_un < p5m:
            v = "SURVIVES_WITHOUT_NEGATION"
        else:
            v = "UNRESOLVED"
        print(f"\n  {'VERDICT' if primary else 'at this layer the rule would read'}: {v}")
        results[str(L)] = dict(real=float(real), no_un=float(no_un), un_median=float(np.median(un_ax)),
                               un_p5=float(p5u), un_p95=float(p95u), un_neg_share=float((un_ax < 0).mean()),
                               real_pct_in_un=float((un_ax < real).mean() * 100),
                               matched_median=float(np.median(matched)), matched_p5=float(p5m), matched_p95=float(p95m),
                               no_un_pct=float((matched < no_un).mean() * 100), verdict=v)
    json.dump(results, open(f"{ROOT}/exp186_results.json", "w"), indent=1)
    print("\nresults saved to exp186_results.json")


if __name__ == "__main__":
    main()
