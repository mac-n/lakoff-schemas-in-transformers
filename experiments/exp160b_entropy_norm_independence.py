"""
exp160b_entropy_norm_independence.py — is attention entropy just NORM
again? (Niamh's catch, 2026-06-11: the BALANCE-entropy lead may be
"guessable" from norm — norm as attention density.)

The day's pattern: every candidate physiology collapses onto norm.
exp158/159: MLP load ≈ ±norm (|cos(d_load,d_norm)|≈0.99). If attention
entropy is likewise norm-redundant, the exp160 BALANCE-entropy lead is
the BALANCE-norm story re-expressed, and exp161 would only re-derive
exp154. Decisive test, exactly parallel to exp159's load-vs-norm:
  (1) corr(entropy, norm) across prompt tokens, per layer, per model
      (the "norm as attention density" scalar test).
  (2) cos(d_entropy, d_norm): build d_entropy as the covariance
      direction of token entropy with unit residuals, d_norm likewise
      from token norm; both stripped. |cos|→1 means entropy IS the norm
      axis (like load was); |cos| small means entropy is an independent
      physiology and the exp160 BALANCE-entropy lead is about attention,
      not norm.
  (3) context: per layer, partial corr(BALANCE, entropy | norm) and
      cos(BALANCE, d_norm) side by side — esp. GPT-2, where BALANCE
      shows entropy-coupling but NO norm-coupling (exp156).
All token-level; entropy normalised by log(n_keys); position controlled
where it's a covariate.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction (honest, mixed): entropy is PARTLY norm in Pythia
  but NOT collinear like load was.
    P1 |corr(entropy, norm)| and |cos(d_entropy, d_norm)|: substantial
       in Pythia (0.4-0.8) — Niamh's intuition partly right — but well
       below load's 0.99 (entropy retains independent variance).
    P2 in GPT-2, |cos(d_entropy, d_norm)| is LOWER than Pythia, so the
       GPT-2 BALANCE-entropy signal is NOT norm-mediated and the
       multiple-realizability lead survives.
  Decision rule (decision layers per model, as exp160):
    K1 |cos(d_entropy,d_norm)| >= 0.9 at all decision layers in ALL
       models: entropy IS norm (third hat). Kill the attention-entropy
       avenue entirely; exp160's BALANCE-entropy was norm leakage
       through an imperfect linear control. Niamh right.
    K2 |cos(d_entropy,d_norm)| < 0.5 in GPT-2 AND BALANCE-entropy partial
       survives there: entropy is independent in GPT-2; the
       multiple-realizability lead (exp161) is live.
    K3 intermediate: entropy partially independent; exp161 must
       orthogonalise entropy against norm explicitly before any claim.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from transformer_lens import HookedTransformer

from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import SCHEMA_NAMES, COMMON, RARE, corrf
from attn_entropy_lib import (
    MODELS, PROMPTS, schema_words, attn_entropy_per_query, partial_corr,
)


def cov_dir(units, scalar_by_idx, idxs, strip):
    sv = np.array([scalar_by_idx[i] for i in idxs], float)
    z = (sv - sv.mean()) / sv.std()
    d = np.sum([z[k] * units[i] for k, i in enumerate(idxs)], axis=0)
    return strip(d / np.linalg.norm(d))


vocab = schema_words()
print("exp160b — is attention entropy independent of residual norm?")

for tag, cfg in MODELS.items():
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    model = HookedTransformer.from_pretrained(cfg["repo"], device="mps")
    model.eval()
    LAYERS = cfg["layers"]
    rhooks = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
    phooks = [f"blocks.{L}.attn.hook_pattern" for L in LAYERS]

    # isolated-word residuals -> aniso/freq strip + BALANCE/schema dirs
    wres = {}
    for w in vocab:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(w), names_filter=rhooks)
        wres[w] = {L: c[f"blocks.{L}.hook_resid_post"][0, -1, :].float().cpu().numpy()
                   for L in LAYERS}
    stripper, schema_d = {}, {}
    for L in LAYERS:
        aniso = np.stack([wres[w][L] for w in vocab]).mean(0)
        aniso /= np.linalg.norm(aniso)
        fr = (np.mean([wres[w][L] for w in COMMON], 0)
              - np.mean([wres[w][L] for w in RARE], 0)); fr /= np.linalg.norm(fr)
        fro = fr - (fr @ aniso) * aniso; fro /= np.linalg.norm(fro)

        def mk(aniso, fro):
            def strip(d):
                d = d - (d @ aniso) * aniso
                d = d - (d @ fro) * fro
                return d / np.linalg.norm(d)
            return strip
        strip = mk(aniso, fro)
        stripper[L] = strip
        sd = {}
        for sn in ("BALANCE",):
            pairs = LAKOFF_SCHEMAS_MML[sn]
            pos = sorted(set(p[0] for p in pairs)); neg = sorted(set(p[1] for p in pairs))
            raw = np.mean([wres[w][L] for w in pos], 0) - np.mean([wres[w][L] for w in neg], 0)
            sd[sn] = strip(raw / np.linalg.norm(raw))
        schema_d[L] = sd

    # prompt tokens
    tok = {L: dict(unit=[], ent=[], pos=[], norm=[], bal=[]) for L in LAYERS}
    for prompt in PROMPTS:
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, names_filter=rhooks + phooks)
        for L in LAYERS:
            resid = c[f"blocks.{L}.hook_resid_post"][0].float().cpu().numpy()
            ent = attn_entropy_per_query(c[f"blocks.{L}.attn.hook_pattern"][0].float().cpu().numpy())
            for q in range(1, resid.shape[0]):
                if np.isnan(ent[q]):
                    continue
                r = resid[q]; nrm = np.linalg.norm(r); u = r / nrm
                tok[L]["unit"].append(u); tok[L]["ent"].append(ent[q])
                tok[L]["pos"].append(q); tok[L]["norm"].append(nrm)
                tok[L]["bal"].append(float(u @ schema_d[L]["BALANCE"]))

    print(f"  {'L':>3} {'corr(ent,norm)':>15} {'cos(d_ent,d_norm)':>18} "
          f"{'partial(BAL,ent|norm)':>22} {'cos(BAL,d_norm)':>16}")
    for L in LAYERS:
        t = tok[L]
        units = t["unit"]; idxs = list(range(len(units)))
        ent = np.array(t["ent"]); norm = np.array(t["norm"]); bal = np.array(t["bal"])
        r_en = corrf(ent, norm)
        d_ent = cov_dir(units, dict(enumerate(ent)), idxs, stripper[L])
        d_nrm = cov_dir(units, dict(enumerate(norm)), idxs, stripper[L])
        cos_en = float(d_ent @ d_nrm)
        p_bal = partial_corr(bal, ent, [norm, np.array(t["pos"], float)])
        cos_bal_dnorm = float(schema_d[L]["BALANCE"] @ d_nrm)
        print(f"  {L:>3} {r_en:>+15.3f} {cos_en:>+18.3f} {p_bal:>+22.3f} "
              f"{cos_bal_dnorm:>+16.3f}")
    del model, wres
    gc.collect(); torch.mps.empty_cache()

print("\nRead: |cos(d_ent,d_norm)|→1 ⇒ entropy IS norm (kill avenue, K1).")
print("     GPT-2 low cos + BALANCE-entropy partial surviving ⇒ K2 (lead live).")
