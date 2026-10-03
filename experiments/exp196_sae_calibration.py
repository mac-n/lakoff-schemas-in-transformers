"""
exp196_sae_calibration.py — calibrating the agreement measure with SAE
features. Frozen rules: PREREG_exp196.md (commit 9a7e37c, 3 Oct 2026 01:16
IST). Written AFTER the freeze. Extends exp193's cells (same model, site,
strength, sentences) with pushes along SAE decoder rows: the 30 latents that
fire most on the sentences (USED) and 30 that never fire (UNUSED).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = "/Users/macn/Documents/embeddingexp"
sys.path.insert(0, ROOT)
from exp181_entropy_on_request import texts

N_SAE = 30
SEED = 196
SITE = 4
N_LAYERS = 24
PREREG = open(f"{ROOT}/PREREG_exp196.md").read()
for _n, _v in [("N_SAE", N_SAE), ("SEED", SEED)]:
    _m = re.search(rf"\b{_n} = ([0-9]+)", PREREG)
    assert _m and int(_m.group(1)) == _v
MEASURES = ["straight", "write_cos", "end_norm", "kl", "dent"]
SAE_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--sae-pythia-410m-65k/snapshots/*/layers.4.mlp"))[0]


def main():
    from transformer_lens import HookedTransformer
    print("exp196 — SAE calibration of the agreement measure (prereg frozen at 9a7e37c)")
    dev = "mps"
    model = HookedTransformer.from_pretrained("pythia-410m", device=dev); model.eval()
    tk = model.tokenizer; BOS = int(tk.bos_token_id)
    cfg = json.load(open(f"{SAE_DIR}/cfg.json")); assert cfg["d_in"] == model.cfg.d_model
    with safe_open(f"{SAE_DIR}/sae.safetensors", "pt") as f:
        W_dec = f.get_tensor("W_dec").float(); W_enc = f.get_tensor("encoder.weight").float()
        b_enc = f.get_tensor("encoder.bias").float(); b_dec = f.get_tensor("b_dec").float()
    k = cfg["k"]
    T = texts()
    toks = [torch.tensor([[BOS] + tk.encode(t, add_special_tokens=False)], device=dev) for t in T]
    # which latents fire on these sentences (top-k encoder on the layer-4 MLP output, as the SAE was trained)
    fires = torch.zeros(W_dec.shape[0], dtype=torch.long)
    mlp_hook = f"blocks.{SITE}.hook_mlp_out"
    for x in toks:
        with torch.no_grad():
            _, cache = model.run_with_cache(x, names_filter=[mlp_hook])
        h = cache[mlp_hook][0, 1:, :].float().cpu()
        pre = (h - b_dec) @ W_enc.T + b_enc
        top = pre.topk(k, dim=-1).indices
        for row in top:
            fires[row] += 1
    order = torch.argsort(fires, descending=True)
    used = order[:N_SAE].tolist()
    never = (fires == 0).nonzero().flatten()
    rng = np.random.default_rng(SEED)
    unused = [int(i) for i in rng.choice(never.numpy(), N_SAE, replace=False)]
    print(f"  latents firing on at least one token: {int((fires > 0).sum())} of {W_dec.shape[0]}; "
          f"USED fire on {fires[used[0]].item()}..{fires[used[-1]].item()} tokens; UNUSED fire on none", flush=True)
    dirs = {}
    for j, i in enumerate(used):
        d = W_dec[i].numpy().astype(np.float64); dirs[f"USED:{j:02d}"] = d / np.linalg.norm(d)
    for j, i in enumerate(unused):
        d = W_dec[i].numpy().astype(np.float64); dirs[f"UNUSED:{j:02d}"] = d / np.linalg.norm(d)

    site_hook = f"blocks.{SITE}.hook_resid_post"
    hooks = [f"blocks.{L}.hook_resid_post" for L in range(SITE, N_LAYERS)]
    clean_logp = []

    def measure(vec):
        out = np.zeros((len(T), len(MEASURES)))
        fwd = []
        if vec is not None:
            def hook(resid, hook):
                resid[:, 1:, :] = resid[:, 1:, :] + vec
                return resid
            fwd = [(site_hook, hook)]
        for i, x in enumerate(toks):
            with torch.no_grad(), model.hooks(fwd_hooks=fwd):
                logits, cache = model.run_with_cache(x, names_filter=hooks)
            Rs = torch.stack([cache[h][0, 1:, :].float() for h in hooks]); Wr = Rs[1:] - Rs[:-1]
            net = Wr.sum(dim=0).norm(dim=1); path = Wr.norm(dim=2).sum(dim=0)
            cosw = torch.nn.functional.cosine_similarity(Wr[:-1], Wr[1:], dim=2).mean(dim=0)
            lp = torch.log_softmax(logits[0].float(), dim=-1); ent = float((-(lp.exp() * lp).sum(dim=-1)).mean())
            if vec is None:
                clean_logp.append(lp); kl = 0.0; dent = 0.0
            else:
                cl = clean_logp[i]; kl = float((cl.exp() * (cl - lp)).sum(dim=-1).mean()); dent = ent - float((-(cl.exp() * cl).sum(dim=-1)).mean())
            out[i] = [float((net / path).mean()), float(cosw.mean()), float(Rs[-1].norm(dim=1).mean()), kl, dent]
        return out

    c193 = dict(np.load(f"{ROOT}/exp193_cells.npz"))
    base = measure(None)
    assert abs(float(base[:, 0].mean()) - float(c193["base"][:, 0].mean())) < 1e-4, "unsteered agreement differs from exp193"
    site_norm = float(c193["site_norm"])
    path_ = f"{ROOT}/exp196_cells.npz"
    cells = dict(np.load(path_)) if os.path.exists(path_) else {}
    for j, n in enumerate(dirs):
        for sign in (+1, -1):
            key = f"{n}|{sign:+d}"
            if key in cells:
                continue
            cells[key] = measure(torch.tensor(sign * site_norm * dirs[n], dtype=torch.float32, device=dev)); np.savez(path_, **cells)
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(dirs)} SAE directions done", flush=True)
    b0 = float(c193["base"][:, 0].mean())
    drop = lambda C, n: b0 - (C[f"{n}|+1"][:, 0].mean() + C[f"{n}|-1"][:, 0].mean()) / 2
    klof = lambda C, n: (C[f"{n}|+1"][:, 3].mean() + C[f"{n}|-1"][:, 3].mean()) / 2
    names193 = sorted({k.split("|")[0] for k in c193 if "|" in k})
    conc = np.array([drop(c193, n) for n in names193 if n.startswith(("SINGLE:", "PAIR:"))]); c5, c95 = np.percentile(conc, [5, 95])
    usedd = np.array([drop(cells, f"USED:{j:02d}") for j in range(N_SAE)]); unusedd = np.array([drop(cells, f"UNUSED:{j:02d}") for j in range(N_SAE)])
    up = drop(c193, "SCHEMA:UP-DOWN")
    print(f"\n=== exp196: drop in later-layer agreement (unsteered {b0:.4f}; strength 1.0 x site norm, as exp193) ===")
    print(f"  concentrated controls (exp193 SINGLE + PAIR): 5% {c5:+.4f}, 95% {c95:+.4f}")
    print(f"  USED SAE features   (30): median {np.median(usedd):+.4f} [5% {np.percentile(usedd,5):+.4f}, 95% {np.percentile(usedd,95):+.4f}]  KL median {np.median([klof(cells, f'USED:{j:02d}') for j in range(N_SAE)]):.3f}")
    print(f"  UNUSED SAE features (30): median {np.median(unusedd):+.4f} [5% {np.percentile(unusedd,5):+.4f}, 95% {np.percentile(unusedd,95):+.4f}]  KL median {np.median([klof(cells, f'UNUSED:{j:02d}') for j in range(N_SAE)]):.3f}")
    print(f"  UP-DOWN (exp193): {up:+.4f}, at the {(usedd < up).mean()*100:.0f}th percentile of USED")
    mu, mn = np.median(usedd), np.median(unusedd)
    if mu > c95 and c5 <= mn <= c95:
        v = "MEASURE_DETECTS_USE"
    elif mu > c95 and mn > c95:
        v = "UNUSED_ALSO"
    elif c5 <= mu <= c95:
        v = "USE_NOT_SPECIAL"
    else:
        v = "OTHER (USED below the control range, or UNUSED above while USED is not)"
    print(f"\n  >>> {v} <<<")
    json.dump({"verdict": v, "used_drops": usedd.tolist(), "unused_drops": unusedd.tolist(), "conc_p5": float(c5), "conc_p95": float(c95),
               "up_drop": up, "used_latents": used, "unused_latents": unused}, open(f"{ROOT}/exp196_results.json", "w"), indent=1)
    print("results saved to exp196_results.json")


if __name__ == "__main__":
    main()
