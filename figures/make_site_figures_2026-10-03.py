"""Figures for the updated glassnest.ai/embodied page (3 Oct 2026).
Built only from the clean, pre-registered results: exp187 (Pythia 1.4B),
exp194 (GPT-2-medium), exp192 (schema structure on clean axes)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/macn/Documents/embeddingexp"
OUT = "/Users/macn/Documents/website/embodied/figures"
ACC, GREY, INK = "#1f8a80", "#b9bfc3", "#1c1f22"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Figure A: HAPPY IS UP, clean steering vs random-word directions
r187 = json.load(open(f"{ROOT}/exp187_results.json"))
r194 = json.load(open(f"{ROOT}/exp194_results.json"))
r203p = json.load(open(f"{ROOT}/exp203_results_pythia.json")); r203g = json.load(open(f"{ROOT}/exp203_results_gpt2.json"))
# affect from exp203 (scored without "uplifted" / "low"); status and quantity unchanged from exp187 / exp194
for eff_, p95_, src in [(r187["effects"]["CLEAN_UP"], r187["null_p95"], r203p)] + [(r194[L]["effects"]["CLEAN_UP"], r194[L]["null_p95"], r203g[L]) for L in ("6", "12", "18")]:
    eff_["affect"] = src["up"]; p95_["affect"] = src["p95"]
cells = [("Pythia 1.4B\nlayer 12", r187["effects"]["CLEAN_UP"], r187["null_p95"]),
         ("GPT-2 medium\nlayer 6", r194["6"]["effects"]["CLEAN_UP"], r194["6"]["null_p95"]),
         ("GPT-2 medium\nlayer 12", r194["12"]["effects"]["CLEAN_UP"], r194["12"]["null_p95"]),
         ("GPT-2 medium\nlayer 18", r194["18"]["effects"]["CLEAN_UP"], r194["18"]["null_p95"])]
measures = [("affect", "happier", ACC), ("status", "higher status", "#3b6ea8"), ("widget_quantity", "more (quantity)", "#d08a2e")]
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
x = np.arange(len(cells)); w = 0.26
for k, (m, lab, col) in enumerate(measures):
    vals = [eff[m] / p95[m] for _, eff, p95 in cells]
    ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab, edgecolor="white")
    for xi, v in zip(x + (k - 1) * w, vals):
        if v <= 1:
            ax.bar(xi, v, w * 0.92, color="white", alpha=0.55, edgecolor="none")
ax.axhline(1, color=INK, lw=1, ls="--", label="95% of random-word directions\nfall below this line")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cells])
ax.set_ylabel("effect of steering UP\n(÷ random-word 95th percentile)")
ax.set_ylim(min(0, ax.get_ylim()[0]), 2.6)
ax.set_title("Steering a clean UP direction, against 20–30 directions built from random words", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/clean_happy_is_up.png", facecolor="white"); plt.close(fig)

# ---------- Figure B: predicted vs unpredicted couplings per layer, clean axes, with scrambled-schema null
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("e192", f"{ROOT}/exp192_structure_clean.py")
src = open(f"{ROOT}/exp192_structure_clean.py").read().split("m2, m3, mats, pm, um = metrics(lists)")[0]
ns = {}
exec(compile(src, "exp192_head", "exec"), ns)
metrics, lists, NAMES, PREDICTED, NL = ns["metrics"], ns["lists"], ns["NAMES"], ns["PREDICTED"], ns["NL"]
_, _, mats, _, _ = metrics(lists)
iu = np.triu_indices(8, 1)
pi = {(NAMES.index(a), NAMES.index(b)) for a, b in PREDICTED} | {(NAMES.index(b), NAMES.index(a)) for a, b in PREDICTED}
is_pred = np.array([(i, j) in pi for i, j in zip(*iu)])
per_layer = lambda M: (M[:, iu[0], iu[1]][:, is_pred].mean(1), M[:, iu[0], iu[1]][:, ~is_pred].mean(1))
pred, unp = per_layer(mats)
pool = sorted({w for a, b in lists.values() for w in a + b}); sizes = {n: (len(a), len(b)) for n, (a, b) in lists.items()}
rng = np.random.default_rng(192); nulls = []
for k in range(100):
    perm = list(rng.permutation(pool)); c = 0; fake = {}
    for n in NAMES:
        a, b = sizes[n]; fake[n] = (perm[c:c + a], perm[c + a:c + a + b]); c += a + b
    p_, u_ = per_layer(metrics(fake)[2]); nulls.append(p_ - u_)
nulls = np.array(nulls); lo, hi = np.percentile(nulls, [5, 95], axis=0)
L = np.arange(NL)
fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)
ax.fill_between(L, lo, hi, color=GREY, alpha=0.45, lw=0, label="same words scrambled into fake schemas (5–95%)")
ax.plot(L, pred - unp, color=ACC, lw=2.4, label="real schemas: predicted pairs minus unpredicted pairs")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xlabel("layer (Pythia 410M)"); ax.set_ylabel("mean cosine difference")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
ax.set_title("The six couplings the theory predicts, against the 22 it doesn't", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/clean_schema_couplings.png", facecolor="white"); plt.close(fig)
print("figures written; per-layer real minus null-95th min:", float((pred - unp - hi).min()))
