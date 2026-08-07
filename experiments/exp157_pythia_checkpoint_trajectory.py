"""
exp157_pythia_checkpoint_trajectory.py — when, during Pythia 410M's
training, do the sink, the norm physiology, and the concept-physiology
coupling each emerge? (Niamh's suggestion, 2026-06-11.)

This is the question the three-model table (Pythia/GPT-2/Llama) cannot
answer. Pythia publishes full training checkpoints precisely for this.
After exp156 (G2: coupling follows the Pythia family, not the norm
type), the live hypothesis space is about HOW Pythia's coincidence of
concept and physiology came to be:
  - two-stage recruitment: a universal, GPT-2/Llama-grade sink
    (~−0.15) appears first WITHOUT norm coupling; the norm physiology
    develops later and amplifies it to Pythia's −0.3+;
  - co-emergence: sink and coupling arrive together — the sink is born
    norm-implemented;
  - coupling-first: the physiological axis exists before any
    markedness geometry uses it.

Checkpoints: steps [0, 512, 4000, 16000, 64000, 143000] (final =
main branch; doubles as a parity check vs exp138/exp154 values).
Protocol per checkpoint: exp154c held-out d_norm standard
(markedness_norm_protocol), layers [4,8,12,16,20], decision layers
[8,12,16,20]; plus corr(zipf, residual norm) per layer (wordfreq).

NB TransformerLens bug + workaround: TL's Pythia checkpoint branch
passes token=os.environ HF token without an empty-string guard →
"Illegal header value b'Bearer '" if HF_TOKEN unset. We set HF_TOKEN
from the stored hub token before importing TL machinery.

PRE-REGISTRATION (2026-06-11, before running; this Claude):
  Committed prediction: TWO-STAGE RECRUITMENT.
    P1 step 0 (random init): nothing — |sink| < 0.05, |cos| < 0.2,
       |corr(zipf,norm)| < 0.2.
    P2 an attenuated sink (mean <= −0.10 at decision layers) appears
       by step 4000, while coupling is still < +0.3.
    P3 coupling crosses +0.5 between steps 4000 and 64000, and
       corr(zipf, norm) >= +0.5 at or before the coupling's arrival
       (physiological regime precedes semantic alignment).
    P4 after coupling arrives, the sink deepens by >= 0.08 vs its
       pre-coupling value (recruitment amplifies).
  Decision rule (on mean-over-decision-layer trajectories), via
  trajectory_verdict() — synthetic-tested at script start:
    T1 two-stage recruitment (sink first without coupling, then
       coupling with >= 0.08 deepening): the universal concept geometry
       is RECRUITED by the developing norm physiology. Part 2's story.
    T2 co-emergence (thresholds first crossed at same checkpoint):
       born together; "recruitment" is wrong, write "joint development".
    T3 coupling-first: physiology precedes concept geometry — the
       sink grows ON the norm axis from the start.
    T4 other/none (incl. sink-never or coupling-never): report raw
       trajectory; final-checkpoint parity tells us whether the run
       can be trusted at all.
"""

import gc
import os

import numpy as np
import torch
from huggingface_hub import get_token

os.environ["HF_TOKEN"] = get_token() or ""

from transformer_lens import HookedTransformer
from wordfreq import zipf_frequency

from markedness_norm_protocol import (
    build_word_lists, collect_residuals, analyze_layer,
)

CHECKPOINTS = [0, 512, 4000, 16000, 64000, 143000]
LAYERS = [4, 8, 12, 16, 20]
DECISION_LAYERS = [8, 12, 16, 20]


def trajectory_verdict(traj, sink_thresh=-0.10, cos_lo=0.3, cos_hi=0.5,
                       deepen=0.08):
    """traj: list of (step, sink_mean, cos_mean), ordered by step.
    Returns (code, detail). Sink is NEGATIVE; deeper = more negative."""
    first_sink = next((i for i, (s, sk, c) in enumerate(traj)
                       if sk <= sink_thresh), None)
    first_cos = next((i for i, (s, sk, c) in enumerate(traj)
                      if c >= cos_hi), None)
    if first_sink is None and first_cos is None:
        return "T4", "neither sink nor coupling ever crosses threshold"
    if first_cos is None:
        return "T4", "sink without coupling at every checkpoint (Llama-like forever)"
    if first_sink is None:
        return "T3", "coupling crosses without a sink ever forming"
    if first_sink < first_cos:
        pre = traj[first_sink]; post = traj[first_cos]
        if pre[2] < cos_lo and (pre[1] - post[1]) >= deepen:
            return "T1", (f"sink at step {pre[0]} (sink {pre[1]:+.3f}, cos "
                          f"{pre[2]:+.2f}); coupling at step {post[0]}; "
                          f"deepened {pre[1]-post[1]:+.3f}")
        if pre[2] < cos_lo:
            return "T1-weak", "sink-first without coupling, but deepening < 0.08"
        return "T2", "sink technically first but coupling already partial (>= 0.3)"
    if first_sink == first_cos:
        return "T2", f"both cross at step {traj[first_sink][0]}"
    return "T3", (f"coupling at step {traj[first_cos][0]} precedes "
                  f"sink at step {traj[first_sink][0]}")


def selftest_trajectory_verdict():
    t1 = [(0, 0.0, 0.0), (512, -0.15, 0.1), (4000, -0.32, 0.6)]
    assert trajectory_verdict(t1)[0] == "T1", trajectory_verdict(t1)
    t2 = [(0, 0.0, 0.0), (512, -0.05, 0.2), (4000, -0.30, 0.7)]
    assert trajectory_verdict(t2)[0] == "T2", trajectory_verdict(t2)
    t3 = [(0, 0.0, 0.0), (512, -0.05, 0.6), (4000, -0.30, 0.7)]
    assert trajectory_verdict(t3)[0] == "T3", trajectory_verdict(t3)
    t4 = [(0, 0.0, 0.0), (512, -0.15, 0.1), (4000, -0.16, 0.2)]
    assert trajectory_verdict(t4)[0] == "T4", trajectory_verdict(t4)
    t1w = [(0, 0.0, 0.0), (512, -0.15, 0.1), (4000, -0.18, 0.6)]
    assert trajectory_verdict(t1w)[0] == "T1-weak", trajectory_verdict(t1w)
    t2b = [(0, 0.0, 0.0), (512, -0.15, 0.4), (4000, -0.32, 0.6)]
    assert trajectory_verdict(t2b)[0] == "T2", trajectory_verdict(t2b)
    print("selftest_trajectory_verdict: all branches fire correctly.")


print("Running trajectory-verdict selftest first (synthetic-test convention)...")
selftest_trajectory_verdict()

all_words, est_words, test_words = build_word_lists()
zipf = np.array([zipf_frequency(w, "en") for w in all_words])
print(f"Word split: {len(all_words)} total = {len(est_words)} estimation "
      f"+ {len(test_words)} held-out test")

traj = []
for step in CHECKPOINTS:
    print(f"\n{'#' * 78}\n# checkpoint step {step}\n{'#' * 78}")
    if step == 143000:
        model = HookedTransformer.from_pretrained("pythia-410m", device="mps")
    else:
        model = HookedTransformer.from_pretrained(
            "pythia-410m", checkpoint_value=step, device="mps")
    model.eval()
    print(f"Collecting residuals ({len(all_words)} words, layers {LAYERS})...")
    residuals = collect_residuals(model, LAYERS, all_words, log_every=200)
    del model
    gc.collect(); torch.mps.empty_cache()

    rng = np.random.default_rng(7)
    rows = {}
    print(f"  {'L':>3} {'sink':>7} {'ho_ret%':>8} {'cos(BAL,dn)':>12} "
          f"{'zipf-norm':>10} {'UN_sink':>8}")
    for L in LAYERS:
        r = analyze_layer(residuals, L, all_words, est_words, test_words, rng)
        nv = np.array([float(np.linalg.norm(residuals[w][L])) for w in all_words])
        r["zipf_norm"] = float(np.corrcoef(zipf, nv)[0, 1])
        rows[L] = r
        ret = 100 * r["infl_ho"] / r["infl_before"] if abs(r["infl_before"]) > 0.02 else float("nan")
        print(f"  {L:>3} {r['infl_before']:>+7.3f} {ret:>7.0f}% "
              f"{r['cos_bal_dnorm']:>+12.3f} {r['zipf_norm']:>+10.3f} "
              f"{r['sink_before']['UN_negation']:>+8.3f}")
    sink_mean = float(np.mean([rows[L]["infl_before"] for L in DECISION_LAYERS]))
    cos_mean = float(np.mean([rows[L]["cos_bal_dnorm"] for L in DECISION_LAYERS]))
    zn_mean = float(np.mean([rows[L]["zipf_norm"] for L in DECISION_LAYERS]))
    print(f"  decision-layer means: sink {sink_mean:+.3f}, cos {cos_mean:+.3f}, "
          f"corr(zipf,norm) {zn_mean:+.3f}")
    traj.append((step, sink_mean, cos_mean, zn_mean))
    del residuals
    gc.collect()

print("\n" + "=" * 78)
print("TRAJECTORY (decision-layer means)")
print("=" * 78)
print(f"  {'step':>8} {'sink':>8} {'cos(BAL,d_norm)':>16} {'corr(zipf,norm)':>16}")
for step, sk, c, zn in traj:
    print(f"  {step:>8} {sk:>+8.3f} {c:>+16.3f} {zn:>+16.3f}")

print("\n" + "=" * 78)
print("VERDICT vs pre-registered decision rule")
print("=" * 78)
code, detail = trajectory_verdict([(s, sk, c) for s, sk, c, _ in traj])
zn_first = next((s for s, sk, c, zn in traj if zn >= 0.5), None)
cos_first = next((s for s, sk, c, zn in traj if c >= 0.5), None)
print(f"  {code}: {detail}")
print(f"  P3 physiology-precedes-alignment: corr(zipf,norm) >= +0.5 first at "
      f"step {zn_first}; coupling >= +0.5 first at step {cos_first}")
final = traj[-1]
print(f"  parity check (step 143000 should match exp154/exp151-era values: "
      f"sink ~ -0.29, cos ~ +0.72): sink {final[1]:+.3f}, cos {final[2]:+.3f}")
