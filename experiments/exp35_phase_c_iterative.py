"""
exp35: Phase C — iterative greedy residual projection over the full canonical
MML axis set + non-Lakoff V/A/TIME anchors.

This is the "minimum basis" / empirical primitive discovery test that's been
queued since Entry 12. The procedure:

  1. Start with N candidate axes (V, A, TIME, all MML schemas).
  2. Compute each axis's "explanation power" = sum over other axes of squared
     projection onto this axis. The axis with highest explanation power is
     the most cluster-central.
  3. Subtract the winner's component from all remaining axes.
  4. Renormalize.
  5. Repeat. The order of selection IS the empirical basis ordering.

Output: discovered ordering, explained-variance at each step, and nearest
neighbors of the residuals to characterize what's left.

Also tests:
  - TIME axis vs FB axis cosine (are they essentially the same?)
  - TIME vs the UD-FB residual after V+A+EXISTENCE (does TIME capture the fourth axis?)
  - Nearest neighbors of the UD-FB triple residual to characterize it semantically
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML,
    PATH_MOTION_MML, LIGHT_DARK_MML, EXISTENCE_MML,
    FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)


print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


VALENCE_PAIRS = [
    ("pleasant", "unpleasant"), ("desirable", "undesirable"),
    ("agreeable", "disagreeable"), ("enjoyable", "distasteful"),
    ("delightful", "awful"), ("beneficial", "harmful"),
    ("wonderful", "terrible"), ("excellent", "dreadful"),
    ("favorable", "unfavorable"), ("satisfying", "frustrating"),
    ("nice", "nasty"), ("kind", "cruel"),
]
AROUSAL_PAIRS = [
    ("intense", "mild"), ("intense", "gentle"),
    ("alert", "drowsy"), ("urgent", "leisurely"),
    ("frantic", "tranquil"), ("energetic", "lethargic"),
    ("aroused", "relaxed"), ("sharp", "dull"),
    ("acute", "subtle"), ("vivid", "faint"),
    ("electric", "placid"), ("turbulent", "still"),
]

# NEW: TIME axis — temporal-sequence vocabulary trying to avoid spatial FB content.
# Honest caveat: English has so internalized TIME IS MOTION FORWARD that "pure"
# time vocabulary may not exist. These pairs lean toward sequence-position
# (recent/old, modern/ancient) rather than direction-of-motion (future/past).
TIME_PAIRS = [
    ("modern", "ancient"),
    ("modern", "antique"),
    ("contemporary", "historical"),
    ("contemporary", "archaic"),
    ("current", "former"),
    ("recent", "old"),
    ("recent", "ancient"),
    ("now", "then"),
    ("today", "yesterday"),
    ("new", "old"),                # has valence loading; residualization will handle it
    ("young", "aged"),
    ("fresh", "stale"),
    ("present", "bygone"),
    ("live", "vintage"),
    ("nowadays", "anciently"),
]


def build_axis(pairs, label=""):
    offs = []
    missing = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            missing.append((a, c))
    if missing and label:
        print(f"  [{label}] missing: {missing}")
    arr = np.stack(offs)
    raw = arr.mean(axis=0)
    return raw / np.linalg.norm(raw)


def residualize(v, axes_to_remove):
    r = v.copy()
    for a in axes_to_remove:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


print("\n=== Building axes ===")
A_val = build_axis(VALENCE_PAIRS, "V")
A_aro = build_axis(AROUSAL_PAIRS, "A")
A_time = build_axis(TIME_PAIRS, "TIME")
A_ud = build_axis(UP_DOWN_MML, "UD")
A_io = build_axis(IN_OUT_MML, "IO")
A_io_clean = build_axis(IN_OUT_MML_CLEAN, "IO_CLEAN")
A_fb = build_axis(FORWARD_BACK_MML, "FB")
A_path = build_axis(PATH_MOTION_MML, "PATH")
A_ld = build_axis(LIGHT_DARK_MML, "LD")
A_exist = build_axis(EXISTENCE_MML, "EXIST")
A_force = build_axis(FORCE_MML, "FORCE")
A_balance = build_axis(BALANCE_MML, "BAL")
A_diff = build_axis(DIFFICULTY_BURDEN_MML, "DIFF")


# =================================================================
# PRELIMINARY: TIME alignment with FB and with UD-FB residual
# =================================================================
print("\n=== Preliminary: TIME alignment with other axes ===")
print(f"  cos(TIME, FB)       = {float(A_time @ A_fb):+.4f}")
print(f"  cos(TIME, UD)       = {float(A_time @ A_ud):+.4f}")
print(f"  cos(TIME, LD)       = {float(A_time @ A_ld):+.4f}")
print(f"  cos(TIME, EXIST)    = {float(A_time @ A_exist):+.4f}")
print(f"  cos(TIME, IO_CLEAN) = {float(A_time @ A_io_clean):+.4f}")
print(f"  cos(TIME, VALENCE)  = {float(A_time @ A_val):+.4f}")
print(f"  cos(TIME, AROUSAL)  = {float(A_time @ A_aro):+.4f}")


# UD-FB residual after V+A+EXISTENCE: what's left?
print("\n=== UD-FB shared residual after V+A+EXISTENCE: what is the 'fourth axis'? ===")
ud_vae = residualize(A_ud, [A_val, A_aro, A_exist])
fb_vae = residualize(A_fb, [A_val, A_aro, A_exist])
shared_residual = (ud_vae + fb_vae) / 2
shared_residual = shared_residual / np.linalg.norm(shared_residual)

print(f"\n  cos(UD_VAE, FB_VAE)             = {float(ud_vae @ fb_vae):+.4f}  (the +0.26 residual)")
print(f"  cos(shared_residual, TIME)      = {float(shared_residual @ A_time):+.4f}")
print(f"  cos(shared_residual, PATH)      = {float(shared_residual @ A_path):+.4f}")
print(f"  cos(shared_residual, FORCE)     = {float(shared_residual @ A_force):+.4f}")
print(f"  cos(shared_residual, BAL)       = {float(shared_residual @ A_balance):+.4f}")
print(f"  cos(shared_residual, DIFF)      = {float(shared_residual @ A_diff):+.4f}")

print(f"\n  Shared residual nearest neighbors (the fourth axis):")
print(f"  Positive pole:")
for w, s in wv.similar_by_vector(shared_residual, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole:")
for w, s in wv.similar_by_vector(-shared_residual, topn=15):
    print(f"    {s:>+.4f}  {w}")


# =================================================================
# THE MAIN EVENT: iterative greedy residual projection
# =================================================================
print("\n\n" + "="*72)
print("ITERATIVE GREEDY RESIDUAL PROJECTION")
print("="*72)
print("""
Greedy procedure: at each step, find the axis whose direction best explains
the rest (highest sum of squared projections onto the others). Pick it as
the next primitive. Subtract from all others. Repeat. The order discovered
IS the empirical basis ordering.

Interpretation:
  - Axes picked first are most cluster-central / most load-bearing.
  - Axes picked late (low explanation power) are most independent / most primitive.
  - When an axis has very low residual norm after others are subtracted, it's
    a derived composite.
""")

# TIME excluded from the principled basis-discovery — only citable axes (MML +
# Warriner-style affect anchors). TIME is our own construction; mixing it with
# citable axes muddles the experimental status. If unexplained residuals
# suggest a TIME primitive after the principled run, that becomes a separate
# follow-up.
axes_init = {
    "VALENCE": A_val,
    "AROUSAL": A_aro,
    "UD": A_ud,
    "IO": A_io,
    "IO_CLEAN": A_io_clean,
    "FB": A_fb,
    "PATH": A_path,
    "LD": A_ld,
    "EXIST": A_exist,
    "FORCE": A_force,
    "BAL": A_balance,
    "DIFF": A_diff,
}

remaining = {k: v.copy() for k, v in axes_init.items()}
order = []
step = 0

while remaining:
    step += 1
    # Explanation power = sum of squared projections onto each other axis
    powers = {}
    for name, v in remaining.items():
        powers[name] = sum(
            float(other @ v) ** 2
            for other_name, other in remaining.items() if other_name != name
        )
    winner = max(powers, key=powers.get)
    winner_power = powers[winner]
    winner_norm = float(np.linalg.norm(remaining[winner]))
    order.append((winner, winner_power, winner_norm))

    print(f"  Step {step:>2}: picked {winner:>10}  explanation_power = {winner_power:>6.3f}  axis_norm = {winner_norm:.4f}")

    # Subtract winner from all remaining axes
    w_v = remaining.pop(winner)
    w_v_unit = w_v / np.linalg.norm(w_v)
    for name in list(remaining.keys()):
        v = remaining[name]
        v_new = v - float(v @ w_v_unit) * w_v_unit
        nrm = float(np.linalg.norm(v_new))
        # If norm drops below threshold, the axis is essentially explained
        if nrm < 0.05:
            print(f"        ↳ {name} norm dropped to {nrm:.4f} (essentially explained)")
        remaining[name] = v_new

print("\nDiscovered ordering:")
for i, (name, power, nrm) in enumerate(order, 1):
    print(f"  {i:>2}. {name:>10}  explanation_power={power:>6.3f}  axis_norm_at_pickup={nrm:.4f}")


# =================================================================
# Look at the residuals at each iteration to characterize what's left
# =================================================================
print("\n=== Nearest neighbors of axes at the point they were picked ===")
print("  (this shows what each axis represents AFTER prior axes have been subtracted)")

# Re-run, capturing each axis's vector at pickup time
remaining2 = {k: v.copy() for k, v in axes_init.items()}
for step_num, (winner_name, _, _) in enumerate(order[:8], 1):  # top 8
    # Compute current explanation power and find winner (should match)
    w_v = remaining2[winner_name]
    w_v_unit = w_v / np.linalg.norm(w_v)
    print(f"\n  Step {step_num} ({winner_name}) — at pickup:")
    print(f"    Positive pole:")
    for w, s in wv.similar_by_vector(w_v_unit, topn=8):
        print(f"      {s:>+.4f}  {w}")
    print(f"    Negative pole:")
    for w, s in wv.similar_by_vector(-w_v_unit, topn=8):
        print(f"      {s:>+.4f}  {w}")
    # Subtract for next iteration
    remaining2.pop(winner_name)
    for name in list(remaining2.keys()):
        v = remaining2[name]
        remaining2[name] = v - float(v @ w_v_unit) * w_v_unit


# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp35_results.npz",
    order=[(name, power, nrm) for name, power, nrm in order],
    time_axis=A_time,
    shared_residual=shared_residual,
)
print("\nSaved: exp35_results.npz")
