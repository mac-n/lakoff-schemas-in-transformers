"""
exp38: Test SUCCESS/FAILURE (concrete outcome-evaluation) as a candidate for
the underlying predictive-processing primitive.

Niamh's reframe: COHERENCE was too abstract. The actual axis we're looking
for is operationalized through concrete outcome-evaluation vocabulary —
win/lose, succeed/fail, score/miss, correct/incorrect, reward/punishment,
gain/loss. These describe states where action-outcomes are evaluated against
expected/preferred outcomes — exactly the predictive-processing structural
axis (was the prediction confirmed or violated?).

Compared to COHERENCE: more concrete, more behaviorally grounded, more
directly about action-outcome-evaluation. May fit the UD-FB residual better
because the residual was about confidence/hope/achievement/winning vocabulary.

Tests:
  - Build SUCCESS/FAILURE from concrete outcome pairs avoiding UD/IO/LD/FB/IO vocabulary
  - Compare to COHERENCE: are they the same axis or different?
  - V/A decomposition — how much is just valence?
  - Alignment with UD-FB residual (the +0.26 residual after V+A+EXIST)
  - Triple residualization: does V+A+SUCCESS make UD-FB cosine collapse?
  - Nearest neighbors to verify semantic shape
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, LIGHT_DARK_MML,
    EXISTENCE_MML, DIFFICULTY_BURDEN_MML, PATH_MOTION_MML,
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

# COHERENCE for comparison (from exp37)
COHERENCE_PAIRS = [
    ("coherent", "incoherent"), ("consistent", "inconsistent"),
    ("aligned", "misaligned"), ("ordered", "disordered"),
    ("organized", "disorganized"), ("harmonious", "discordant"),
    ("predictable", "surprising"), ("predictable", "unpredictable"),
    ("expected", "unexpected"), ("ordinary", "anomalous"),
    ("regular", "irregular"), ("normal", "aberrant"),
    ("orderly", "chaotic"), ("structured", "unstructured"),
    ("uniform", "erratic"),
]

# SUCCESS/FAILURE: concrete outcome-evaluation vocabulary
# Avoiding UD vocabulary (grow/shrink already there) and LR overlap (right/wrong skipped)
SUCCESS_FAILURE_PAIRS = [
    ("win", "lose"),
    ("won", "lost"),
    ("winning", "losing"),
    ("winner", "loser"),
    ("succeed", "fail"),
    ("succeeded", "failed"),
    ("success", "failure"),
    ("successful", "unsuccessful"),
    ("score", "miss"),
    ("scored", "missed"),
    ("correct", "incorrect"),
    ("accomplish", "fail"),
    ("accomplished", "failed"),
    ("achieve", "miss"),
    ("achievement", "failure"),
    ("triumph", "defeat"),
    ("triumphed", "defeated"),
    ("victory", "defeat"),
    ("gain", "loss"),
    ("profit", "loss"),
    ("pass", "fail"),
    ("passed", "failed"),
    ("hit", "miss"),
    ("complete", "fail"),
    ("reward", "punishment"),
]


def build_axis(pairs, label=""):
    offs, missing = [], []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            offs.append(wv[a] - wv[c])
        else:
            missing.append((a, c))
    if missing and label:
        print(f"  [{label}] missing: {missing}")
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def residualize(v, axes):
    r = v.copy()
    for a in axes:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


print("\n=== Building axes ===")
A_val = build_axis(VALENCE_PAIRS)
A_aro = build_axis(AROUSAL_PAIRS)
A_coh = build_axis(COHERENCE_PAIRS, "COHERENCE")
A_suc = build_axis(SUCCESS_FAILURE_PAIRS, "SUCCESS_FAILURE")

A_ud = build_axis(UP_DOWN_MML)
A_io_clean = build_axis(IN_OUT_MML_CLEAN)
A_fb = build_axis(FORWARD_BACK_MML)
A_ld = build_axis(LIGHT_DARK_MML)
A_exist = build_axis(EXISTENCE_MML)
A_diff = build_axis(DIFFICULTY_BURDEN_MML)
A_path = build_axis(PATH_MOTION_MML)


# ============================================================
# TEST 1: Affect-loading of candidates
# ============================================================
print("\n=== TEST 1: V/A loading ===")
print(f"  {'axis':>20}  {'cos(V)':>8}  {'cos(A)':>8}  {'%V':>5}  {'%A':>5}  {'%resid':>7}")
for name, axis in [("COHERENCE", A_coh), ("SUCCESS_FAILURE", A_suc)]:
    cv = float(axis @ A_val)
    ca = float(axis @ A_aro)
    r = axis - cv*A_val - ca*A_aro
    rn = float(np.linalg.norm(r))
    print(f"  {name:>20}  {cv:>+8.3f}  {ca:>+8.3f}  {cv**2*100:>4.1f}%  {ca**2*100:>4.1f}%  {rn**2*100:>6.1f}%")


# ============================================================
# TEST 2: COHERENCE vs SUCCESS_FAILURE
# ============================================================
print("\n=== TEST 2: COHERENCE vs SUCCESS_FAILURE ===")
coh_va = residualize(A_coh, [A_val, A_aro])
suc_va = residualize(A_suc, [A_val, A_aro])
print(f"  cos(COHERENCE, SUCCESS_FAILURE)         = {float(A_coh @ A_suc):+.4f}")
print(f"  cos(COHERENCE_VA, SUCCESS_FAILURE_VA)   = {float(coh_va @ suc_va):+.4f}")


# ============================================================
# TEST 3: Alignments with cluster axes
# ============================================================
print("\n=== TEST 3: Candidate alignments with cluster (raw) ===")
schemas = {
    "UD": A_ud, "IO_CLEAN": A_io_clean, "FB": A_fb, "LD": A_ld,
    "EXIST": A_exist, "DIFF": A_diff, "PATH": A_path,
}
print(f"  {'':>20}  " + "  ".join(f"{n:>8}" for n in schemas))
for name, axis in [("COHERENCE", A_coh), ("SUCCESS_FAILURE", A_suc)]:
    print(f"  {name:>20}  " + "  ".join(f"{float(axis @ a):>+8.3f}" for a in schemas.values()))

print("\n=== TEST 3b: After V+A residualization ===")
schemas_va = {name: residualize(axis, [A_val, A_aro]) for name, axis in schemas.items()}
print(f"  {'':>20}  " + "  ".join(f"{n:>8}" for n in schemas))
for name, axis in [("COHERENCE_VA", coh_va), ("SUCCESS_FAILURE_VA", suc_va)]:
    print(f"  {name:>20}  " + "  ".join(f"{float(axis @ a):>+8.3f}" for a in schemas_va.values()))


# ============================================================
# TEST 4: vs the UD-FB residual
# ============================================================
print("\n=== TEST 4: Candidates vs UD-FB shared residual (after V+A+EXIST) ===")
ud_vae = residualize(A_ud, [A_val, A_aro, A_exist])
fb_vae = residualize(A_fb, [A_val, A_aro, A_exist])
shared_resid = (ud_vae + fb_vae) / 2
shared_resid = shared_resid / np.linalg.norm(shared_resid)

print(f"  cos(residual, COHERENCE)         = {float(shared_resid @ A_coh):+.4f}")
print(f"  cos(residual, COHERENCE_VA)      = {float(shared_resid @ coh_va):+.4f}")
print(f"  cos(residual, SUCCESS_FAILURE)   = {float(shared_resid @ A_suc):+.4f}")
print(f"  cos(residual, SUCCESS_FAILURE_VA)= {float(shared_resid @ suc_va):+.4f}")


# ============================================================
# TEST 5: Triple residualization
# ============================================================
print("\n=== TEST 5: UD-FB cosine through residualization stages ===")
ud_v = residualize(A_ud, [A_val])
fb_v = residualize(A_fb, [A_val])
ud_va = residualize(A_ud, [A_val, A_aro])
fb_va = residualize(A_fb, [A_val, A_aro])
ud_vasuc = residualize(A_ud, [A_val, A_aro, A_suc])
fb_vasuc = residualize(A_fb, [A_val, A_aro, A_suc])
ud_full = residualize(A_ud, [A_val, A_aro, A_exist, A_suc])
fb_full = residualize(A_fb, [A_val, A_aro, A_exist, A_suc])

print(f"  Raw:                              cos(UD, FB) = {float(A_ud @ A_fb):+.4f}")
print(f"  V removed:                        cos = {float(ud_v @ fb_v):+.4f}")
print(f"  V+A removed:                      cos = {float(ud_va @ fb_va):+.4f}")
print(f"  V+A+EXIST removed:                cos = {float(ud_vae @ fb_vae):+.4f}")
print(f"  V+A+SUCCESS removed:              cos = {float(ud_vasuc @ fb_vasuc):+.4f}")
print(f"  V+A+EXIST+SUCCESS removed:        cos = {float(ud_full @ fb_full):+.4f}")


# ============================================================
# TEST 6: Nearest neighbors of SUCCESS_FAILURE
# ============================================================
print("\n=== TEST 6: SUCCESS_FAILURE nearest neighbors ===")
print(f"  Positive pole (success):")
for w, s in wv.similar_by_vector(A_suc, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole (failure):")
for w, s in wv.similar_by_vector(-A_suc, topn=15):
    print(f"    {s:>+.4f}  {w}")

print("\n=== SUCCESS_FAILURE after V+A residualization ===")
print(f"  Positive pole:")
for w, s in wv.similar_by_vector(suc_va, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Negative pole:")
for w, s in wv.similar_by_vector(-suc_va, topn=15):
    print(f"    {s:>+.4f}  {w}")


# ============================================================
# Save
# ============================================================
np.savez(
    "/Users/macn/Documents/embeddingexp/exp38_results.npz",
    success_failure=A_suc,
    success_failure_va=suc_va,
    coherence=A_coh,
    coherence_va=coh_va,
)
print("\nSaved: exp38_results.npz")
