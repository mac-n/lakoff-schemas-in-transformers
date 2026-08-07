"""
exp34: Canonical replication of exp27/28/30/31/32 using Lakoff MML vocabulary.

Replicates the core word2vec findings with citable anchor sets drawn from the
Master Metaphor List (Lakoff, Espenson, & Schwartz 1991) via `lakoff_canonical_vocabulary.py`.

Tests:
  1. Cross-method validation: hand-curated axes vs MML-canonical axes — are
     we tracking the same things?
  2. Pairwise cosine matrix (replicates exp27)
  3. V+A decomposition (replicates exp28)
  4. Cluster-survival under V+A residualization (the salience cluster check)
  5. **NEW critical test**: cos(EXISTENCE_MML_VA, UD-FB-LD residual cluster).
     We previously INFERRED a yang/yin direction by averaging UD/FB/LD residuals.
     Now we have Lakoff's canonical EXISTENCE vocabulary (composed from his
     attested constellation of existence-change metaphors). If EXISTENCE_MML_VA
     aligns highly with the inferred direction, we have cross-method validation
     of the yang/yin / becoming-unbecoming axis using Lakoff's own vocabulary.
  6. IO_CLEAN_MML vs EXISTENCE_MML — does the IN=yin Taoist prediction hold
     against canonical Lakoff containment?
  7. Triple-residualization: subtract V + A + EXISTENCE from UD and FB.
     Does the UD-FB cluster finally collapse?
"""
import numpy as np
import gensim.downloader as api
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML,
    PATH_MOTION_MML, LIGHT_DARK_MML, EXISTENCE_MML,
    FORCE_MML, BALANCE_MML, DIFFICULTY_BURDEN_MML,
)


print("Loading glove-wiki-gigaword-300 (cached)...")
wv = api.load("glove-wiki-gigaword-300")


# Non-Lakoff anchor axes (VALENCE, AROUSAL) - unchanged from exp28
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

# Hand-curated versions from exp27 (for cross-method comparison)
HAND_UD = [
    ("rose", "fell"), ("rising", "falling"), ("climbing", "descending"),
    ("ascended", "descended"), ("soaring", "plunging"), ("lifted", "sunk"),
    ("higher", "lower"), ("upward", "downward"), ("up", "down"),
    ("happy", "sad"), ("cheerful", "gloomy"), ("elated", "dejected"),
    ("joyful", "sorrowful"), ("uplifted", "depressed"),
    ("more", "less"), ("increase", "decrease"), ("grew", "shrank"),
    ("expanded", "contracted"), ("multiplied", "dwindled"),
    ("promoted", "demoted"), ("prestigious", "disgraced"),
    ("esteemed", "despised"), ("eminent", "lowly"),
    ("healthy", "sick"), ("thriving", "ailing"), ("strong", "weak"),
    ("vigorous", "feeble"), ("robust", "frail"),
]
HAND_IO = [
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    ("remembered", "forgotten"), ("retained", "dismissed"),
    ("married", "divorced"), ("engaged", "estranged"), ("together", "apart"),
    ("trapped", "escaped"), ("stranded", "rescued"), ("captured", "freed"),
    ("included", "excluded"), ("admitted", "expelled"),
]
HAND_FB = [
    ("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"),
    ("forwards", "backwards"), ("progress", "regress"), ("improved", "declined"),
    ("gained", "lost"), ("advanced", "retreated"),
    ("future", "past"), ("next", "previous"), ("later", "earlier"),
    ("upcoming", "former"), ("evolved", "devolved"), ("developed", "regressed"),
    ("onward", "return"),
]
HAND_LD = [
    ("bright", "dark"), ("light", "dark"), ("illuminated", "shadowed"),
    ("radiant", "dim"), ("sunny", "gloomy"), ("shining", "dim"),
    ("luminous", "shadowy"), ("glowing", "darkened"),
    ("clear", "obscure"), ("lucid", "murky"), ("transparent", "opaque"),
    ("obvious", "hidden"), ("hopeful", "hopeless"), ("optimistic", "pessimistic"),
    ("good", "evil"), ("pure", "tainted"), ("noble", "wicked"),
    ("virtuous", "sinful"), ("righteous", "corrupt"),
    ("enlightened", "ignorant"), ("informed", "uninformed"),
    ("known", "unknown"), ("revealed", "concealed"),
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
        print(f"  [{label}] missing: {len(missing)} pairs ({missing[:3]}{'...' if len(missing) > 3 else ''})")
    if not offs:
        return None, 0, None
    arr = np.stack(offs)
    raw = arr.mean(axis=0)
    return raw / np.linalg.norm(raw), len(offs), arr


def residualize(v, axes_to_remove):
    r = v.copy()
    for a in axes_to_remove:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


print("\n=== Building axes ===")
A_val, n_val, _ = build_axis(VALENCE_PAIRS, "VALENCE")
A_aro, n_aro, _ = build_axis(AROUSAL_PAIRS, "AROUSAL")

# Build hand-curated and MML versions
A_ud_hand, _, _ = build_axis(HAND_UD, "HAND_UD")
A_io_hand, _, _ = build_axis(HAND_IO, "HAND_IO")
A_fb_hand, _, _ = build_axis(HAND_FB, "HAND_FB")
A_ld_hand, _, _ = build_axis(HAND_LD, "HAND_LD")

A_ud_mml, n_ud, _ = build_axis(UP_DOWN_MML, "UP_DOWN_MML")
A_io_mml, n_io, _ = build_axis(IN_OUT_MML, "IN_OUT_MML")
A_io_clean_mml, n_ioc, _ = build_axis(IN_OUT_MML_CLEAN, "IN_OUT_MML_CLEAN")
A_fb_mml, n_fb, _ = build_axis(FORWARD_BACK_MML, "FORWARD_BACK_MML")
A_path_mml, n_path, _ = build_axis(PATH_MOTION_MML, "PATH_MOTION_MML")
A_ld_mml, n_ld, _ = build_axis(LIGHT_DARK_MML, "LIGHT_DARK_MML")
A_exist_mml, n_exist, _ = build_axis(EXISTENCE_MML, "EXISTENCE_MML")
A_force_mml, n_force, _ = build_axis(FORCE_MML, "FORCE_MML")
A_balance_mml, n_bal, _ = build_axis(BALANCE_MML, "BALANCE_MML")
A_diff_mml, n_diff, _ = build_axis(DIFFICULTY_BURDEN_MML, "DIFFICULTY_BURDEN_MML")

print(f"\n  Pair counts (after vocab filter):")
for name, n in [("VALENCE", n_val), ("AROUSAL", n_aro), ("UP_DOWN_MML", n_ud),
                ("IN_OUT_MML", n_io), ("IN_OUT_MML_CLEAN", n_ioc),
                ("FORWARD_BACK_MML", n_fb), ("PATH_MOTION_MML", n_path),
                ("LIGHT_DARK_MML", n_ld), ("EXISTENCE_MML", n_exist),
                ("FORCE_MML", n_force), ("BALANCE_MML", n_bal),
                ("DIFFICULTY_BURDEN_MML", n_diff)]:
    print(f"    {name:>25}: {n}")


# =================================================================
# TEST 1: Cross-method validation
# =================================================================
print(f"\n=== TEST 1: Cross-method validation (hand-curated vs MML) ===")
print(f"  High cosine = both methods recover the same axis.")
print(f"  cos(UD_HAND, UD_MML) = {float(A_ud_hand @ A_ud_mml):+.4f}")
print(f"  cos(IO_HAND, IO_MML) = {float(A_io_hand @ A_io_mml):+.4f}")
print(f"  cos(FB_HAND, FB_MML) = {float(A_fb_hand @ A_fb_mml):+.4f}")
print(f"  cos(LD_HAND, LD_MML) = {float(A_ld_hand @ A_ld_mml):+.4f}")


# =================================================================
# TEST 2: Pairwise cosines among MML axes (replicates exp27 with MML)
# =================================================================
print(f"\n=== TEST 2: Pairwise cosines between MML schema axes ===")
mml_axes = {
    "UD": A_ud_mml, "IO": A_io_mml, "IO_clean": A_io_clean_mml,
    "FB": A_fb_mml, "PATH": A_path_mml, "LD": A_ld_mml,
    "EXIST": A_exist_mml, "FORCE": A_force_mml,
    "BAL": A_balance_mml, "DIFF": A_diff_mml,
}
names = list(mml_axes.keys())
M = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M[i, j] = float(mml_axes[ni] @ mml_axes[nj])
print(f"  {'':>10}" + "  ".join(f"{n:>8}" for n in names))
for i, ni in enumerate(names):
    print(f"  {ni:>10}" + "  ".join(f"{M[i, j]:>+8.3f}" for j in range(len(names))))


# =================================================================
# TEST 3: V+A decomposition (replicates exp28)
# =================================================================
print(f"\n=== TEST 3: V+A decomposition of each MML axis ===")
print(f"  {'schema':>10}  {'cos(V)':>8}  {'cos(A)':>8}  {'%V':>5}  {'%A':>5}  {'%resid':>7}")
for name, axis in mml_axes.items():
    cv = float(axis @ A_val)
    ca = float(axis @ A_aro)
    r = axis - cv * A_val - ca * A_aro
    rn = float(np.linalg.norm(r))
    print(f"  {name:>10}  {cv:>+8.3f}  {ca:>+8.3f}  {cv**2*100:>4.1f}%  {ca**2*100:>4.1f}%  {rn**2*100:>6.1f}%")


# =================================================================
# TEST 4: V+A-residualized pairwise cosines (does salience cluster survive?)
# =================================================================
print(f"\n=== TEST 4: V+A-residualized pairwise cosines (does cluster survive?) ===")
mml_va = {name: residualize(axis, [A_val, A_aro]) for name, axis in mml_axes.items()}
M_va = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M_va[i, j] = float(mml_va[ni] @ mml_va[nj])
print(f"  {'':>10}" + "  ".join(f"{n:>8}" for n in names))
for i, ni in enumerate(names):
    print(f"  {ni:>10}" + "  ".join(f"{M_va[i, j]:>+8.3f}" for j in range(len(names))))


# =================================================================
# TEST 5 (CRITICAL): EXISTENCE_MML vs inferred yang/yin direction
# =================================================================
print(f"\n=== TEST 5 (CRITICAL): EXISTENCE_MML vs inferred yang/yin axis ===")
print(f"  Inferred yang/yin was mean(UD_VA, FB_VA, LD_VA) — see exp31.")
inferred_yangyin = (mml_va["UD"] + mml_va["FB"] + mml_va["LD"]) / 3
inferred_yangyin = inferred_yangyin / np.linalg.norm(inferred_yangyin)

print(f"  cos(EXISTENCE_MML, inferred_yangyin)         = {float(A_exist_mml @ inferred_yangyin):+.4f}")
print(f"  cos(EXISTENCE_MML_VA, inferred_yangyin)      = {float(mml_va['EXIST'] @ inferred_yangyin):+.4f}")
print(f"  ^^^ HIGH cosine = canonical Lakoff EXISTENCE recovers the same axis we")
print(f"      inferred from UD/FB/LD residuals. Direct support for yang/yin as a")
print(f"      shared underlying primitive across Lakoff schemas.")


# =================================================================
# TEST 6: IO_CLEAN_MML vs EXISTENCE_MML (the IN=yin Taoist prediction)
# =================================================================
print(f"\n=== TEST 6: IO_CLEAN_MML vs EXISTENCE_MML (IN=yin?) ===")
print(f"  Traditional Taoism predicts IN=yin → cos should be NEGATIVE.")
print(f"  cos(IO_CLEAN, EXIST)             = {float(A_io_clean_mml @ A_exist_mml):+.4f}")
print(f"  cos(IO_CLEAN_VA, EXIST_VA)       = {float(mml_va['IO_clean'] @ mml_va['EXIST']):+.4f}")
print(f"  cos(IO_CLEAN, inferred_yangyin)  = {float(A_io_clean_mml @ inferred_yangyin):+.4f}")
print(f"  ^^^ If near zero: containment orthogonal to becoming (self/other holds).")
print(f"      If negative: IN=yin (traditional Taoism).")
print(f"      If positive: IN=yang (modern English reorganization OR valence residue).")


# =================================================================
# TEST 7: Triple-residualization on UD and FB with V + A + EXISTENCE
# =================================================================
print(f"\n=== TEST 7: After subtracting V + A + EXISTENCE_MML from UD/FB ===")
print(f"  Does the UD-FB cluster finally collapse, or is there still residual?")
ud_full = residualize(A_ud_mml, [A_val, A_aro, A_exist_mml])
fb_full = residualize(A_fb_mml, [A_val, A_aro, A_exist_mml])
ld_full = residualize(A_ld_mml, [A_val, A_aro, A_exist_mml])
print(f"  Pre (raw):                       cos(UD, FB) = {float(A_ud_mml @ A_fb_mml):+.4f}")
print(f"  After V removal:                  cos(UD_V, FB_V) = {float(residualize(A_ud_mml, [A_val]) @ residualize(A_fb_mml, [A_val])):+.4f}")
print(f"  After V+A removal:                cos(UD_VA, FB_VA) = {float(mml_va['UD'] @ mml_va['FB']):+.4f}")
print(f"  After V+A+EXISTENCE removal:      cos(UD_full, FB_full) = {float(ud_full @ fb_full):+.4f}")
print(f"  After V+A+EXISTENCE on UD-LD:     cos(UD_full, LD_full) = {float(ud_full @ ld_full):+.4f}")
print(f"  After V+A+EXISTENCE on FB-LD:     cos(FB_full, LD_full) = {float(fb_full @ ld_full):+.4f}")


# =================================================================
# TEST 8: Nearest neighbors of EXISTENCE_MML and its V+A residual
# =================================================================
print(f"\n=== TEST 8: Nearest neighbors of EXISTENCE_MML ===")
print(f"  EXISTENCE_MML positive pole (emergence):")
for w, s in wv.similar_by_vector(A_exist_mml, topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  EXISTENCE_MML negative pole (dissolution):")
for w, s in wv.similar_by_vector(-A_exist_mml, topn=15):
    print(f"    {s:>+.4f}  {w}")

print(f"\n=== EXISTENCE_MML after V+A residualization ===")
print(f"  Yang pole (after V+A removal):")
for w, s in wv.similar_by_vector(mml_va["EXIST"], topn=15):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Yin pole (after V+A removal):")
for w, s in wv.similar_by_vector(-mml_va["EXIST"], topn=15):
    print(f"    {s:>+.4f}  {w}")


# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp34_results.npz",
    mml_axes={name: axis for name, axis in mml_axes.items()},
    mml_va={name: axis for name, axis in mml_va.items()},
    inferred_yangyin=inferred_yangyin,
    pairwise_pre=M,
    pairwise_va=M_va,
)
print("\nSaved: exp34_results.npz")
