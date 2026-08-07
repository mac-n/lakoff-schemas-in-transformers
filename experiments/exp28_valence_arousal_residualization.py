"""
exp28: VALENCE and AROUSAL axes in word2vec + residualization of schema axes.

The question: how much of each Lakoff schema's apparent "axis" is just valence
(good-bad direction) or arousal (high-arousal vs low-arousal direction)? If we
build VALENCE and AROUSAL from non-Lakoff vocabulary and project them out of
each schema axis, what's left?

Two outcomes to discriminate:
  - UD-LD-FB cluster COLLAPSES after subtracting valence+arousal → the salience
    cluster IS the affect cluster wearing schema clothes
  - UD-LD-FB cluster SURVIVES → there's schema-specific content beyond affect

Also informative: does IO become more aligned (had hidden valence loading)?
Does LR develop alignment (had salience-aligned content orthogonal to V/A)?

Caveats up front:
  - Hand-curated anchors, not Warriner VAD norms. First pass.
  - AROUSAL is the lexical proxy for "salience"; not identical to cognitive
    salience (attention allocation). It's the cleanest single-word substitute.
  - Valence and arousal are known to be non-orthogonal in human lexicons.
    We compute cos(VALENCE, AROUSAL) and report.
"""
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300 (cached from exp27)...")
wv = api.load("glove-wiki-gigaword-300")
print(f"Loaded: {len(wv.key_to_index)} words, {wv.vector_size}-dim")


# ============================== ANCHOR WORD PAIRS ==============================
# VALENCE: high-valence (positive) vs low-valence (negative) anchors.
# Curated to avoid Lakoff schema vocabulary (no rising/falling, bright/dark,
# inside/outside, forward/backward, left/right, or any clear directional/spatial
# words). Also avoids HAPPY-IS-UP children (happy, sad, etc.) since those would
# duplicate UD's mood vocabulary.
VALENCE_PAIRS = [
    ("pleasant", "unpleasant"),
    ("desirable", "undesirable"),
    ("agreeable", "disagreeable"),
    ("enjoyable", "distasteful"),
    ("delightful", "awful"),
    ("beneficial", "harmful"),
    ("wonderful", "terrible"),
    ("excellent", "dreadful"),
    ("favorable", "unfavorable"),
    ("satisfying", "frustrating"),
    ("nice", "nasty"),
    ("kind", "cruel"),
]

# AROUSAL: high-arousal vs low-arousal, trying to balance valence across pairs.
# Hardest to make clean — many arousal-asymmetric words also carry valence.
# Hand-curated; flag known issues in code.
AROUSAL_PAIRS = [
    ("intense", "mild"),         # relatively valence-neutral
    ("intense", "gentle"),
    ("alert", "drowsy"),         # alert slightly positive, drowsy slightly negative
    ("urgent", "leisurely"),     # urgent slightly negative, leisurely slightly positive — opposing valence, helps cancel
    ("frantic", "tranquil"),     # frantic negative, tranquil positive — opposing
    ("energetic", "lethargic"),  # energetic positive, lethargic negative — opposing
    ("aroused", "relaxed"),      # both can be positive — valence-balanced
    ("sharp", "dull"),           # sharp slightly positive, dull negative — opposing
    ("acute", "subtle"),         # both can be positive
    ("vivid", "faint"),          # vivid positive, faint slightly negative
    ("electric", "placid"),      # both relatively positive
    ("turbulent", "still"),      # turbulent negative, still positive — opposing
]


# ============================== SCHEMA PAIRS (from exp27) ==============================
UD_PAIRS = [
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
IO_PAIRS = [
    ("inside", "outside"), ("contained", "released"), ("enclosed", "freed"),
    ("in", "out"), ("entered", "exited"), ("internal", "external"),
    ("remembered", "forgotten"), ("retained", "dismissed"),
    ("married", "divorced"), ("engaged", "estranged"), ("together", "apart"),
    ("trapped", "escaped"), ("stranded", "rescued"), ("captured", "freed"),
    ("included", "excluded"), ("admitted", "expelled"),
]
FB_PAIRS = [
    ("forward", "backward"), ("advance", "retreat"), ("ahead", "behind"),
    ("forwards", "backwards"), ("progress", "regress"), ("improved", "declined"),
    ("gained", "lost"), ("advanced", "retreated"),
    ("future", "past"), ("next", "previous"), ("later", "earlier"),
    ("upcoming", "former"), ("evolved", "devolved"), ("developed", "regressed"),
    ("onward", "return"),
]
LD_PAIRS = [
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
LR_PAIRS = [
    ("left", "right"), ("leftward", "rightward"),
    ("port", "starboard"), ("liberal", "conservative"),
]

SCHEMAS = {
    "UP-DOWN":      UD_PAIRS,
    "IN-OUT":       IO_PAIRS,
    "FORWARD-BACK": FB_PAIRS,
    "LIGHT-DARK":   LD_PAIRS,
    "LEFT-RIGHT":   LR_PAIRS,
}


# ============================== AXIS BUILDER ==============================
def build_axis(pairs, label):
    offsets = []
    missing = []
    for a, c in pairs:
        if a not in wv.key_to_index:
            missing.append(a); continue
        if c not in wv.key_to_index:
            missing.append(c); continue
        offsets.append(wv[a] - wv[c])
    if missing:
        print(f"  [{label}] missing from vocab: {missing}")
    if not offsets:
        return None, None
    arr = np.stack(offsets)
    axis_raw = arr.mean(axis=0)
    nrm = np.linalg.norm(axis_raw)
    axis = axis_raw / nrm if nrm > 1e-12 else axis_raw
    return axis, arr


print(f"\n=== Building anchor axes ===")
A_val, val_offsets = build_axis(VALENCE_PAIRS, "VALENCE")
A_aro, aro_offsets = build_axis(AROUSAL_PAIRS, "AROUSAL")
print(f"  VALENCE: built from {len(val_offsets)} pairs")
print(f"  AROUSAL: built from {len(aro_offsets)} pairs")

print(f"\n=== Building schema axes ===")
schema_axes = {}
for name, pairs in SCHEMAS.items():
    a, _ = build_axis(pairs, name)
    schema_axes[name] = a
    print(f"  {name}: ok")


# ============================== INDEPENDENCE OF V/A ==============================
print(f"\n=== Independence of VALENCE and AROUSAL ===")
va_cos = float(A_val @ A_aro)
print(f"  cos(VALENCE, AROUSAL) = {va_cos:+.4f}")
print(f"  ({'~independent' if abs(va_cos) < 0.2 else 'NOT independent — projections will be entangled'})")


# ============================== PROJECTION DECOMPOSITION ==============================
def decompose(schema_axis, axes_list, axis_names):
    """Decompose a schema axis into projections onto each anchor axis + residual.
    Reports: ||proj_v|| / ||schema|| for each axis, and ||residual|| / ||schema||.
    schema_axis is unit-length, so we just compute (axis · anchor)."""
    out = {}
    schema_unit = schema_axis / np.linalg.norm(schema_axis)
    residual = schema_unit.copy()
    for anchor, name in zip(axes_list, axis_names):
        proj_scalar = float(schema_unit @ anchor)
        out[f"cos_{name}"] = proj_scalar
        residual = residual - proj_scalar * anchor
    out["residual_norm"] = float(np.linalg.norm(residual))
    out["residual"] = residual
    return out


print(f"\n=== Schema axis decomposition ===")
print(f"  {'schema':>14}  {'cos(VAL)':>9}  {'cos(ARO)':>9}  {'||resid||':>10}  {'%V':>5}  {'%A':>5}  {'%R':>5}")
decompositions = {}
for name, axis in schema_axes.items():
    d = decompose(axis, [A_val, A_aro], ["VAL", "ARO"])
    decompositions[name] = d
    v_frac = d["cos_VAL"]**2
    a_frac = d["cos_ARO"]**2
    r_frac = d["residual_norm"]**2
    # Note: v_frac + a_frac + r_frac may exceed 1 because VAL and ARO aren't orthogonal
    print(f"  {name:>14}  {d['cos_VAL']:>+9.4f}  {d['cos_ARO']:>+9.4f}  "
          f"{d['residual_norm']:>10.4f}  {100*v_frac:>4.1f}%  {100*a_frac:>4.1f}%  {100*r_frac:>4.1f}%")

# Sanity check: also project schema axes onto each other to compare pre-residualization
print(f"\n=== Pre-residualization pairwise schema cosines (from exp27, for reference) ===")
names_list = list(SCHEMAS.keys())
M_pre = np.zeros((len(names_list), len(names_list)))
for i, ni in enumerate(names_list):
    for j, nj in enumerate(names_list):
        M_pre[i, j] = float(schema_axes[ni] @ schema_axes[nj])
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names_list))
for i, ni in enumerate(names_list):
    print(f"  {ni:>14}" + "".join(f"  {M_pre[i, j]:>+13.4f}" for j in range(len(names_list))))

# ============================== RESIDUALIZED PAIRWISE COSINES ==============================
# Take each schema's residual (after subtracting valence + arousal components),
# renormalize, and recompute pairwise cosines.
print(f"\n=== Residualized pairwise schema cosines (after subtracting VALENCE + AROUSAL) ===")
residual_axes = {}
for name, d in decompositions.items():
    r = d["residual"]
    nrm = np.linalg.norm(r)
    residual_axes[name] = r / nrm if nrm > 1e-12 else r

M_post = np.zeros((len(names_list), len(names_list)))
for i, ni in enumerate(names_list):
    for j, nj in enumerate(names_list):
        M_post[i, j] = float(residual_axes[ni] @ residual_axes[nj])
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names_list))
for i, ni in enumerate(names_list):
    print(f"  {ni:>14}" + "".join(f"  {M_post[i, j]:>+13.4f}" for j in range(len(names_list))))

# Difference (post - pre)
print(f"\n=== Change in pairwise cosines (post - pre) ===")
print(f"  Positive = pair became MORE aligned after removing V/A (was masked)")
print(f"  Negative = pair became LESS aligned after removing V/A (V/A was the alignment)")
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names_list))
for i, ni in enumerate(names_list):
    print(f"  {ni:>14}" + "".join(f"  {M_post[i, j] - M_pre[i, j]:>+13.4f}" for j in range(len(names_list))))

# ============================== READING GUIDE ==============================
print(f"\n=== Reading guide ===")
print(f"  Schemas with HIGH cos(VAL) → mostly a goodbad axis dressed up")
print(f"    expected: UD ~0.5-0.7, LD ~0.6-0.8 (both heavily valence-loaded)")
print(f"    surprising would be: IO or LR with high cos(VAL)")
print(f"")
print(f"  Schemas with HIGH cos(ARO) → mostly an arousal/salience axis dressed up")
print(f"    expected: maybe FB or UD (action-content, energy)")
print(f"")
print(f"  After residualization (V/A removed):")
print(f"    If UD-LD-FB cluster collapses (cos drops near 0) → cluster WAS V/A")
print(f"    If cluster survives → schema-specific shared structure beyond V/A")
print(f"    If IO becomes aligned → IO had hidden V/A loading masking shared shape")

# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp28_results.npz",
    pre_cosines=M_pre,
    post_cosines=M_post,
    names=names_list,
    valence_axis=A_val,
    arousal_axis=A_aro,
    schema_axes={k: v for k, v in schema_axes.items()},
)
print(f"\nSaved: exp28_results.npz")
