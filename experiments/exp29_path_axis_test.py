"""
exp29: PATH axis test — is the UD-FB residual really directed motion?

Construct a PATH axis from motion-vs-stasis word pairs that avoid all
UP/DOWN/FORWARD/BACK vocabulary. Test:
  1. cos(PATH, UD) and cos(PATH, FB) — should both be substantial if PATH
     is the shared content
  2. cos(PATH, IO/LD/LR) — should be small if PATH is genuinely orthogonal
     to the non-motion schemas
  3. cos(PATH, VALENCE) and cos(PATH, AROUSAL) — should be small if PATH is
     independent from affect (otherwise we're just rediscovering V or A)
  4. Residualize UD and FB with VALENCE + AROUSAL + PATH. Does UD-FB drop to
     near zero? If yes: cluster is V + A + PATH. If no: there's a fourth thing.
"""
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300 (cached)...")
wv = api.load("glove-wiki-gigaword-300")

# Motion vs stasis. No up/down/forward/back/in/out/left/right vocabulary.
PATH_PAIRS = [
    ("moving", "stationary"),
    ("moved", "stayed"),
    ("traveled", "remained"),
    ("travels", "remains"),
    ("traveling", "staying"),
    ("journey", "rest"),
    ("going", "staying"),
    ("departed", "stayed"),
    ("shifted", "fixed"),
    ("migrated", "settled"),
    ("wandered", "paused"),
    ("flowing", "static"),
    ("flowing", "still"),
    ("dynamic", "static"),
    ("motion", "rest"),
    ("mobile", "immobile"),
    ("changing", "constant"),
    ("drifted", "anchored"),
    ("roamed", "stopped"),
    ("transit", "stationary"),
]

# Reuse anchors and schemas from exp28
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
        print(f"  [{label}] missing: {missing}")
    arr = np.stack(offsets)
    axis_raw = arr.mean(axis=0)
    return axis_raw / np.linalg.norm(axis_raw), arr


print("\n=== Building axes ===")
A_path, path_offs = build_axis(PATH_PAIRS, "PATH")
A_val, _  = build_axis(VALENCE_PAIRS, "VAL")
A_aro, _  = build_axis(AROUSAL_PAIRS, "ARO")
print(f"  PATH built from {len(path_offs)} pairs")

schema_axes = {}
for name, pairs in [("UP-DOWN", UD_PAIRS), ("IN-OUT", IO_PAIRS),
                    ("FORWARD-BACK", FB_PAIRS), ("LIGHT-DARK", LD_PAIRS),
                    ("LEFT-RIGHT", LR_PAIRS)]:
    a, _ = build_axis(pairs, name)
    schema_axes[name] = a

# Within-PATH coherence (do the motion pairs agree on a direction?)
norms = np.linalg.norm(path_offs, axis=1, keepdims=True)
unit = path_offs / norms
iu = np.triu_indices_from(unit @ unit.T, k=1)
within_path = float((unit @ unit.T)[iu].mean())
print(f"\nWithin-PATH cross-pair coherence: mean cos = {within_path:+.4f}")


print("\n=== PATH vs everything ===")
print(f"  cos(PATH, VALENCE)   = {float(A_path @ A_val):+.4f}")
print(f"  cos(PATH, AROUSAL)   = {float(A_path @ A_aro):+.4f}")
for name, axis in schema_axes.items():
    print(f"  cos(PATH, {name:>13}) = {float(A_path @ axis):+.4f}")


print("\n=== Triple decomposition: schema = a*VAL + b*ARO + c*PATH + residual ===")
print(f"  {'schema':>14}  {'cos(VAL)':>9}  {'cos(ARO)':>9}  {'cos(PATH)':>10}  {'||resid||':>10}")

# Use Gram-Schmidt-style sequential projection
def triple_decomp(schema_axis):
    s = schema_axis / np.linalg.norm(schema_axis)
    cv = float(s @ A_val)
    ca = float(s @ A_aro)
    cp = float(s @ A_path)
    # Subtract sequentially (note V, A, PATH not orthogonal)
    r = s - cv*A_val - ca*A_aro - cp*A_path
    return cv, ca, cp, float(np.linalg.norm(r)), r

residuals_triple = {}
for name, axis in schema_axes.items():
    cv, ca, cp, rn, r = triple_decomp(axis)
    residuals_triple[name] = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-12 else r
    print(f"  {name:>14}  {cv:>+9.4f}  {ca:>+9.4f}  {cp:>+10.4f}  {rn:>10.4f}")


print("\n=== Pairwise cosines after triple residualization (V + A + PATH removed) ===")
names_list = list(schema_axes.keys())
M = np.zeros((len(names_list), len(names_list)))
for i, ni in enumerate(names_list):
    for j, nj in enumerate(names_list):
        M[i, j] = float(residuals_triple[ni] @ residuals_triple[nj])
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names_list))
for i, ni in enumerate(names_list):
    print(f"  {ni:>14}" + "".join(f"  {M[i, j]:>+13.4f}" for j in range(len(names_list))))


print("\n=== Comparison of UD-FB cosine across removal stages ===")
# Compute UD-FB cosine progressively:
ud = schema_axes["UP-DOWN"] / np.linalg.norm(schema_axes["UP-DOWN"])
fb = schema_axes["FORWARD-BACK"] / np.linalg.norm(schema_axes["FORWARD-BACK"])

# Stage 0: raw
print(f"  raw UD-FB cosine                     = {float(ud @ fb):+.4f}")

# Stage 1: remove V
ud_v = ud - (ud @ A_val) * A_val
fb_v = fb - (fb @ A_val) * A_val
ud_v /= np.linalg.norm(ud_v); fb_v /= np.linalg.norm(fb_v)
print(f"  after VALENCE removal                = {float(ud_v @ fb_v):+.4f}")

# Stage 2: remove V + A
ud_va = ud - (ud @ A_val) * A_val - (ud @ A_aro) * A_aro
fb_va = fb - (fb @ A_val) * A_val - (fb @ A_aro) * A_aro
ud_va /= np.linalg.norm(ud_va); fb_va /= np.linalg.norm(fb_va)
print(f"  after VALENCE + AROUSAL removal      = {float(ud_va @ fb_va):+.4f}")

# Stage 3: remove V + A + PATH
ud_vap = ud - (ud @ A_val) * A_val - (ud @ A_aro) * A_aro - (ud @ A_path) * A_path
fb_vap = fb - (fb @ A_val) * A_val - (fb @ A_aro) * A_aro - (fb @ A_path) * A_path
ud_vap /= np.linalg.norm(ud_vap); fb_vap /= np.linalg.norm(fb_vap)
print(f"  after VALENCE + AROUSAL + PATH       = {float(ud_vap @ fb_vap):+.4f}")


print("\n=== Reading guide ===")
print(f"  Strong PATH alignment with UD and FB (both > +0.4) → PATH is real")
print(f"  Low PATH alignment with IO/LD/LR → PATH is independent of those")
print(f"  Triple-residualized UD-FB drops near zero → cluster is V + A + PATH")
print(f"  Triple-residualized UD-FB still substantial → there's a 4th axis we haven't named")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp29_results.npz",
    cosine_matrix_residualized=M,
    names=names_list,
    path_axis=A_path,
)
print(f"\nSaved: exp29_results.npz")
