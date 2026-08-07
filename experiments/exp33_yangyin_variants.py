"""
exp33: multiple yang/yin variants and their cross-agreement.

Build separate yang/yin candidate axes from different aspects of the
multi-dimensional concept:
  - EXISTENCE (what exp32 built): born-died, created-destroyed, ...
  - EXPANSION: expanded-contracted, outward-inward, projected-withdrew,
    extended-retracted, swelled-shrank, dilated-constricted, broadened-narrowed
  - ACTIVITY: active-passive, vigorous-quiescent, energetic-still,
    dynamic-static, restless-tranquil
  - HEAT (Taoist yang/yin maps onto warm vs cool): warm-cool, hot-cool,
    fiery-frigid (yang is fiery, yin is cool — separate from VALENCE)
  - MASCULINE/FEMININE archetypal (the most loaded one — heavy cultural overlay):
    masculine-feminine, male-female, father-mother, brother-sister
    (note: these will encode gender bias from training data, not philosophy)
  - HEAVEN/EARTH (Taoist: yang=heaven, yin=earth): heaven-earth, sky-ground,
    celestial-terrestrial

Then:
  1. Pairwise cosines between variants — do they agree? If yes, unified
     yang/yin axis. If no, cluster of related axes.
  2. Each variant's cosine with valence, arousal, UD, FB, LD, IO_CLEAN —
     diagnostic of what each variant has confounded with.
  3. EXPANSION_yang/yin SHOULD align strongly with IO_CLEAN by construction
     (outward/inward IS in/out). This is the methodological check Niamh
     flagged — if true, the "is IN yin" test only makes sense against
     non-direction-flavor yang/yin.
  4. The "core yang/yin" candidate = mean of unit-normalized variants.
     Nearest neighbors and cosines with everything else.
"""
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")


VARIANTS = {
    "EXISTENCE": [
        ("born", "died"), ("birth", "death"), ("birth", "demise"),
        ("creating", "destroying"), ("creation", "destruction"),
        ("create", "destroy"), ("created", "destroyed"),
        ("built", "demolished"), ("building", "demolishing"),
        ("construct", "demolish"), ("constructed", "dismantled"),
        ("emerging", "vanishing"), ("emerged", "vanished"),
        ("emerge", "vanish"), ("appearing", "disappearing"),
        ("appeared", "disappeared"), ("arising", "dissipating"),
        ("arise", "dissipate"), ("assembled", "disassembled"),
        ("formed", "disintegrated"), ("generated", "dissolved"),
        ("materialized", "vanished"), ("founded", "abandoned"),
        ("establish", "abandon"), ("established", "abandoned"),
        ("starting", "ending"), ("began", "ended"), ("begin", "end"),
        ("started", "ended"), ("originated", "terminated"),
        ("commenced", "concluded"), ("active", "dormant"),
        ("activated", "deactivated"),
    ],
    "EXPANSION": [
        ("expanded", "contracted"), ("expanding", "contracting"),
        ("expand", "contract"), ("outward", "inward"),
        ("projected", "withdrew"), ("extended", "retracted"),
        ("extending", "retracting"), ("swelled", "shrank"),
        ("dilated", "constricted"), ("broadened", "narrowed"),
        ("widening", "narrowing"), ("opening", "closing"),
        ("unfurled", "furled"), ("stretched", "shriveled"),
        ("inflated", "deflated"),
    ],
    "ACTIVITY": [
        ("active", "passive"), ("vigorous", "quiescent"),
        ("energetic", "still"), ("dynamic", "static"),
        ("restless", "tranquil"), ("animated", "inert"),
        ("lively", "motionless"), ("agitated", "calm"),
        ("stirring", "quiet"), ("bustling", "still"),
    ],
    "HEAT": [
        ("warm", "cool"), ("hot", "cool"), ("fiery", "frigid"),
        ("burning", "freezing"), ("scorching", "chilly"),
        ("sweltering", "cold"), ("toasty", "icy"),
    ],
    "MASC_FEM": [
        ("masculine", "feminine"), ("male", "female"),
        ("father", "mother"), ("brother", "sister"),
        ("son", "daughter"), ("man", "woman"),
        ("boy", "girl"), ("husband", "wife"),
        ("king", "queen"), ("prince", "princess"),
    ],
    "HEAVEN_EARTH": [
        ("heaven", "earth"), ("sky", "ground"),
        ("celestial", "terrestrial"), ("heavens", "earth"),
        ("divine", "mundane"), ("spirit", "matter"),
    ],
}

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
IO_CLEAN_PAIRS = [
    ("inside", "outside"), ("contained", "released"),
    ("enclosed", "freed"), ("in", "out"),
    ("entered", "exited"), ("internal", "external"),
    ("interior", "exterior"), ("inward", "outward"),
    ("indoors", "outdoors"),
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
    raw = np.stack(offs).mean(axis=0)
    return raw / np.linalg.norm(raw)


def residualize(v, axes):
    r = v.copy()
    for a in axes:
        r = r - float(r @ a) * a
    nrm = np.linalg.norm(r)
    return r / nrm if nrm > 1e-12 else r


print("\n=== Building yang/yin variants ===")
yyn_axes = {}
for name, pairs in VARIANTS.items():
    yyn_axes[name] = build_axis(pairs, name)

A_val = build_axis(VALENCE_PAIRS)
A_aro = build_axis(AROUSAL_PAIRS)
A_ud = build_axis(UD_PAIRS)
A_fb = build_axis(FB_PAIRS)
A_ld = build_axis(LD_PAIRS)
A_io = build_axis(IO_CLEAN_PAIRS)


print("\n=== Pairwise cosines between yang/yin variants ===")
print(f"  (high cosine = variants agree, unified yang/yin axis)")
names = list(VARIANTS.keys())
M = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M[i, j] = float(yyn_axes[ni] @ yyn_axes[nj])
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names))
for i, ni in enumerate(names):
    print(f"  {ni:>14}" + "".join(f"  {M[i, j]:>+13.4f}" for j in range(len(names))))


print("\n=== Each variant's cosines with anchors and schemas ===")
print(f"  {'variant':>14}  {'V':>7}  {'A':>7}  {'UD':>7}  {'FB':>7}  {'LD':>7}  {'IO_clean':>8}")
for name, axis in yyn_axes.items():
    cv = float(axis @ A_val)
    ca = float(axis @ A_aro)
    cu = float(axis @ A_ud)
    cf = float(axis @ A_fb)
    cl = float(axis @ A_ld)
    ci = float(axis @ A_io)
    print(f"  {name:>14}  {cv:>+7.3f}  {ca:>+7.3f}  {cu:>+7.3f}  {cf:>+7.3f}  {cl:>+7.3f}  {ci:>+8.3f}")


print("\n=== After V+A residualization ===")
yyn_va = {name: residualize(axis, [A_val, A_aro]) for name, axis in yyn_axes.items()}
io_va = residualize(A_io, [A_val, A_aro])

print(f"  Pairwise cosines (V+A removed):")
print(f"  {'':>14}" + "".join(f"  {n:>13}" for n in names))
M_va = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        M_va[i, j] = float(yyn_va[ni] @ yyn_va[nj])
    print(f"  {ni:>14}" + "".join(f"  {M_va[i, j]:>+13.4f}" for j in range(len(names))))

print(f"\n  Each variant_VA's cosine with IO_CLEAN_VA:")
for name in names:
    print(f"    cos({name}_VA, IO_CLEAN_VA) = {float(yyn_va[name] @ io_va):+.4f}")


# Core yang/yin candidate: mean of unit-normalized variants
print("\n=== Core yang/yin candidate (mean of unit-normalized variants) ===")
core = sum(yyn_axes.values()) / len(yyn_axes)
core = core / np.linalg.norm(core)
print(f"  Built from {len(yyn_axes)} variants")
print(f"\n  Yang pole (top 20):")
for w, s in wv.similar_by_vector(core, topn=20):
    print(f"    {s:>+.4f}  {w}")
print(f"\n  Yin pole (top 20):")
for w, s in wv.similar_by_vector(-core, topn=20):
    print(f"    {s:>+.4f}  {w}")

print(f"\n  Cosines of core with everything:")
print(f"    cos(core, VALENCE)  = {float(core @ A_val):+.4f}")
print(f"    cos(core, AROUSAL)  = {float(core @ A_aro):+.4f}")
print(f"    cos(core, UD)       = {float(core @ A_ud):+.4f}")
print(f"    cos(core, FB)       = {float(core @ A_fb):+.4f}")
print(f"    cos(core, LD)       = {float(core @ A_ld):+.4f}")
print(f"    cos(core, IO_CLEAN) = {float(core @ A_io):+.4f}")

# Core after V+A removal
core_va = residualize(core, [A_val, A_aro])
print(f"\n  After V+A residualization:")
print(f"    cos(core_VA, UD_VA) = {float(core_va @ residualize(A_ud, [A_val, A_aro])):+.4f}")
print(f"    cos(core_VA, FB_VA) = {float(core_va @ residualize(A_fb, [A_val, A_aro])):+.4f}")
print(f"    cos(core_VA, LD_VA) = {float(core_va @ residualize(A_ld, [A_val, A_aro])):+.4f}")
print(f"    cos(core_VA, IO_VA) = {float(core_va @ io_va):+.4f}")

np.savez(
    "/Users/macn/Documents/embeddingexp/exp33_results.npz",
    variants={k: v for k, v in yyn_axes.items()},
    core=core,
    core_va=core_va,
)
print("\nSaved: exp33_results.npz")
