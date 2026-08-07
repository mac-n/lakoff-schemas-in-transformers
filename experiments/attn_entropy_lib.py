"""attn_entropy_lib.py — shared prompts/models/helpers for the
attention-entropy experiments (exp160/160b/161). Extracted 2026-06-11
after exp160b's import of exp160 re-ran the whole experiment (exp160's
run-loop was at module level). Import-safe: definitions only, no run.
"""
import numpy as np
from lakoff_canonical_vocabulary import LAKOFF_SCHEMAS_MML
from markedness_norm_protocol import SCHEMA_NAMES, COMMON, RARE, corrf

MODELS = {
    "pythia-410m":   dict(repo="pythia-410m",            layers=[3, 8, 12, 16, 20], decision=[8, 12, 16]),
    "gpt2-medium":   dict(repo="gpt2-medium",            layers=[3, 8, 12, 16, 20], decision=[8, 12, 16]),
    "Llama-3.2-1B":  dict(repo="meta-llama/Llama-3.2-1B", layers=[2, 5, 8, 11, 13], decision=[5, 8, 11]),
}


PROMPTS = [
    "The room was quiet and the afternoon light fell across the floor.",
    "She could not understand the instructions no matter how she tried.",
    "He solved the equation quickly and moved on to the next page.",
    "A thick fog rolled in and the path ahead became impossible to see.",
    "The argument was clear, each step following plainly from the last.",
    "Everything felt muddled, the words blurring together on the screen.",
    "The child laughed and ran across the bright open field.",
    "Static filled the radio and the message was lost in the noise.",
    "We agreed on the plan and signed the papers before lunch.",
    "The meaning kept slipping away every time she reached for it.",
    "Sunlight streamed through the window and warmed the wooden desk.",
    "His thoughts were tangled and he could not find a place to begin.",
    "The recipe listed the steps in a simple and orderly way.",
    "Smoke obscured the exit and people stumbled in the dark hallway.",
    "The lecture made the whole subject suddenly obvious to everyone.",
    "Doubt crept in and the once-firm decision dissolved into confusion.",
    "The river was calm and the stones beneath were perfectly visible.",
    "A migraine pressed behind her eyes and the page swam out of focus.",
    "They mapped the route carefully and knew exactly where to turn.",
    "The contract was vague and no one could say what it required.",
    "Dawn broke and the valley emerged crisp and sharp below them.",
    "He mumbled an answer that nobody in the room could make sense of.",
    "The diagram explained the engine far better than the manual had.",
    "Shadows lengthened and the woods grew indistinct and threatening.",
    "She read the proof twice and saw at once why it was true.",
    "The crowd surged and the speaker's point was swallowed by the din.",
    "Cold rain fell steadily on the grey and featureless plain.",
    "A single clear note rang out and the whole melody fell into place.",
    "The data was inconsistent and the conclusion remained uncertain.",
    "Morning was clean and quiet and the coffee was hot on the table.",
    "His vision blurred at the edges and the street signs went unreadable.",
    "The teacher drew one line and the difficult idea became simple.",
    "Confusion spread through the office as the rumours multiplied.",
    "The lake mirrored the mountains in still and faultless detail.",
    "Everything he said contradicted what he had said the hour before.",
    "The lantern steadied and the cave walls came sharply into view.",
    "She lost the thread of the story somewhere in the third chapter.",
    "The answer was plain once the extra clutter had been cleared away.",
    "Murk and weed choked the pond until nothing could be seen below.",
    "The map was precise and every landmark stood out at a glance.",
]


def schema_words():
    s = set(COMMON + RARE)
    for sn in SCHEMA_NAMES:
        for p, n in LAKOFF_SCHEMAS_MML[sn]:
            s.add(p); s.add(n)
    return sorted(s)


def attn_entropy_per_query(pattern):
    # pattern: [n_heads, q, k]; normalised Shannon entropy per query (q>=1)
    nH, Q, _ = pattern.shape
    out = np.full(Q, np.nan)
    for q in range(1, Q):
        p = pattern[:, q, :q + 1]                  # [heads, keys]
        p = np.clip(p, 1e-12, 1.0)
        H = -(p * np.log(p)).sum(axis=1)           # per head
        out[q] = float(H.mean() / np.log(q + 1))   # normalise by log(n_keys)
    return out


def partial_corr(x, y, covars):
    """corr(x, y | covars). covars: list of arrays."""
    A = np.vstack(covars + [np.ones_like(x)]).T
    def resid(v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    return corrf(resid(np.asarray(x, float)), resid(np.asarray(y, float)))

