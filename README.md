# Embodied Cognition in Transformers

Do Lakoff's image schemas (UP-DOWN, BALANCE, PATH, FORCE...) exist inside language models trained on text alone? This repository holds the complete experimental record, every script, raw output, pre-registration, and lab-notebook volume, of a project that tested that question on three model families.

**Read the write-up:** the blog post is [`index.md`](index.md), also served as a web page via GitHub Pages. It is a v0.2 working draft: the research was conducted collaboratively with Claude, who drafted much of the exposition from the shared lab notebooks; interpretation and writing are still being revised.

## The findings, briefly

The method: schema directions are recoverable from bare single-word activations, once a severe frequency confound is stripped (raw contrast vectors are ~99% frequency direction). Then:

1. **Steering on UP shifts valence.** Injecting the UP direction causes metaphor-congruent valence shifts, beating a random-direction control by +0.72 to +0.94, dose-responsively.
2. **The schemas form a relational system.** The predicted inter-schema couplings average +0.21; unpredicted pairs sit at +0.00. The configuration persists across the depth of Pythia 410M.
3. **The transformer reorganises inflectional morphology onto the schema axes.** Off-the-shelf static embeddings (GloVe, word2vec) learned from textual distribution develop a different inflectional geometry: the BALANCE markedness sink is about −0.38 in Pythia and ~0.00 in the static spaces.
4. **BALANCE is coupled to the model's own computational operations,** with the carrier varying by model: residual norm in Pythia, attention entropy in GPT-2 and Llama. Causal evidence runs from operation to concept; the reverse test found nothing specific. A candidate case of multiple realizability.

The claim is structural, not phenomenological: bodily origin, bodily implementation, and embodied-style structure are not the same thing, and these models force the distinction.

## Repository layout

| Path | Contents |
|---|---|
| `index.md` | The blog post (canonical write-up) |
| `figures/` | All experiment figures |
| `experiments/` | Every experiment script, `exp1` through `exp172` plus supporting libraries |
| `results/` | Raw text outputs for each experiment, as produced at run time |
| `prereg/` | Pre-registration documents, committed before code execution, with frozen checksummed prompt sets |
| `notebooks/` | The lab notebook, five volumes, plus reference and planning documents |
| `data/` | Curated schema vocabularies and corpus reports |

The scripts were written and run in a single flat working directory; `experiments/` and `results/` are separated here for browsability. To re-run an experiment, place the script alongside the results and data files it reads (scripts also import sibling experiment modules by name). Large regenerable binaries (`.pt`/`.npz` activation caches and result tensors) are not included; every script that produced them is.

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python experiments/exp123_relational_structure.py   # example: the schema-system result
```

Models are fetched from the HuggingFace Hub on first run (Pythia 70M–1.4B, GPT-2 medium, Llama-3.2-1B; static spaces via gensim). Experiments were run on Apple Silicon (MPS); seeds and probed layers are set inside each script.

## Method culture

The project's standing rule: when a finding seems big, name the control that would falsify it, then run it. The record includes the kills alongside the survivors: two of three behavioural measures in the original steering battery, a bulletproof-looking Pythia entropy band, the full-collapse version of the norm-strip analysis, and all eight mechanistic hypotheses died under their own controls. The pre-registrations in `prereg/` contain committed point predictions written before code execution; the confirmatory runs assert frozen prompt checksums at runtime.

The research was conducted as a collaboration between Niamh McAllister and Claude (Anthropic) across many sessions, with continuity maintained through the lab-notebook tradition in `notebooks/` and handoff documents between sessions. A small number of personal and machine-local notes are redacted from the public notebook copies; all scientific content is unmodified.

## Citation

If you use this work, please cite the blog post:

> McAllister, N. (2026). *Embodied Cognition in Transformers: image schemas, metaphorical mappings, and computational grounding in language models trained on text alone.*

A paper version is in preparation.

## License

Code is MIT-licensed; written content and figures are CC BY 4.0. See `LICENSE` and `LICENSE-CONTENT.md`.
