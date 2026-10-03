# Literature check: how models represent the goal of an instruction (3 Oct 2026, morning)

Checked by Claude (Opus 5.5) before freezing the "goals in instructions"
experiment (steps 1-3 proposed at 09:25). Searched the web; read Dong et
al. 2025 in full (pages 1-8); others from abstracts/summaries.

## Already done (build on it, don't rediscover)
- **Task / function vectors** (Hendel et al. 2023, EMNLP Findings; Todd et
  al. 2024, ICLR). A few attention heads carry a compact representation of
  the task shown in few-shot examples; adding it to a zero-shot run makes
  the model do the task. = our step 2 (patching the goal), for in-context
  tasks rather than instructions.
- **Instruction steering vectors** (Stolfo et al. 2025, ICLR, Microsoft:
  "Improving Instruction-Following ... through Activation Steering").
  Difference of activations with and without an instruction (format,
  length, word inclusion); adding it makes the model follow the
  constraint. = step 2 for instructions specifically.
- **Emergent Response Planning** (Dong et al. 2025, ICML). Probes on the
  prompt representation predict global attributes of the coming response:
  length, reasoning steps, which character a story will feature, final
  multiple-choice answer, confidence. Llama-2/3, Mistral, Qwen, base and
  instruct; tuned models plan structure better. = our step 1, done.
  **Dynamics during generation (their 5.2, Fig. 7):** probe accuracy for
  an attribute NOT YET REVEALED is high at the start, dips in the middle,
  and rises toward the end: a U shape. They stop measuring at the token
  where the attribute appears.
- **Goal distance in an LLM agent** (2602.08964, GPT-OSS-20B in text grid
  worlds): goal-distance and cognitive maps decodable; goal-location info
  drops after reasoning (75% -> 60%). Spatial navigation, not tasks.
- **Task progress in robot models** (2608.13474, VLA pi-0.5): "fraction
  of the task remaining" is linearly readable and steerable. Robots, not
  text.
- **Image schemas in LLMs** (ACM 2025, "When Cognition Meets Data"):
  behavioural probes only (questions like "does progress always reach a
  destination?"). No internal representations.

## Not found (where our version is new)
1. **The goal through and past its completion.** Dong et al. track a
   not-yet-revealed attribute up to its reveal. Nobody we found tracks
   the representation of the asked-for output (a haiku) across the whole
   answer, including after the thing is finished, to ask whether it is
   SPENT (fades as achieved: Niamh's potential-energy prediction of 2 Oct)
   or HELD (flat until done, then drops). Their U shape is a warning:
   a naive "fades as it's reached" may instead look like "dips in the
   middle, returns at the end".
2. **Text progress = spatial progress.** The robot paper shows progress
   is readable in robots; nobody tests whether a model's "how close to
   done" in a text task shares a direction with "how close to the river"
   in literal travel. That is the Lakoff link (step 4) and appears new.
3. **Source/goal roles in instructions** (exp199): no prior work found.

## Consequence for the design
- Step 1 (decode the goal before generation) is a replication of Dong et
  al. on Llama-3.2-1B; keep it as the gate, not the finding.
- Step 2 (patch it) is a replication of Stolfo / Todd; keep it short, as
  the check that our goal state is causal.
- Step 3 is the contribution. Pre-register spent vs held vs U-shaped,
  with Dong et al.'s U as a named alternative.

Sources: arxiv.org/abs/2502.06258; aclanthology.org/2023.findings-emnlp.624;
proceedings.iclr.cc/paper_files/paper/2024/file/4ae163cb8788970e53b4fd9578141139-Paper-Conference.pdf;
arxiv.org/abs/2410.12877; arxiv.org/html/2602.08964; arxiv.org/html/2608.13474v1;
dl.acm.org/doi/10.1145/3783669.3783757
