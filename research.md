# SwimBird Research Notes

## 1. Goal of this document

This file is a developer-oriented research note for the current `SWIMBIRD` repository. It is meant to help future work quickly answer:

- What problem the original SwimBird paper is solving.
- How the current codebase implements that idea.
- Which parts are training code, model patches, data processing, and evaluation glue.
- Where the repository likely matches the paper, and where implementation-specific assumptions exist.

This is not intended to be a polished paper summary. It is a working reference for later development.

## 2. Project summary

SwimBird is a multimodal large language model that tries to **switch reasoning mode adaptively** based on the problem:

- `text-only reasoning`
- `vision-only reasoning` via continuous latent visual thoughts
- `interleaved vision-text reasoning`

The key motivation from the paper is that many MLLMs over-rely on text CoT, while some latent-visual reasoning methods swing too far in the other direction by enforcing a fixed visual-thinking pattern. SwimBird’s claim is that the right reasoning substrate should depend on the query.

In this repository, that idea is implemented on top of Qwen-VL family models by:

- adding latent control tokens
- monkey-patching the model forward pass
- training with both text-token loss and latent-embedding loss
- preparing SFT data that contains question images, optional reasoning images, reasoning text, and final answers
- evaluating with a custom VLMEvalKit model wrapper and prompt format

## 3. Original paper: what matters for engineering

### 3.1 Paper identity

- Title: `SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs`
- arXiv: `2602.06040`
- Submitted on arXiv: `February 5, 2026`
- Paper URL: `https://arxiv.org/abs/2602.06040`
- Project page: `https://accio-lab.github.io/SwimBird/`

### 3.2 Core method from the paper

From the paper abstract and HTML version, the main technical claims are:

1. SwimBird supports three reasoning modes:
   - text-only
   - vision-only
   - interleaved vision-text
2. It uses a **hybrid autoregressive formulation**:
   - next-token prediction for textual thoughts
   - next-embedding prediction for visual thoughts
3. It uses a **dynamic latent token budget** rather than a fixed latent-token count.
4. It trains on a curated dataset, `SwimBird-SFT-92K`, covering all three reasoning patterns.

The paper’s section structure is also useful as a map for implementation:

- `3.1 Hybrid Autoregressive Modeling`
- `3.2 Dynamic Latent Token Budget`
- `3.3 Switchable Reasoning SFT Dataset Construction`
- `4.1 Main Results`
- `4.3 Analysis of Switchable Reasoning Mode`

### 3.3 Dataset composition reported by the paper

The paper reports the following composition for `SwimBird-SFT-92K`:

| Source | Total | Text Only | Vision Only | Interleave | Domain |
| --- | ---: | ---: | ---: | ---: | --- |
| Zebra-CoT | 26.3K | 0 | 5.9K | 20.4K | Visual Search, Jigsaw, Maze, Geometry, Chess |
| ThinkMorph | 7.1K | 0 | 1.2K | 5.9K | Visual Search, Spatial Navigation, Jigsaw, Chart |
| MathCanvas | 8.9K | 0 | 1.7K | 7.2K | Geometry, Algebra, Calculus, Statistics |
| OpenMMReasoner | 50K | 50K | 0 | 0 | General VQA, Math VQA, Text QA |
| Total | 92.3K | 50K | 8.8K | 33.5K | - |

This lines up closely with the training script in this repo, which points to:

- `SwimBird-ZebraCoT`
- `SwimBird-ThinkMorph`
- `SwimBird-MathCanvas`
- `SwimBird-OpenMMReasoner`

### 3.4 Results reported by the paper

The paper/project page highlights that SwimBird improves both:

- fine-grained visual understanding
- general multimodal reasoning

Numbers explicitly visible in the paper HTML include:

- `V* Bench`: `85.5`
- `HR-Bench 4K`: `79.0`
- `HR-Bench 8K`: `74.9`
- `MMStar`: `71.2`
- `RealWorldQA`: `73.1`

The paper’s analysis section also claims a benchmark-dependent mode pattern:

- text-logic datasets such as `DynaMath` and `MathVerse_MINI` mostly trigger text-only reasoning
- visually dense datasets such as `V* Bench` and `HR-Bench` trigger more vision-only or interleaved reasoning

That claim is important because later code changes should preserve not just accuracy, but also the model’s ability to choose different reasoning traces for different tasks.

## 4. Current repository structure

The repo is effectively split into two layers:

### 4.1 Main SwimBird training/inference layer

- `src/model/swimbird.py`
- `src/train/train.py`
- `src/train/monkey_patch_forward.py`
- `src/trainer/swimbird_trainer.py`
- `src/dataset/swimbird_dataset.py`
- `src/dataset/data_utils.py`
- `src/constants.py`
- `src/params.py`
- `scripts/train.sh`
- `data_process.py`

### 4.2 Evaluation layer based on VLMEvalKit

- `VLMEvalKit/vlmeval/vlm/swimbird/model.py`
- `VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`
- `VLMEvalKit/vlmeval/config.py`
- `VLMEvalKit/test.sh`
- `VLMEvalKit/slurm_scripts/run_*.sh`

In practice:

- the top-level `src/` code is where SwimBird is implemented
- `VLMEvalKit/` is how the model is registered and benchmarked

## 5. How the model is implemented in code

### 5.1 Base models

The training entrypoint supports two Qwen-VL backbones:

- `Qwen2.5-VL`
- `Qwen3-VL`

See:

- [`src/train/train.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/train.py)
- [`src/model/swimbird.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/model/swimbird.py)

The code loads the Hugging Face config first, then chooses which patched SwimBird class to instantiate.

### 5.2 Added latent control tokens

Training adds three special tokens:

- `<|latent|>`
- `<|latent_start|>`
- `<|latent_end|>`

These are registered in the processor/tokenizer during training and copied into `model.config`:

- `latent_id`
- `latent_start_id`
- `latent_end_id`
- `max_latent_token`

This is the token-level interface that connects text generation with latent visual reasoning.

### 5.3 Monkey-patched forward path

The central implementation trick is in [`src/train/monkey_patch_forward.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/monkey_patch_forward.py).

The repo replaces the stock Qwen forward methods with mixed-modality versions that accept extra arguments such as:

- `pixel_values_latent`
- `image_grid_thw_latent`
- `in_latent_mode`
- `latent_hidden_state`
- `image_out_mask`

Conceptually, the patched model does three things:

1. It still handles normal question images through the original image-token pathway.
2. It can also inject reasoning images into positions marked by latent tokens.
3. During generation, it can carry hidden states across steps while the model is in latent mode.

This is the implementation backbone of the paper’s “hybrid autoregressive” idea.

### 5.4 Hybrid loss

Training uses two supervision signals:

1. Standard language-model loss on output text.
2. Latent supervision on hidden-state embeddings for reasoning-image spans.

The latent loss is computed in the patched generation forward:

- `mse` loss, or
- cosine-similarity-based loss (`sim`)

and combined as:

`total_loss = text_loss + latent_lambda * latent_loss`

This is the code-level realization of:

- next-token prediction for text
- next-embedding prediction for visual thoughts

### 5.5 Generation-time latent mode switching

In [`src/model/swimbird.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/model/swimbird.py), generation is customized by overriding `_sample`.

Key behavior:

- the model detects entry into latent mode based on `<|latent_start|>`
- while in latent mode, emitted tokens are replaced with `<|latent|>`
- the next-step embedding is fed by the previous `latent_hidden_state`
- latent generation stops when the model emits `<|latent_end|>` or when a max latent budget is hit

This means the model is not “drawing” images directly. Instead, it autoregressively emits a latent span represented by continuous hidden states, while the token surface form is collapsed to repeated `<|latent|>`.

### 5.6 Dynamic latent budget in the current code

The paper emphasizes a dynamic latent budget. The implementation seems to realize this in two related ways:

1. **Resolution-aware visual token count**
   - question images and reasoning images use different pixel budgets
   - reasoning-image budget is tied to `max_latent_token * 32 * 32`
2. **Variable latent span length at inference**
   - the model can continue latent generation until `<|latent_end|>` is produced

Important implementation detail:

- current generation also imposes a hard cap using `self.config.max_latent_token * 2`

So the runtime behavior is dynamic but still bounded. This is a practical engineering compromise rather than “fully unbounded” latent reasoning.

## 6. How training data is represented

### 6.1 Expected raw JSON schema

The SFT dataset loader expects JSON files containing fields like:

- `conversations`
- `image`
- `reasoning_image`
- `answer`

See:

- [`src/dataset/swimbird_dataset.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/dataset/swimbird_dataset.py)

The effective sample structure is:

- user question text, optionally containing `<image>` placeholders
- problem images in `image`
- assistant reasoning text, optionally containing `<image>` placeholders
- reasoning images in `reasoning_image`
- final answer in `answer`

### 6.2 Conversion into chat format

`cot_preprocess_function` converts one sample into a multi-turn chat:

- `system`
- `user`
- `assistant`

Important formatting choices:

- reasoning text is wrapped in `<reason>...</reason>`
- final answer is wrapped in `<answer>...</answer>`
- reasoning images are inserted into assistant content as images

This mirrors the paper’s unified switchable reasoning format.

### 6.3 Why assistant images matter

The assistant-side images are the key supervision signal for latent reasoning.

The collator builds three variants of each example:

- full example
- user-images-only view
- assistant-images-only view

Then it creates:

- `pixel_values` for question images
- `pixel_values_latent` for reasoning images

This separation is a strong clue about the intended training semantics:

- question images are model inputs
- reasoning images supervise latent reasoning states

### 6.4 Token replacement and label masking

In [`src/dataset/data_utils.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/dataset/data_utils.py), several helper functions are critical:

- `replace_visual_spectial_tokens`
- `replace_latent`
- `generate_labels_after_multi_token_start`
- `mask_image_output_tokens`

What they do:

1. Replace assistant image placeholders with latent delimiters.
2. Convert interior latent spans into repeated `<|latent|>` tokens.
3. Only train labels after assistant output begins.
4. Mask image output positions so latent loss can be applied on the correct time steps.

This preprocessing pipeline is easy to break during refactors, so future changes here should be done very carefully.

## 7. Prompting and mode control

### 7.1 System instruction during training

The system prompt in [`src/constants.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/constants.py) explicitly teaches the model:

- `<reason>` is textual mode
- `<|latent_start|> ... <|latent_end|>` is visual mode
- outputs may mix both modes
- the final answer must be wrapped in `<answer>`

This is important: the reasoning-mode behavior is not only architectural, but also instruction-conditioned.

### 7.2 Evaluation prompts

The VLMEvalKit wrapper in [`VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/prompt.py) injects dataset-specific prompts that explicitly request:

- textual thought in `<reason>`
- visual thought in `<|latent_start|> ... <|latent_end|>`
- final answer in `<answer>`

The prompting differs slightly by dataset type:

- MCQ
- Y/N
- VQA
- MMMU

This means benchmark performance is partially dependent on prompt design, not only model weights.

## 8. Training workflow in this repo

### 8.1 Main training entrypoint

The main entry is:

- [`scripts/train.sh`](/data/chaoyiz/workspace/code/SWIMBIRD/scripts/train.sh)

Observed default setup:

- base model: `Qwen/Qwen3-VL-8B-Instruct`
- data sources: the four SwimBird-SFT subsets
- deepspeed: `scripts/zero2.json`
- BF16 enabled
- global batch size: `128`
- `freeze_vision_tower=True`
- `freeze_merger=True`
- `freeze_llm=False`
- `latent_loss=mse`
- `latent_lambda=0.2`
- `max_latent_token=32`
- image token budget up to `16384`

Interpretation:

- current default training primarily updates the language side
- the vision tower and merger are frozen in the default script
- this may be a deliberate efficiency/stability choice, but it is also a meaningful implementation decision that may differ from alternative reproduction settings

### 8.2 Dataset path rewriting

`data_process.py` is a small but important utility. It rewrites `image` and `reasoning_image` paths inside dataset JSON files by prefixing them with the absolute dataset root.

Without this step, multimodal loading will likely fail at runtime.

## 9. Evaluation workflow in this repo

### 9.1 VLMEvalKit integration

The SwimBird model is registered inside:

- [`VLMEvalKit/vlmeval/config.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/config.py)

The config currently exposes:

- `SwimBird-SFT-8B`
- `SwimBird-SFT-8B_retrain`
- `SwimBird-SFT-2B_retrain`
- `SwimBird-SFT-2B-Thought0-Latent`
- various checkpoint aliases

This suggests the repo is already being used not just for the released model, but also for internal retraining and checkpoint sweeps.

### 9.2 SwimBird evaluation model wrapper

The evaluation wrapper:

- loads the processor
- patches Qwen3-VL forward functions
- instantiates `SwimBird_Qwen3VL`
- supports regular HF inference, `vLLM`, and `lmdeploy`

See:

- [`VLMEvalKit/vlmeval/vlm/swimbird/model.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/model.py)

Notable engineering implication:

- evaluation depends on monkey patching too, so inference code and training code are tightly coupled

### 9.3 Benchmarks used in this repo

`VLMEvalKit/test.sh` and the SLURM scripts show the main target benchmarks:

- `DynaMath`
- `WeMath`
- `MathVerse_MINI`
- `HRBench4K`
- `HRBench8K`
- `VStarBench`
- `MMStar`
- `RealWorldQA`

This is broadly consistent with the paper’s emphasis on:

- logic-heavy multimodal reasoning
- fine-grained visual understanding

### 9.4 Judge model assumption

The top-level README states that evaluation uses an **LLM-based API judge**, with `gpt-4o-0806` as the default judge in the project description.

In local SLURM scripts, the placeholder is still:

- `--judge your_judge_model`

So anyone re-running evaluation needs to supply a real judge backend and should not assume exact-match scoring is the canonical setup.

## 10. Where the code maps well to the paper

The current codebase appears to align well with the paper in these places:

1. **Three reasoning modes are explicit**
   - text reasoning via `<reason>`
   - visual reasoning via latent spans
   - mixed mode via interleaving both
2. **Hybrid autoregressive training is real**
   - text uses token loss
   - visual latent spans use embedding loss
3. **Reasoning images are first-class supervision**
   - dataset schema and collator both preserve them carefully
4. **Dynamic budget idea is implemented**
   - different image budgets plus latent stop token
5. **Benchmark-specific prompting exists**
   - consistent with switchable reasoning analysis in the paper

## 11. Implementation-specific caveats and open questions

These are the most important caveats I see for future development.

### 11.1 Runtime cap differs from purely conceptual “dynamic”

The paper frames latent length as dynamic, but the code enforces:

- `max_latent_num = self.config.max_latent_token * 2`

This is reasonable, but if future work studies true latent-budget scaling, this cap will matter.

### 11.2 Default training freezes the visual stack

The default `scripts/train.sh` sets:

- `freeze_vision_tower=True`
- `freeze_merger=True`

If paper reproduction assumes broader finetuning, results may differ. This should be validated before drawing research conclusions from new runs.

### 11.3 Evaluation currently appears Qwen3-centric

Although training supports both Qwen2.5-VL and Qwen3-VL, the VLMEvalKit wrapper explicitly imports and instantiates `SwimBird_Qwen3VL`.

So evaluation code may not be as backbone-agnostic as the training entrypoint.

### 11.4 Prompt format is part of the method

The `<reason>`, `<|latent_start|>`, and `<answer>` conventions are wired into:

- system prompt
- dataset conversion
- evaluation prompts

Changing any one of these without changing the others will likely degrade performance or silently break mode selection.

### 11.5 Path assumptions are environment-specific

Some evaluation configs contain hard-coded absolute paths for checkpoint roots. That is fine for local experiments but should be normalized if the repo is to be shared or deployed more broadly.

## 12. Recommended developer entry points

If future work focuses on:

### 12.1 Changing reasoning behavior

Start with:

- [`src/constants.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/constants.py)
- [`src/dataset/swimbird_dataset.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/dataset/swimbird_dataset.py)
- [`VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/prompt.py)

### 12.2 Changing latent modeling or loss

Start with:

- [`src/train/monkey_patch_forward.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/monkey_patch_forward.py)
- [`src/model/swimbird.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/model/swimbird.py)
- [`src/params.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/params.py)

### 12.3 Changing training recipe

Start with:

- [`scripts/train.sh`](/data/chaoyiz/workspace/code/SWIMBIRD/scripts/train.sh)
- [`src/train/train.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/train.py)
- [`src/trainer/swimbird_trainer.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/trainer/swimbird_trainer.py)

### 12.4 Changing evaluation or benchmark registration

Start with:

- [`VLMEvalKit/vlmeval/vlm/swimbird/model.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/model.py)
- [`VLMEvalKit/vlmeval/config.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/config.py)
- [`VLMEvalKit/slurm_scripts/`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/slurm_scripts)

## 13. Suggested follow-up checks for future work

Before major development, it would be useful to verify:

1. Whether the released paper setting matches the current default training freeze policy.
2. Whether latent spans are generated as expected across all target benchmarks.
3. Whether the 2B and 8B branches share identical prompt formatting and loss behavior.
4. Whether any recent changes in `transformers` or Qwen model internals could break the monkey patches.
5. Whether benchmark prompts in VLMEvalKit still match the paper’s intended evaluation protocol.

## 14. Sources

### External references

- arXiv abstract / paper landing page: `https://arxiv.org/abs/2602.06040`
- arXiv HTML paper: `https://arxiv.org/html/2602.06040`
- project page: `https://accio-lab.github.io/SwimBird/`

### Repository references

- [`README.md`](/data/chaoyiz/workspace/code/SWIMBIRD/README.md)
- [`src/train/train.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/train.py)
- [`src/train/monkey_patch_forward.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/train/monkey_patch_forward.py)
- [`src/model/swimbird.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/model/swimbird.py)
- [`src/dataset/swimbird_dataset.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/dataset/swimbird_dataset.py)
- [`src/dataset/data_utils.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/dataset/data_utils.py)
- [`src/constants.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/constants.py)
- [`src/params.py`](/data/chaoyiz/workspace/code/SWIMBIRD/src/params.py)
- [`scripts/train.sh`](/data/chaoyiz/workspace/code/SWIMBIRD/scripts/train.sh)
- [`data_process.py`](/data/chaoyiz/workspace/code/SWIMBIRD/data_process.py)
- [`VLMEvalKit/vlmeval/vlm/swimbird/model.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/model.py)
- [`VLMEvalKit/vlmeval/vlm/swimbird/prompt.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/vlm/swimbird/prompt.py)
- [`VLMEvalKit/vlmeval/config.py`](/data/chaoyiz/workspace/code/SWIMBIRD/VLMEvalKit/vlmeval/config.py)

