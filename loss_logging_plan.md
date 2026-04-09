# Loss Logging Plan

## Goal

Add separate logging for:

- total training loss
- CE loss
- latent loss (`mse` or `1 - cosine similarity`, depending on `latent_loss`)

The immediate target is to make these values visible in `wandb` runs, while keeping `Trainer` compatibility and avoiding fragile hacks.

## Current finding

After checking the current codebase:

1. The patched model forward computes:
   - CE loss via `self.loss_function(...)`
   - latent loss via `mse` or `sim`
   - final loss via `loss = ce_loss + latent_lambda * latent_loss`
2. But only the aggregated `loss` is returned in the model output.
3. `SwimBirdSFTTrainer` does not currently override:
   - `compute_loss`
   - `log`
   - `training_step`
   - any callback hook for custom metric logging
4. Therefore, the default Hugging Face Trainer logging path will only see the final `loss`, not the loss components.

Conclusion:

- yes, the current implementation appears to log only the total loss
- CE and MSE are not separately exposed to `wandb` today

## Important clarification

The default top-level training script is not currently using `wandb`:

- `scripts/train.sh` uses `--report_to tensorboard`

There is a separate script that does use `wandb`:

- `scripts/train_2b.sh` uses `--report_to wandb`

So the logging feature should ideally work for both TensorBoard and Weights & Biases, because it will flow through `Trainer.log(...)`.

## Recommended implementation

### Plan A: Return component losses from model output, then log them in Trainer

This is the cleanest approach.

#### Step 1

Extend the custom model output dataclasses in:

- `src/train/monkey_patch_forward.py`

Add optional fields such as:

- `ce_loss`
- `latent_loss`
- `weighted_latent_loss`

Notes:

- `latent_loss` should mean the raw component before multiplying by `latent_lambda`
- `weighted_latent_loss` can be useful because it explains the actual contribution to total loss
- when no latent supervision is present, these values should be `None` or zero in a consistent way

#### Step 2

In both Qwen2.5-VL and Qwen3-VL patched generation forward functions:

- compute `ce_loss` explicitly
- compute `latent_loss` explicitly
- compute `weighted_latent_loss = latent_lambda * latent_loss`
- compute `loss = ce_loss + weighted_latent_loss`
- return all of them in the output object

This keeps the logic centralized at the true source of the losses.

#### Step 3

Override `compute_loss` in:

- `src/trainer/swimbird_trainer.py`

Behavior:

- call the model normally
- read `outputs.loss`
- if present, detach and cache:
  - `ce_loss`
  - `latent_loss`
  - `weighted_latent_loss`
- return the normal training loss to preserve HF Trainer behavior

#### Step 4

Override `log` in `SwimBirdSFTTrainer` so that whenever Trainer emits a logging event, we append cached component metrics into the `logs` dict, for example:

- `train/loss_total`
- `train/loss_ce`
- `train/loss_latent`
- `train/loss_latent_weighted`

This should make them appear in:

- wandb
- tensorboard
- trainer state history

### Why this plan is preferred

- minimal disruption to existing training flow
- works with current HF Trainer logging
- keeps the model as the source of truth for loss decomposition
- easy to reuse later for eval logging if needed

## Alternative options considered

### Option B: Log directly inside model forward

Not recommended.

Reasons:

- model code should not depend on a logger backend
- can break under distributed training
- easy to duplicate logs unexpectedly

### Option C: Recompute CE and latent loss inside Trainer

Not recommended unless necessary.

Reasons:

- duplicates the model logic
- increases maintenance burden
- risks drift if forward loss logic changes later

## Naming proposal

Suggested logged keys:

- `loss`
- `loss_ce`
- `loss_latent`
- `loss_latent_weighted`

If we want to avoid collisions with HF’s built-in `loss`, then log:

- `train/loss_ce`
- `train/loss_latent`
- `train/loss_latent_weighted`

and leave the built-in `loss` untouched.

Recommendation:

- keep HF’s built-in `loss` as-is
- add:
  - `train/loss_ce`
  - `train/loss_latent`
  - `train/loss_latent_weighted`

## Edge cases to handle

1. Samples or batches without latent supervision
   - raw latent loss may be absent
   - log `0.0` or skip consistently
2. `latent_loss == "sim"`
   - logged `train/loss_latent` should still reflect the final optimized raw component, i.e. `1 - cosine_similarity`
3. Gradient accumulation
   - metrics should reflect the same scale/step semantics as Trainer’s logged `loss`
4. Distributed training / deepspeed
   - only rely on Trainer logging path, avoid manual backend logging calls

## Validation plan

After implementation:

1. Run a very short training job with `--logging_steps 1`.
2. Confirm logs contain:
   - total loss
   - CE loss
   - latent loss
   - weighted latent loss
3. Confirm values satisfy approximately:

`loss ~= ce_loss + loss_latent_weighted`

4. Confirm this works in the current logging backend:
   - TensorBoard for `scripts/train.sh`
   - wandb for `scripts/train_2b.sh`

## Scope of the first code change

For the first implementation pass, only do:

1. expose loss components from model outputs
2. log them through `SwimBirdSFTTrainer`

Do not yet do:

- eval-time metric logging
- per-dataset loss breakdown
- image-token-count diagnostics
- mode-distribution logging

## Files likely to change

- `src/train/monkey_patch_forward.py`
- `src/trainer/swimbird_trainer.py`

Possibly:

- `scripts/train.sh`
- `scripts/train_2b.sh`

only if we want to standardize the default logging backend afterward.

