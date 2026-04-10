# ZebraCoT THOUGHT 0 Implicit Planning Experiment Plan

## 1. Goal

Run a first controlled experiment to test whether SwimBird can internalize the **first reasoning step** on ZebraCoT by replacing explicit `THOUGHT 0` text with a new hidden planning span.

This is a **small-scope mechanism validation**, not a full method rewrite.

The experiment should answer:

- Can the model still reason correctly if `THOUGHT 0` is removed from visible text supervision?
- Can a new planning token span act as a hidden prefix reasoning channel without collapsing into an obvious shortcut?
- Can we do this while preserving the existing SwimBird behavior for later visible text reasoning and visual latent reasoning?

## 2. Scope of V1

This first version is intentionally narrow.

### Included

- only `ZebraCoT`
- only samples with explicit `THOUGHT 0` marker
- only the first reasoning step is hidden
- later reasoning text remains visible
- later reasoning images remain supervised by the original SwimBird latent-image loss

### Excluded

- `ThinkMorph`
- `MathCanvas`
- `OpenMMReasoner`
- hiding `THOUGHT 1+`
- dynamic planning-length prediction
- direct supervision on the new planning span
- mixed-data training with the other three datasets

## 3. New token design

We will introduce a new reasoning-mode entry for hidden planning:

- `<|plan_start|>`
- `<|plan_end|>`

Inside the planning span, we will **reuse** the existing body token:

- `<|latent|>`

So the training-time replacement will look like:

```text
<|plan_start|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|plan_end|>
```

### Why this design

- do not reuse `<reason>` because it carries explicit-language bias
- do not reuse `<|latent_start|>` because it carries visual-latent bias
- reuse `<|latent|>` internally because it is only the carrier token for recurrent hidden-state propagation

## 4. Planning span length

### Default

- fixed planning length = `8`

### Requirement

- this must be configurable from one place so later ablations can test other values such as `4`, `8`, `16`

### V1 choice

- use `8` for the first run
- do not make it dynamic in V1

## 5. Data strategy

### 5.1 Offline preprocessing

Do not mutate the original ZebraCoT file in place.

Instead:

- create a new offline-processed ZebraCoT JSON
- keep the original file untouched
- use the new JSON as the training input for this experiment

This makes the experiment:

- reproducible
- easy to diff
- easy to disable or roll back

### 5.2 New dataset artifact

Planned new artifact:

- one new ZebraCoT JSON file containing the modified assistant reasoning strings

The exact path/name can be decided during implementation, but it should clearly indicate:

- ZebraCoT
- plan-token experiment
- fixed plan length

Example naming direction:

- `Zebra-CoT-plan8.json`

## 6. THOUGHT 0 extraction rule

### 6.1 Source of truth

Use the existing ZebraCoT textual markers already present in the raw dataset:

- `THOUGHT 0:`
- `THOUGHT 1:`
- `THOUGHT 2:`
- ...

### 6.2 V1 extraction rule

For each ZebraCoT sample:

1. locate `THOUGHT 0:`
2. identify the textual span belonging to `THOUGHT 0`
3. replace only that span with the planning span
4. preserve everything after that according to the original sequence order

### 6.3 Important rule

V1 should use a **single deterministic parsing rule**.

Do not mix multiple heuristics per sample.

That means once the boundary policy is chosen, it must apply uniformly.

### 6.4 Current intended boundary policy

The most natural initial policy is:

- `THOUGHT 0` starts at `THOUGHT 0:`
- `THOUGHT 0` ends right before `THOUGHT 1:`, if `THOUGHT 1:` exists

If a sample structure is more complex, that sample should be:

- either handled by a clearly defined fallback
- or skipped from V1 preprocessing

V1 should prefer cleanliness over aggressive coverage.

## 7. Replacement behavior

### 7.1 Original form

Typical ZebraCoT assistant reasoning looks like:

```text
THOUGHT 0: ...
THOUGHT 1: ...
<image>
THOUGHT 2: ...
...
```

### 7.2 New form

After preprocessing, the first step should become:

```text
<|plan_start|><|latent|>x8<|plan_end|>
THOUGHT 1: ...
<image>
THOUGHT 2: ...
...
```

Then the normal SwimBird preprocessing can continue to:

- remove `THOUGHT n:` labels from the visible text pieces
- wrap visible text pieces in `<reason>...</reason>`
- keep assistant reasoning images as the original visual-latent supervision path

### 7.3 Important semantic rule

The new plan span is **not** visible reasoning text.

So it must not be wrapped as `<reason>...</reason>`.

## 8. Loss behavior

### 8.1 Planning span

The new plan span must receive:

- no text CE supervision
- no visual latent MSE supervision

It is an unsupervised hidden planning prefix.

### 8.2 Visible text after THOUGHT 0

All later visible reasoning text continues to use:

- normal CE loss

### 8.3 Reasoning images after THOUGHT 0

All later reasoning-image latent spans continue to use:

- original SwimBird visual latent loss (`mse` or configured latent loss)

### 8.4 Final answer

Final answer continues to use:

- normal CE loss

## 9. Core implementation constraint

Even though the plan span reuses `<|latent|>` internally, it must **not** be confused with the existing image-backed latent spans.

So implementation must distinguish:

- planning latent span
- visual latent span

This distinction matters for:

- label masking
- image latent mask generation
- latent loss application
- mode switching during forward/generation if needed

## 10. First-pass implementation plan

### Step 1: offline ZebraCoT preprocessing

Create a preprocessing script that:

- reads the original ZebraCoT JSON
- detects samples containing `THOUGHT 0`
- replaces the `THOUGHT 0` text span with:
  - `<|plan_start|>`
  - `N` copies of `<|latent|>`
  - `<|plan_end|>`
- writes a new JSON file

The script should:

- accept configurable `plan_length`
- print summary statistics
- not overwrite the original file

### Step 2: tokenizer/model token registration

Add new special tokens:

- `<|plan_start|>`
- `<|plan_end|>`

These must be registered in training similarly to how latent tokens are currently registered.

### Step 3: dataset processing support

Update dataset preprocessing so it:

- preserves the new planning span in assistant content
- does not convert it into visible reason text
- can distinguish plan latent tokens from image latent tokens for masking

### Step 4: masking/loss routing

Update masking logic so:

- plan span is excluded from CE labels
- plan span is excluded from image latent loss
- later image latent spans still contribute to image latent loss

### Step 5: training recipe

Run a ZebraCoT-only training experiment using the processed file.

## 11. Training data policy for V1

Use only ZebraCoT in the first run.

This is important because:

- the step boundary is explicit there
- the experiment question is specifically about `THOUGHT 0`
- mixing with the other three datasets would make failure analysis harder

## 12. Ablation directions after V1

If V1 runs successfully, the next ablations should be:

1. plan length
   - `4`
   - `8`
   - `16`

2. training mixture
   - ZebraCoT only
   - ZebraCoT + one additional dataset unchanged

3. more hidden steps
   - hide only `THOUGHT 0`
   - hide `THOUGHT 0` and `THOUGHT 1`

4. token design
   - `plan_start/plan_end + latent body`
   - if needed later, a dedicated plan-body token

## 13. Success criteria

V1 should be considered promising if:

- training is stable
- later visible reasoning text remains coherent
- final task accuracy does not collapse
- the model does not obviously degenerate into a trivial shortcut

The purpose is not to prove the full method immediately. The purpose is to validate that hidden planning for `THOUGHT 0` is trainable inside SwimBird without breaking the existing text/image reasoning pipeline.

## 14. Non-goals for V1

V1 does **not** attempt to prove:

- that the model fully internalizes all early reasoning
- that the plan span is semantically interpretable
- that the best token design has already been found
- that the approach generalizes to all four training datasets

Those are later questions.

## 15. Summary

The first experiment is:

- new tokens: `<|plan_start|>`, `<|plan_end|>`
- plan body: `8` repeated `<|latent|>` tokens
- dataset: ZebraCoT only
- preprocessing: offline new ZebraCoT JSON
- replacement target: explicit `THOUGHT 0`
- plan supervision: none
- later text: CE as usual
- later reasoning image: MSE as usual

This gives a clean, minimally confounded starting point for testing hidden planning in SwimBird.

