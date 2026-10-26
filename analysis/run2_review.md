# Run 2 review and stage 3 training changes

Reviewed 2026-10-26. Source: `kaggle_outputs/run2_1025/`.
Stage 3 is implemented and smoke-tested; its full FER results are not yet measured.

## Results

| Model / candidate | Validation accuracy | Validation macro F1 | Selected epoch |
| --- | ---: | ---: | ---: |
| M1 baseline | 65.8307% | 0.6133 | 68 |
| M1 lighting | 65.3431% | 0.6119 | 61 |
| M2 baseline | 67.4330% | 0.6514 | 19 |
| M2 lighting | 68.0599% | 0.6584 | 15 |
| M2 regularized | 67.8161% | 0.6490 | 19 |

Selection correctly retained M1 baseline and chose M2 lighting using validation.

| Run / selected model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| Run 1 M1 | 65.9933% | 0.6337 |
| Run 2 M1 | 65.7704% | 0.6260 |
| Run 1 M2 | 67.6930% | 0.6577 |
| Run 2 M2 | 67.0800% | 0.6573 |

The second round did not improve the selected models' test accuracy. Its modest
validation improvements should not be described as an established generalization
gain. Future candidate selection continues to use validation, never these test scores.

## Evidence for the next choices

- M1 lighting made 148 previously incorrect validation predictions correct, but
  introduced 162 new mistakes: net -14 out of 2,871. Its final training accuracy
  also remains around 66%. Adding stronger augmentation indiscriminately is not
  supported by this result.
- M2 lighting corrected 245 mistakes and introduced 227: net +18 (0.627 percentage
  points) relative to its control. It helped validation precision/recall across
  several classes, but fear F1 remained approximately unchanged.
- A paired bootstrap of validation examples gives approximate 95% intervals for
  lighting minus control accuracy of [-1.71, +0.73] percentage points for M1 and
  [-0.87, +2.09] for M2. These describe resampling uncertainty on the observed
  validation predictions, not variation across independent training seeds.
  Both span zero; small differences should be checked in another training seed
  before making strong claims.
- M2 regularized fits the training set less strongly: final train accuracy is
  78.63%, compared with lighting's 88.31%. Its validation loss is also lower
  (approximately 0.949 versus 0.992 at the end). But it does not beat lighting in
  validation accuracy, and its best accuracy is near the end at epoch 19/20.
- The regularized candidate changed erasing, weight decay and backbone LR together.
  We cannot conclude that erasing alone helped or harmed it from that experiment.
- Run 1 and run 2 controls differ despite the same seed. Loader-worker behavior
  and validation precision changed between implementations, so cross-run controls
  are not a strict reproducibility experiment. Re-evaluate archived models with
  identical full-precision validation before comparing them.

## Stage 3: combine conservatively and isolate the unresolved part

### M1

Keep the exact supplied CNN. Compare its unchanged control against gentle_combined:

1. Grayscale and resize to 48x48.
2. Horizontal flip with probability 0.5.
3. Reflect-pad by 2 pixels, then randomly crop back to 48x48.
4. Rotate up to 8 degrees and scale between 0.95 and 1.05.
5. Apply brightness/contrast jitter of 0.08 with probability 0.5.
6. Convert to tensor and normalize exactly as before.

The crop replaces affine translation rather than stacking two shifts. Compared
with the previous lighting trial, geometric and lighting perturbations are milder.
No additional erasing or dropout is introduced. This tests preservation of facial
detail, rather than assuming that more distortion is always better.

### M2

Use run 2's 20-epoch lighting configuration as the control.

- lighting_erasing combines its existing flip/affine/lighting augmentation with
  erasing probability 0.10 and area 2-6%. It keeps the original optimizer settings
  and 20-epoch schedule, isolating the extra occlusion augmentation.
- regularized_long combines the same augmentation with backbone LR 5e-5, weight
  decay 5e-4 and a fresh 40-epoch cosine schedule. This tests whether the slower
  regularized candidate benefits from more learning time. It is not a continuation
  of the old optimizer and is not a one-factor erasing ablation.

Square-root inverse-frequency class weights and label smoothing 0.05 are retained.
Validation/test/inference preprocessing, architectures, and epoch-only logging are
unchanged. Stage 2's recipes remain available as `accuracy_recipes(..., stage=2)`.

## Preserve earlier models as validation challengers

Both notebooks now write under `fer2013_accuracy_v3`. Five fits are needed without
archives. Attaching run 2's full outputs/ as BASELINE_RESULTS_ROOT skips the two
control fits, leaving three new fits. The local notebook detects the supplied
run 2 folder automatically.

REFERENCE_RESULTS_ROOTS can include run 1, for example:

```python
BASELINE_RESULTS_ROOT = Path("/kaggle/input/YOUR_RUN2_DATASET/outputs")
REFERENCE_RESULTS_ROOTS = {
    "run1": Path("/kaggle/input/YOUR_RUN1_DATASET/outputs"),
}
```

Full archives contain each model's best.pt, config.json, split_manifest.json
and training provenance. Compact archives may instead contain a shared
outputs/split_manifest.json plus sibling weights/*_best.pt; config and original
best-epoch metadata are read from the checkpoints. This supports the supplied
run 1 folder after its per-model training folders were removed. Missing histories
are marked rather than reconstructed; keep its original report for the curves.
The helper verifies train/validation membership and
re-evaluates all models in float32 on the same validation images. Earlier models
can win. A higher macro F1 never overrides a lower validation accuracy.

The Git clone does not contain ignored local Kaggle outputs. Use a fresh Kaggle
session with the updated repo and attach the archived results if reusing them.
No external upload or full training was performed as part of this implementation.

## Log check

The new log has 461 lines, down from roughly 21,000 lines in the supplied first
log. It contains zero occurrences of the previous child-process assertion and
reaches archive creation successfully. Keep NUM_WORKERS=0 and epoch-only output.

## API references

- [RandomCrop and reflection padding](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html)
- [RandomApply](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomApply.html)

API references describe transform behavior; the chosen strengths and combinations
are hypotheses based on the local run evidence, not externally established gains.
