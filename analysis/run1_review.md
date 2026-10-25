# First FER-2013 run: findings and proposed next experiments

Reviewed: `kaggle_outputs/run1_1025/` on 2026-10-25.
Sources: saved configs, training histories, classification reports, confusion
matrices, error galleries and the Kaggle execution log. No new FER training or
validation experiment was performed during this review. The training code and
notebooks have not been modified by this review.

## Measured baseline

| Model | Test accuracy | Test macro F1 | Best epoch | Best validation accuracy | Selected validation macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 supplied CNN | 65.99% | 0.6337 | 72 / 80 | 66.11% | 0.6268 |
| M2 adapted ResNet-18 | 67.69% | 0.6577 | 18 / 20 | 68.55% | 0.6569 |

Both models meet the lab's test-accuracy targets. The downloaded M2 run used
**20 epochs**, not the 40 originally suggested in the local notebook.

CPU model-forward benchmarks were 449.6 FPS for M1 and 36.2 FPS for M2 on Kaggle's
CPU. They do not measure complete application throughput or the student's laptop.

## The main weakness is fear/sad separation

| Class | Training images | Current normalized loss weight | M1 test F1 | M2 test F1 | M1 test recall | M2 test recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| angry | 3,596 | 0.8056 | 0.5826 | 0.5836 | 59.08% | 58.14% |
| disgust | 392 | 2.4399 | 0.6222 | 0.6368 | 63.06% | 63.96% |
| fear | 3,687 | 0.7956 | 0.4422 | 0.5368 | 35.64% | 49.90% |
| happy | 6,493 | 0.5995 | 0.8640 | 0.8701 | 86.81% | 87.03% |
| neutral | 4,469 | 0.7226 | 0.6304 | 0.6326 | 75.59% | 69.26% |
| sad | 4,347 | 0.7327 | 0.5110 | 0.5430 | 46.43% | 52.13% |
| surprise | 2,854 | 0.9042 | 0.7833 | 0.8014 | 82.43% | 80.87% |

The class distribution is uneven: happy has 16.6 times as many training images as
disgust. But class imbalance is already handled by square-root inverse-frequency
loss weights, calculated only from training labels. A disgust example receives
about 4.07 times the target-class weight of a happy example before label smoothing.

Disgust's F1 is substantially better than fear's despite having far fewer examples.
That does not prove the current weighting is optimal, but it argues against treating
missing class balancing as the main explanation for the weak macro F1.

Notable errors:

- M1: 327 sad images predicted neutral; 185 fear predicted sad; 173 fear predicted neutral.
- M2: 268 sad predicted neutral; 163 fear predicted sad; 161 neutral predicted sad.
- M1 predicts neutral 1,724 times although only 1,233 test images are labelled neutral.
  M2 reduces this excess to 1,467 predictions.

The model has difficulty distinguishing expressions with similar facial cues. Merely
duplicating the smallest class does not directly address that difficulty. Test-set
errors are diagnostic evidence; they must not be used to tune class-specific weights
or to choose the winning experiment.

## What the curves say

### M1: gradual learning and a late plateau

M1 completes 80 epochs; its lowest validation loss is at epoch 71, and highest
validation accuracy at epoch 72. At epoch 80, train accuracy is 66.04% and
validation accuracy is 65.69%.

Train measurements include dropout and augmentation while validation does not.
Their similar values therefore cannot, by themselves, establish underfitting.
There is no obvious generalization gap like M2's. Keep the specified architecture
and avoid adding strong regularization to M1 without a controlled test.

### M2: a substantial generalization gap

At epoch 20, train accuracy is 90.14%, versus 68.34% validation accuracy. Validation
cross entropy is lowest at epoch 7 (0.9371), then reaches 1.0155 by epoch 20 while
training loss continues falling. This is consistent with overfitting and/or growing
confidence in the remaining validation mistakes.

However, accuracy and macro F1 are both highest at epoch 18. Do not retrospectively
choose epoch 7 solely because its loss is smallest: loss and correct-label metrics
measure different things. Extending the same recipe is not the first change to try.

## Proposed experiment order

Use the same split manifest and fixed epoch budget for each direct comparison.
Keep each run in a separate output directory. Change one factor at a time, first
for M2, then transfer a successful change to M1 only if validation supports it.

| Experiment | Change from control | Purpose |
| --- | --- | --- |
| A | Existing augmentation + brightness/contrast jitter, each 0.15 | Reduce sensitivity to illumination/contrast without removing facial features |
| B | A + small random erasing, p=0.10, area 2-6% | Test mild occlusion regularization for M2 |
| C | Best augmentation + M2 AdamW weight decay 5e-4 instead of 1e-4 | Test a modest increase in parameter regularization |
| D | Best preceding recipe + M2 backbone LR 5e-5 instead of 1e-4; keep head LR unchanged | Test gentler fine-tuning of pretrained features |
| E | Best recipe + softened weighted sampling instead of weighted loss | Compare class exposure versus class-dependent gradient weighting |

For A-D, retain the current loss weights, label smoothing 0.05, architecture,
head dropout, split and batch size. Keep M2 at the actual 20-epoch budget initially
so a gain cannot simply be attributed to a different schedule or longer training.
If validation still improves at the end under the winning recipe, then test a
longer schedule as a separate experiment. These are hypotheses, not measured gains.

### Augmentation details

Keep horizontal flipping and the existing mild affine transform. Add this **before
ToTensor and normalization**:

```python
transforms.ColorJitter(brightness=0.15, contrast=0.15)
```

Although named ColorJitter, brightness and contrast also apply to grayscale images.
Hue and saturation are unnecessary here.

For experiment B, add this **after tensor conversion and normalization**:

```python
transforms.RandomErasing(
    p=0.10, scale=(0.02, 0.06), ratio=(0.5, 2.0), value=0.0,
)
```

In the current normalized coordinates, zero represents mid-gray. Start with a
small erased region: 48x48 images have little spare detail. Large crops, large
erased patches, vertical flips and aggressive rotations can destroy the expression
cues we want the model to learn. Inspect a training-only augmentation grid before
running the full experiment. Validation/test/inference preprocessing stays unchanged.

### Class balancing experiment

Keep square-root inverse-frequency weighted loss as the default control. If testing
weighted sampling, use training-only sample weights proportional to `1/sqrt(n_class)`,
draw approximately one training-set-sized epoch with replacement, and use unweighted
cross entropy. The expected class exposure becomes proportional to `sqrt(n_class)`:
this softens the imbalance rather than forcing perfectly uniform classes.

Do not combine the sampler with the existing inverse-frequency loss weights in
the first comparison. Their effects compound and would make it unclear which
change helped. Uniform inverse-frequency sampling is a later experiment if needed;
it repeats the 392 disgust examples much more heavily.

Do not begin with class-specific fear/sad weights chosen from these test results.
Do not assume focal loss will help: focusing on hard examples can also focus on
ambiguous or mislabeled examples. Some gallery examples are low-detail, non-frontal
or atypical face images, which makes indiscriminate hard-example emphasis risky.

## Selection and evaluation

- Record validation accuracy, macro F1, per-class precision/recall/F1, and confusion
  matrices for every candidate, not just a single aggregate number.
- Save separate best-validation-accuracy and best-validation-macro-F1 checkpoints
  in future runs. Pick the deployment checkpoint using validation before final test.
- An equal-weight mean of validation accuracy and macro F1 selects epoch 72 for M1
  and 18 for M2 in this run: exactly the existing accuracy-selected checkpoints.
  Switching the selection rule alone does not improve this baseline.
- Improving accuracy and macro F1 simultaneously is possible but not guaranteed;
  increasing minority-class recall can reduce its precision or overall accuracy.
  Prefer a candidate that improves both validation metrics. If none does, inspect
  the tradeoff and choose it explicitly rather than claiming a universal improvement.
- Validation disgust has only 44 examples. One changed prediction shifts that
  class's recall by 2.27 percentage points. Tiny aggregate improvements need more
  than one stochastic run to be convincing. For finalists, keep the split fixed
  and repeat training with a different model/augmentation seed.
- Evaluate the chosen final recipe on test once. Because the existing test set
  has already been inspected, acknowledge that fact when reporting future results.
  Avoid iterating against test scores or deleting awkward test images.

Optional low-cost follow-up: compare original versus horizontal-flip-averaged
predictions on validation using the saved checkpoints. This needs no retraining,
but two-view inference performs twice as many model forward passes, must also be
used in the application if reported as deployment behavior, and needs a new FPS
measurement. Do not present its scores as the original single-view model's scores.

## Runtime log

The execution completed both training runs, evaluation and packaging. The log has
1,102 occurrences of `Exception ignored in:` and 2,204 of the child-process assertion
text (the latter includes repeated trace text). No `RuntimeError` or CUDA-out-of-memory
message was found by literal search. This supports treating the visible spam as a
worker-cleanup problem, rather than an explanation for the observed class errors;
it does not prove every runtime detail was healthy.

The current local Kaggle notebook already sets NUM_WORKERS=0. Keep that for the
next run. Rebuild loaders with that setting and use a fresh session if stale worker
cleanup persists. Switching `from tqdm.auto import tqdm` to `from tqdm import tqdm`
is an additional reported workaround. AMP can remain enabled. A fresh `fit()` still
starts from scratch; preserve existing result folders and checkpoints.

## References

- [Torchvision ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html)
- [Torchvision RandomErasing](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html)
- [Matching DataLoader/tqdm cleanup issue](https://github.com/pytorch/pytorch/issues/53703)

The specific augmentation strengths and experiment ordering above are engineering
hypotheses based on this run, not claims established by those API documents.
