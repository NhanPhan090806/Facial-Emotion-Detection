> Historical record. Paths and model choices below describe an earlier run; use the root README for the current layout.

# Run 3: model size, latest results, and the next accuracy experiment

Reviewed 2026-10-27 from `kaggle_outputs/run3_1026/`.
This review does not modify the training notebooks or deploy an ensemble.

## Measured results

| Model | Parameters | Test accuracy | Test macro F1 | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 supplied CNN | 980,711 | 66.4113% | 0.6415 | 66.6667% | 0.6397 |
| M2 adapted ResNet-18 | 11,173,831 | 66.1744% | 0.6488 | 68.1644% | 0.6525 |

M1 correctly classifies 4,767 test images; M2 classifies 4,750. The difference is
17 images out of 7,178, too small to conclude from this one run that M1 is generally
more accurate. M2 retains a small macro-F1 advantage.

The first run's M2 test accuracy was 67.6930% and macro F1 0.6577. The latest
regularized M2 recipe has regressed from that result. M1 improved modestly from its
first-run 65.9933% / 0.6337. The narrowing gap reflects both effects; it is not
evidence that model size has ceased to matter in general.

## Why a small model can match the larger one

Parameter count describes capacity, not how effectively the trained network uses
the available data. Both networks receive seven-class, grayscale 48x48 inputs;
more parameters cannot restore image detail that is absent from those inputs.

M2's last epoch has 90.67% training accuracy and 67.61% validation accuracy, despite
training augmentation and dropout. Its validation accuracy peaks at epoch 21 and
early stopping occurs at 33. Validation loss is lowest at epoch 11, then rises.
That is consistent with a substantial generalization gap, rather than a shortage
of epochs or capacity. M1's curves show a much smaller gap.

Transfer learning also depends on the input pipeline. Current M2 changes the
ImageNet model to a one-channel first convolution, stride 1, no initial max pool,
48x48 inputs and normalization with mean/std 0.5. These are valid adaptation choices,
but they differ from the original pretrained model's RGB input, stem, resolution
and normalization. Their effect on FER performance has not been isolated here.

## A measured, validation-only combination

The two models make different errors on the same 2,871 validation images:

- Both correct: 1,649.
- Only M1 correct: 265.
- Only M2 correct: 308.
- Both incorrect: 649.

This motivates checking combined predictions. One predefined rule was evaluated
directly from the saved validation_predictions.csv files:

```python
if cnn_confidence >= resnet_confidence:
    prediction = cnn_prediction
else:
    prediction = resnet_prediction
```

When the predicted labels agree, the rule naturally returns that shared label.
There was no threshold or weight search. It is confidence arbitration, not an
average of the full class-probability vectors (those vectors were not saved).

| Validation predictor | Correct / 2,871 | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| M1 alone | 1,914 | 66.6667% | 0.6397 |
| M2 alone | 1,957 | 68.1644% | 0.6525 |
| Higher-confidence arbitration | 1,978 | 68.8959% | 0.6601 |

The rule adds 21 correct predictions relative to M2 on this validation split.
This is exploratory validation evidence, not an independently established accuracy
gain. Softmax scores are not necessarily calibrated or directly comparable across
models. No combined test score was computed during this review. A proper soft
ensemble needs the full probability vectors or another inference pass over the
original validation images; it cannot be reconstructed from top-label confidence
alone.

The validation predictions and exact diagnostic metrics are saved alongside this
review. Before deployment, fix the rule using validation, evaluate it on held-out
data, and benchmark the combined application on the target laptop. It adds a CNN
forward pass to each ResNet prediction; existing single-model FPS numbers cannot
be reused as ensemble FPS. Keep M1/M2 individual results for the required lab table.

## Recommended next experiments, in order

1. **Investigate the combination before retraining.** The saved models already
   have complementary errors. Validate confidence arbitration and a simple equal
   soft-probability average, choosing only on validation. There is no measured result
   yet for probability averaging, flip averaging or an ensemble on test data.
2. **For a single-model M2 improvement, test a more faithful ImageNet input setup.**
   Keep the original three-channel 7x7 stride-2 stem and max pool, convert grayscale
   faces to repeated RGB, and use the pretrained normalization. Start with 224x224
   to match the documented pretrained evaluation scale; smaller 112/160 variants
   are a later speed/accuracy tradeoff to measure. Upsampling adds no information;
   the hypothesis is to better match the scales and feature responses of pretraining.
   This requires matching checkpoint preprocessing and app changes if adopted.
3. **Then test staged fine-tuning as a separate factor.** Briefly train the new
   classifier while freezing the backbone, then unfreeze and fine-tune with a small
   backbone learning rate. Frozen BatchNorm statistics need attention during the
   head-only phase. Treat this as a hypothesis, not an automatic improvement.

Keep M1's architecture and current successful gentle augmentation as the reference.
More augmentation alone has not consistently improved M2; do not spend another
long run stacking transformations without a specific hypothesis. Keep archived
first-run M2 weights available too, because the latest weights are not the strongest
measured M2 checkpoint on the supplied FER test set.

All new choices should use validation. The FER test set has already been inspected
across several iterations, so repeated tuning against it would make its score an
optimistic estimate of performance on genuinely unseen classroom faces. Retain a
small separate phone-camera validation set for lighting, pose and crop robustness;
it does not replace the lab's FER test evaluation.

## References

- [Torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
- [Yosinski et al.: How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)

These sources support the transfer/preprocessing considerations. The specific
diagnosis and proposed experiments above are inferences from this project's runs.
