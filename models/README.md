# Selected emotion models

These are the two selected checkpoints from run 4:

| File | Model | Epoch | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: | ---: |
| cnn_best.pt | M1 custom CNN, trained from scratch | 67 | 66.41% | 0.6415 |
| transfer_best.pt | M2 DDAMFN, AffectNet-7 initialization | 32 | 72.05% | 0.7059 |

`manifest.json` records the original SHA-256 checksums and exact metrics.
The files are unchanged from the selected Kaggle checkpoints. Each includes
architecture, class order, preprocessing and training configuration metadata.
Loading them for inference does not download pretrained weights.

`config.yaml` selects these paths automatically. New completed training runs
export their chosen checkpoints under their own models/ directory; review
results before replacing the selected files here. Keep the manifest consistent
with any replacement. The face detector is separately bundled under assets/.
