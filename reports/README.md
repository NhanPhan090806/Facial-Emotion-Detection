# Selected run reports

- `report.pdf`: the original run-4 six-page training report.
- `training.log`: the original Kaggle log.
- `history/`: earlier reports and historical notes, without obsolete checkpoint binaries.
- `results/comparison.json` and `comparison.csv`: final M1/M2 comparison.
- `results/class_distribution.png`: train/validation/test class distribution.
- `results/split_manifest.json`: the common recorded file split.
- `results/cnn/` and `results/transfer/`: training curves, confusion matrices,
  misclassified examples, per-class metrics, histories, validation diagnostics,
  test predictions and pretraining provenance.

Selected model binaries live in models/, not in the report results.
The original archive's duplicate best/last checkpoints have been removed.

The report still needs your phone screenshot, name/student ID and image-specific
error discussion before submission. The current application uses YuNet; the
report generator describes this updated detector when you regenerate the PDF.
Press s in the camera window to save reports/demo_screenshot.jpg, then run:

```cmd
python -m src.report --results reports/results --output reports/report.pdf --screenshot reports/demo_screenshot.jpg --author "Your name / student ID"
```

CPU FPS in the saved metrics measures classifier forward passes only on Kaggle.
Record the application's actual capture-and-processing FPS on your own computer
for the real-time demonstration requirement.
