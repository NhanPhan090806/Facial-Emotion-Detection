# FER-2013 facial emotion recognition

PyTorch implementation of the supplied lab PDF, with the later supplied architecture
image taking precedence for M1. The project trains two classifiers and runs emotion
recognition on phone IP cameras, webcams, image files and video files.

**Implementation status:** the downloaded first run measured M1 accuracy 65.99%
and M2 accuracy 67.69%, meeting the lab's FER accuracy targets. The new accuracy-first
comparison is implemented, but its FER gains still need measurement. Large weights
and run outputs are excluded from Git by default. The >=10 FPS realtime requirement
still needs verification on the demonstration laptop.

## Structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── notebooks/
│   └── train.ipynb           # Local training + final evaluation of M1 and M2
├── train_kaggle.ipynb        # Clone your Git repo, import helpers, train on Kaggle
├── src/
│   ├── model.py             # M1 and adapted ResNet-18
│   ├── detector.py          # Haar face detector bundled with OpenCV
│   ├── predict.py           # Shared preprocessing, checkpoint loading, prediction
│   ├── app.py               # Required realtime/image/video command line
│   ├── data.py              # Dataset, augmentation, stratified split, class weights
│   ├── train.py             # Explicit PyTorch training loop
│   ├── experiments.py       # Compare candidates on validation accuracy
│   ├── evaluate.py          # Metrics, plots, errors and CPU model timing
│   └── report.py            # Generate a six-page PDF from actual results
├── weights/                 # Put cnn_best.pt and transfer_best.pt here after training
├── tests/test_project.py    # Offline integration and regression tests
└── report.pdf               # Generated after training; not fabricated in advance
```

The original PDF is the specification; the Kaggle clone/import workflow follows
the example notebook supplied at the start of the project.
`data/`, `outputs/`, `runs/`, checkpoints, generated PDFs and development files in `tmp/`
are ignored by Git. Include the final report and weights in your submission ZIP,
or add real download links for separately hosted checkpoints.

## Installation

Use your existing environment in PowerShell:

```powershell
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -m pip install -r requirements.txt
```

Choose that same Python environment as the notebook kernel in VS Code or Jupyter.
The implementation was checked with Python 3.12, PyTorch 2.5.1 and torchvision
0.20.1. Install a matching torch/torchvision pair appropriate to your machine;
see [PyTorch's installation selector](https://pytorch.org/get-started/locally/).
The requirements use lower bounds and are not an exact environment lock file.

On a different machine, `python -m pip install -r requirements.txt` works once
the correct virtual environment is active. A GUI-capable OpenCV install is needed
for the local display; `--headless` avoids creating a window.

## Data

Download the primary [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
and extract it to:

```text
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...the same seven class folders...
```

You may use a different directory by editing `DATA_ROOT` in the notebook. The
loader also discovers this layout inside a parent directory, but raises an error
if more than one candidate exists. This project supports the linked **image-folder
release**, not the original `fer2013.csv` format.

Validation is a stratified 10% of the original `train/` folder, with seed 42.
The existing `test/` folder is kept for final evaluation. This is a split of the
linked Kaggle release; it is not a reconstruction of the original competition's
PublicTest/PrivateTest split. Both models use the same split, and each run exports
`split_manifest.json`. Different datasets or split protocols are not directly comparable.

The label order is alphabetical:

```text
0 angry, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sad, 6 surprise
```

This differs from the ordering in some FER CSV versions. Checkpoint metadata
stores the mapping, and inference rejects incompatible class/preprocessing metadata.

Both models use grayscale 48x48 images, tensor values in [0,1], then normalization
with mean=0.5 and std=0.5. Training adds a horizontal flip and mild affine rotation,
translation and scaling. Validation/test/inference use deterministic preprocessing.
Class imbalance is handled using square-root inverse training class frequencies
as loss weights. No weights are calculated from validation/test labels. No extra
FER/RAF/AffectNet training data is used; M2 uses ImageNet pretraining.

## The two models

### M1: supplied four-block CNN, trained from scratch

| Stage | Layers | Output (C x H x W) |
| --- | --- | --- |
| Input | Grayscale | 1 x 48 x 48 |
| Block 1 | Conv3x3(32), BN, ReLU, MaxPool2, Dropout0.10 | 32 x 24 x 24 |
| Block 2 | Conv3x3(64), BN, ReLU, MaxPool2, Dropout0.15 | 64 x 12 x 12 |
| Block 3 | Conv3x3(128), BN, ReLU, MaxPool2, Dropout0.20 | 128 x 6 x 6 |
| Block 4 | Conv3x3(256), BN, ReLU, MaxPool2, Dropout0.25 | 256 x 3 x 3 |
| Flatten | 256 x 3 x 3 | 2304 |
| Dense | Linear256, BN, ReLU, Dropout0.50 | 256 |
| Output | Linear7 | 7 logits |

All convolutions have stride 1 and padding 1, so pooling halves the spatial size.
Convolution biases are disabled because BatchNorm already provides an offset.
The network has **980,711 parameters**. It uses ordinary `nn.Dropout`, matching
the supplied image, rather than substituting spatial dropout.

The diagram's final softmax is applied during prediction. Training returns raw
logits because PyTorch `CrossEntropyLoss` already computes log-softmax internally.

### M2: fine-tuned ResNet-18

Start from [torchvision's ImageNet ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
and make these changes:

- Replace the RGB first convolution with a one-channel 7x7 convolution. Sum its
  pretrained RGB filters to initialize it. For three identical normalized grayscale
  channels, summing filters preserves the response; taking their mean would scale it down.
- Change the initial convolution stride from 2 to 1 and remove the initial max
  pool. Small 48x48 faces retain more spatial detail in the residual stages.
- Replace the classification head with Dropout0.30 and Linear(512,7).
- Fine-tune all layers with a smaller learning rate in the backbone.

The adapted model has **11,173,831 parameters**. Using 48x48 grayscale and FER
normalization is a deliberate fine-tuning choice; these are not the stock
224x224 RGB ImageNet transforms. Initial weights download on first training use.
Loading a saved FER checkpoint for inference requires no download.

## Train locally: accuracy-first comparison

Open `notebooks/train.ipynb`, check DATA_ROOT, and run all cells. The exact supplied
M1 architecture and adapted ResNet-18 architecture are retained. The new default
workflow compares candidates and exports the **highest-validation-accuracy** model;
macro F1 breaks exact ties. It does not assume the new recipe improves accuracy.

The first supplied Kaggle run measured M1 accuracy 65.99% / macro F1 0.6337, and
M2 accuracy 67.69% / macro F1 0.6577. See [the first-run review](analysis/run1_review.md)
for the diagnosis behind the candidate choices. M2 showed a substantial training
versus validation gap; the changes target generalization without aggressively
rebalancing the classes.

| Candidate | Models | Training augmentation | Other changes |
| --- | --- | --- | --- |
| baseline | M1, M2 | Original flip + mild affine | Original optimizer settings |
| lighting | M1, M2 | Baseline + brightness/contrast jitter, 0.15 each | None |
| regularized | M2 | Lighting + erasing p=0.10, area 2-6% | Weight decay 5e-4, backbone LR 5e-5 |

M1 candidates use 80 epochs maximum, head LR 1e-3, weight decay 1e-4 and patience
18. M2 candidates each use the first run's **20-epoch** maximum and patience 12;
baseline/lighting use head LR 1e-3, backbone LR 1e-4 and weight decay 1e-4.
The regularized candidate combines modest changes for a practical candidate search,
so a gain cannot be attributed to a single change without a later ablation.

All candidates retain square-root inverse-frequency class weights, label smoothing
0.05, AdamW, cosine LR decay and gradient clipping at norm 5. No oversampling is
stacked with the weighted loss. The same split seed and validation images are used
throughout. CUDA mixed precision is enabled for training; checkpoint selection and
validation predictions use float32, matching deployment.

The default batch size is 128 to match the first Kaggle run. Both notebooks use
NUM_WORKERS=0 to avoid the notebook worker-cleanup issue; this does not disable GPU
training. The progress bar uses plain tqdm instead of tqdm.auto. Training-only
augmentation never changes validation/test/inference preprocessing.

RUN_NAME defaults to `fer2013_accuracy_v2`. Local results are stored in
`runs/fer2013_accuracy_v2/`; this preserves the downloaded first-run folder and
keeps candidate weights separate from active application weights. Five fits take
longer than the original two-model run. To reuse the downloaded original baselines,
set this in the local notebook:

```python
BASELINE_RESULTS_ROOT = PROJECT_ROOT / "kaggle_outputs" / "run1_1025" / "outputs"
```

This requires `cnn/best.pt`, `transfer/best.pt`, their configs and their split
manifests. The helper verifies that each reused baseline's training/validation
membership and training config match before comparison. Archived baselines are
re-evaluated on validation in float32; old test results do not choose the winner.
If no archive is supplied, baseline candidates are trained from scratch too.

Completed candidates with matching configs, loader settings and split manifests
are reused on rerun. An interrupted fit cannot resume its optimizer; preserve its
files and choose a new RUN_NAME. Changed recipes/settings also require a new name.
The lower-level `fit()` still starts fresh and overwrites its own output directory;
`run_accuracy_experiments()` adds the completion checks around it.

The selected model can remain the baseline. Winning on this validation split does
not guarantee a gain on FER test data or the classroom camera. Inspect the final
validation diagnostics and then run the notebook's final test cell once for the
selected models. Keep the original checkpoints available for comparison.

## Train on Kaggle

1. Upload this project to your Git repository, including `src/` and requirements.
2. Upload `train_kaggle.ipynb` as a Kaggle notebook.
3. Replace `REPO_URL` in its first code cell with the real repository HTTPS URL.
   `REPO_REF` optionally selects an existing branch/tag on the first clone.
4. Start a fresh Kaggle session after uploading the updated helpers. Enable Internet and a GPU accelerator. Add the linked FER-2013 dataset using
   **Add Input**. The notebook finds the folder below `/kaggle/input`; if none is
   attached it optionally downloads through [kagglehub](https://github.com/Kaggle/kagglehub).
   If more than one candidate exists, set `DATA_ROOT_OVERRIDE` explicitly.
5. Run all cells. Both models train through the cloned `src` helpers. The
   notebook records the Git commit and writes results under `/kaggle/working`.
6. Download `fer2013_accuracy_v2.zip`. After reviewing the measured results, copy its `weights/cnn_best.pt` and
   `weights/transfer_best.pt` into this project's `weights/` directory.

Kaggle defaults to batch size 128 and zero data-loader workers. Reduce batch size
if memory is insufficient. A second GPU is not required. On rerun, the notebook
reuses the clone and verifies its origin; it does not silently pull or reset files.
Start a fresh session or explicitly update the clone and restart its kernel to use a newer commit. The notebook checks that the new experiments helper exists.

The PDF requires an executed `notebooks/train.ipynb` with outputs. Downloading the
executed Kaggle notebook does not execute the separate local notebook. Run the
local notebook too, or upload that notebook to Kaggle with its setup paths adapted,
and retain the executed version at the required submission path.

## Evaluation artifacts

Under RUN_ROOT (`runs/fer2013_accuracy_v2` locally; `/kaggle/working/fer2013_accuracy_v2` on Kaggle):

```text
outputs/
├── class_distribution.png
├── split_manifest.json
├── comparison.csv / comparison.json
├── experiments/
│   ├── cnn/{baseline,lighting}/
│   └── transfer/{baseline,lighting,regularized}/
├── cnn/
│   ├── best.pt / last.pt
│   ├── config.json / data_config.json / training_summary.json
│   ├── selection.json / validation_metrics.json
│   ├── validation_predictions.csv / validation_classification_report.json
│   ├── validation_confusion_matrix.csv
│   ├── class_counts.json / split_manifest.json
│   ├── history.csv / training_curves.png
│   ├── metrics.json / classification_report.json
│   ├── confusion_matrix.csv / confusion_matrix.png
│   ├── predictions.csv / misclassified.png
│   └── error_analysis.md
└── transfer/
    └── ...same artifacts...
```

M3 contains test accuracy, macro F1, parameter count and CPU inference FPS. The CPU
benchmark uses batch size one, 48x48 grayscale, float32 and four threads, with
warm-up passes. It times **model forward only**, excluding image preprocessing,
face detection and camera latency. It is not proof that the full application runs
at 10 FPS. Timing hardware, scope and thread count are saved in the metrics.

M4 exports ten errors per model when available, selecting different true classes
before filling remaining slots. Score-based comments describe each error, but
their suggested causes need your own visual inspection. Edit the error analysis
and report source with concrete observations before submission. M5 plots training
and validation loss/accuracy across epochs.

## Run the application

From the project root, with the environment active:

```powershell
python src/app.py --source http://<ip>:8080/video --model cnn
python src/app.py --source test.jpg --model transfer
python src/app.py --source 0 --model cnn
python src/app.py --source demo.mp4 --model transfer --output outputs/annotated.mp4
python src/app.py --source test.jpg --model cnn --headless --output outputs/test_result.jpg
```

For your environment, you can replace `python` with
`& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe'` in PowerShell.

Edit `config.yaml` to set the source, checkpoint paths, CPU threads, confidence
threshold, detector settings and reconnect behavior. Config checkpoint paths
are relative to the config file; command-line file paths are relative to your
working directory. `--weights`, `--device`, `--confidence`, `--cpu-threads`, and
`--max-frames` can override individual settings.

Phone setup:

1. Connect the laptop and phone to the same Wi-Fi network or phone hotspot.
2. Start an IP camera app on the phone and copy its actual stream URL.
3. Confirm the URL works on the laptop. Supply it using `--source` or config.
4. Start with a 640x480 stream for lower latency. Measure both models on CPU.

The app detects multiple faces and batches their classification. It draws boxes,
emotion labels, softmax confidence and smoothed capture + processing FPS. Low
scores are labelled `uncertain`; this threshold is not a calibrated error bound.
Press **q** to exit, or **s** to save `outputs/demo_screenshot.jpg`. A static image
is always saved (default `outputs/prediction.jpg`) and shown unless `--headless`.

Phone disconnects and failed opens trigger bounded retries, then an explanatory
error; video files stop at EOF. Network timeouts use OpenCV's FFmpeg backend.
Return codes: 0 success/user exit, 1 invalid input/config/model, 2 exhausted
camera retries. Haar detection is fast and offline, but can miss profile or
occluded faces. Faces need enough resolution and reasonable frontal lighting.

The application's FPS includes frame acquisition, face detection, preprocessing
and model inference; display and video output writing are excluded. Network
buffering can still add latency. More faces and larger frames increase work.

## Finish the report and submission

The notebooks generate a six-page `RUN_ROOT/report.pdf` using real experiment outputs, including the selected augmentation profile.
Without a real demo screenshot, page six is explicitly marked pending. Add
your own image-specific error discussion in `src/report.py`, then regenerate:

```powershell
python -m src.report --results runs/fer2013_accuracy_v2/outputs --output runs/fer2013_accuracy_v2/report.pdf --screenshot outputs/demo_screenshot.jpg --author "Your name / student ID"
```

For a Kaggle run, extract the archive into `runs/fer2013_accuracy_v2/` first. Include
data/preprocessing, both architectures, hyperparameters, actual M3 results,
confusion matrices, ten errors per model and a phone-camera screenshot. Verify
both required commands and the >=10 CPU FPS requirement on the demonstration
laptop. Keep the executed notebook, report and trained weights in the submission.

## Offline checks

```powershell
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -m unittest discover -s tests -v
```

These tests validate M1 shapes, the pretrained RGB-to-grayscale conversion,
split isolation, checkpoint/preprocessing consistency, both notebook syntaxes,
and an end-to-end notebook smoke run on generated images. The smoke run trains
all five candidates for one epoch **without pretrained downloads**, produces evaluation
artifacts/report, and checks image, video, multiple-face and disconnect handling.
The checks also verify accuracy-first selection, split-safe baseline reuse, caching, and training-only augmentation. Synthetic-image test scores are not FER-2013 results. When CUDA is available,
an additional test checks mixed-precision training for both models on the same
generated images. Copy the reviewed final run's PDF to the root `report.pdf` for
the submission layout required by the specification.
