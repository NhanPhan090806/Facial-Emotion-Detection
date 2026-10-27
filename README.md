# FER-2013 facial emotion recognition

PyTorch implementation of the supplied lab PDF, with the later supplied architecture
image taking precedence for M1. The project trains two classifiers and runs emotion
recognition on phone IP cameras, webcams, image files and video files.

**Implementation status:** the earlier CNN/ResNet-18 run reached 65.99%/67.69%
test accuracy. M2 has now been replaced by AffectNet-pretrained DDAMFN; its FER
accuracy is not measured yet. Both notebooks train each model once and export
validation/test results. Large weights and outputs are excluded from Git.
The >=10 FPS realtime requirement needs verification on the demonstration laptop.

## Structure

```text
.
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ config.yaml
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ train.ipynb           # Local training + final evaluation of M1 and M2
â”œâ”€â”€ train_kaggle.ipynb        # Clone your Git repo, import helpers, train on Kaggle
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ model.py             # M1 and model builders (DDAMFN default for M2)
│   ├── ddamfn.py            # Architecture, pinned download, label mapping
â”‚   â”œâ”€â”€ detector.py          # Haar face detector bundled with OpenCV
â”‚   â”œâ”€â”€ predict.py           # Shared preprocessing, checkpoint loading, prediction
â”‚   â”œâ”€â”€ app.py               # Required realtime/image/video command line
â”‚   â”œâ”€â”€ data.py              # Dataset, augmentation, stratified split, class weights
â”‚   â”œâ”€â”€ train.py             # Explicit PyTorch training loop
â”‚   â”œâ”€â”€ experiments.py       # Validation export + optional candidate-comparison API
â”‚   â”œâ”€â”€ evaluate.py          # Metrics, plots, errors and CPU model timing
â”‚   â””â”€â”€ report.py            # Generate a six-page PDF from actual results
â”œâ”€â”€ weights/                 # Put cnn_best.pt and transfer_best.pt here after training
â”œâ”€â”€ tests/test_project.py    # Offline integration and regression tests
â””â”€â”€ report.pdf               # Generated after training; not fabricated in advance
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
â”œâ”€â”€ train/
â”‚   â”œâ”€â”€ angry/
â”‚   â”œâ”€â”€ disgust/
â”‚   â”œâ”€â”€ fear/
â”‚   â”œâ”€â”€ happy/
â”‚   â”œâ”€â”€ neutral/
â”‚   â”œâ”€â”€ sad/
â”‚   â””â”€â”€ surprise/
â””â”€â”€ test/
    â””â”€â”€ ...the same seven class folders...
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

FER-2013 contains grayscale 48x48 images. M1 uses those dimensions and
normalizes with mean/std 0.5. M2 repeats grayscale into three channels, resizes
to 112x112, and uses ImageNet mean/std to match the official DDAMFN++ checkpoint.
Upsampling adds no new detail. The camera app also converts crops to grayscale,
then selects the correct transform from checkpoint metadata.

Class imbalance uses square-root inverse **training** class frequencies as loss
weights. No validation/test labels enter those weights. M2 now uses external
**AffectNet-7 expression pretraining**, replacing the previous ImageNet ResNet-18;
that difference must be disclosed when comparing architectures. FER fine-tuning
uses only the original FER training split.

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

### M2: DDAMFN with DDAMFN++ AffectNet-7 initialization

The [official DDAMFN repository](https://github.com/SainingZhang/DDAMFN) provides
a MixedFeatureNet face backbone and two coordinate attention heads. The heads
learn vertical/horizontal attention maps and their elementwise maximum weights
the face features before a seven-class linear classifier. The model has
**4,187,421 parameters**, compared with 11,173,831 in our earlier adapted ResNet.
Smaller parameter count alone does not establish faster CPU inference; the
notebooks benchmark the actual 112x112 model.

`src/ddamfn.py` implements the network in ordinary PyTorch and strictly loads the
official `DDAMFN++/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth` initialization.
The source commit is `cebfe5c5969b7d1fd64ecd775e318d3834ca76bb`; its SHA-256 is
`edf731a458162eef6b2be12c3572bd10cbe92ce8c009da4331184a2943d7ed14`.
The download is cached under PyTorch's hub checkpoints directory and verified
before use. There is no silent fallback to random weights on download failure.
Only tensor state dictionaries are loaded with `weights_only=True`.

AffectNet's class order is neutral/happy/sad/surprise/fear/disgust/angry.
Classifier rows are explicitly reordered `[6,5,4,1,0,2,3]` to match our labels.
We retain this pretrained expression head, then fine-tune all layers with a
smaller backbone LR. We use weighted cross entropy rather than upstream SAM
or reciprocal attention-diversity loss; this is a transfer-learning adaptation,
not a paper reproduction. Retained unused upstream layers preserve strict
checkpoint compatibility and are included in the parameter count.

Pretrained source URL, commit, hash and class mapping are saved with the run and
checkpoint. Saved FER checkpoints load without downloading anything. Existing
version-1 ResNet checkpoints remain supported for inference, selected by their
metadata; new `transfer` training defaults to DDAMFN.

## Train locally: one run per model

Open `notebooks/train.ipynb`, check DATA_ROOT, and run all cells. It performs
exactly **two fits**, without archived outputs, candidate search, BASELINE_RESULTS_ROOT
or REFERENCE_RESULTS_ROOTS. Both notebooks use the same fixed configurations:

| Setting | M1 | M2 |
| --- | --- | --- |
| Architecture | Exact supplied CNN | DDAMFN / DDAMFN++ AffectNet-7 |
| Epoch limit | 80 | 40 |
| Augmentation | gentle_combined | fer_moderate |
| Head LR | 1e-3 | 1e-4 |
| Backbone LR | n/a | 1e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Early-stopping patience | 18 | 10 |
| Label smoothing | 0.05 | 0.02 |

M1 combines horizontal flips, reflect-padded 2-pixel crops, rotation up to 8 degrees,
scaling 0.95-1.05, and brightness/contrast jitter 0.08 with probability 0.5.
Crop shifts replace affine translation. No extra dropout or erasing is added.
M2 combines horizontal flips, rotation up to 10 degrees, shifts up to 5%,
scale 0.95-1.05, brightness/contrast jitter 0.15 with probability 0.7, and small
erasures with probability 0.15 covering 1-4% of the image. Augmentation starts
at 48x48, before resizing. This is moderate augmentation: stronger crops or
occlusions can erase the eyes/mouth cues needed for the label. M1's successful
gentle recipe is retained. These settings still need a full FER validation run.
M2's backbone LR ramps up over its first two epochs, then follows cosine decay.

Both use AdamW, cosine LR decay, moderate square-root inverse-frequency loss
weights and gradient clipping at norm 5. Oversampling is not added. GPU training
uses AMP; validation and deployment use float32. Histories record successful
optimizer updates and any updates skipped by AMP while adjusting its scale. Validation accuracy selects the
best epoch; macro F1 breaks exact ties. Train/validation membership is fixed with
seed 42, and test images are evaluated only after training.

Batch size defaults to 128 for M1 and is capped at 64 for M2; NUM_WORKERS is zero. The latter avoids notebook
worker cleanup errors and does not disable GPU training. Logs print one summary
per epoch. Validation/test/inference preprocessing has no random augmentation.

Local outputs are under `runs/fer2013_ddamfn/`: both model histories, best/last
checkpoints, validation diagnostics, final test metrics, copied deployment weights
and a report draft. Copy the reviewed weights into project-level `weights/` for
`config.yaml` to use them. A training cell starts fresh and overwrites that model's
output directory; choose a new RUN_NAME to preserve a separate run. Checkpoints
do not contain optimizer-resume state.

The earlier comparison workflow remains an optional API in `src/experiments.py`.
It is not called by these notebooks. See [run2_review.md](analysis/run2_review.md)
for the historical diagnosis and comparison settings.

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
6. Download `fer2013_ddamfn.zip`. After reviewing the measured results, copy its `weights/cnn_best.pt` and
   `weights/transfer_best.pt` into this project's `weights/` directory.

Kaggle uses batch size 128 for M1, up to 64 for M2, and zero data-loader workers. Reduce batch size
if memory is insufficient. A second GPU is not required. On rerun, the notebook
reuses the clone and verifies its origin; it does not silently pull or reset files.
Start a fresh session or explicitly update the clone and restart its kernel to use a newer commit. The notebook checks that the new DDAMFN helper exists. The pinned pretrained checkpoint is downloaded and verified before M1 training starts.

The PDF requires an executed `notebooks/train.ipynb` with outputs. Downloading the
executed Kaggle notebook does not execute the separate local notebook. Run the
local notebook too, or upload that notebook to Kaggle with its setup paths adapted,
and retain the executed version at the required submission path.

## Evaluation artifacts

Under RUN_ROOT (`runs/fer2013_ddamfn` locally; `/kaggle/working/fer2013_ddamfn` on Kaggle):

```text
outputs/
â”œâ”€â”€ class_distribution.png
â”œâ”€â”€ split_manifest.json
â”œâ”€â”€ comparison.csv / comparison.json
â”œâ”€â”€ cnn/
â”‚   â”œâ”€â”€ best.pt / last.pt
â”‚   â”œâ”€â”€ config.json / data_config.json / training_summary.json
â”‚   â”œâ”€â”€ validation_metrics.json
â”‚   â”œâ”€â”€ validation_predictions.csv / validation_classification_report.json
â”‚   â”œâ”€â”€ validation_confusion_matrix.csv
â”‚   â”œâ”€â”€ class_counts.json / split_manifest.json
â”‚   â”œâ”€â”€ history.csv / training_curves.png
â”‚   â”œâ”€â”€ metrics.json / classification_report.json
â”‚   â”œâ”€â”€ confusion_matrix.csv / confusion_matrix.png
â”‚   â”œâ”€â”€ predictions.csv / misclassified.png
â”‚   â””â”€â”€ error_analysis.md
â””â”€â”€ transfer/
    â””â”€â”€ ...same artifacts...
```

M3 contains test accuracy, macro F1, parameter count and CPU inference FPS. The CPU
benchmark uses batch size one, each model's native input shape (1x48x48 or 3x112x112), float32 and four threads, with
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
python -m src.report --results runs/fer2013_ddamfn/outputs --output runs/fer2013_ddamfn/report.pdf --screenshot outputs/demo_screenshot.jpg --author "Your name / student ID"
```

For a Kaggle run, extract the archive into `runs/fer2013_ddamfn/` first. Include
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
both models for one epoch **without pretrained downloads**, produces evaluation
artifacts/report, and checks image, video, multiple-face and disconnect handling.
Additional optional-helper checks verify accuracy-first selection, split-safe archive reuse, caching, and training-only augmentation. Synthetic-image test scores are not FER-2013 results. When CUDA is available,
an additional test checks mixed-precision training for both models on the same
generated images. Copy the reviewed final run's PDF to the root `report.pdf` for
the submission layout required by the specification.
