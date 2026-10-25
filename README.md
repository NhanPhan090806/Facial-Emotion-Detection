# FER-2013 facial emotion recognition

PyTorch implementation of the supplied lab PDF, with the later supplied architecture
image taking precedence for M1. The project trains two classifiers and runs emotion
recognition on phone IP cameras, webcams, image files and video files.

**Implementation status:** both training notebooks and the application are provided.
The repository does not contain trained FER checkpoints or measured FER scores yet.
The PDF's targets are M1 accuracy >= 60%, M2 accuracy >= 65%, and >= 10 FPS for the
CPU realtime application. These are measurements to verify after training, not promises.

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
│   ├── evaluate.py          # Metrics, plots, errors and CPU model timing
│   └── report.py            # Generate a six-page PDF from actual results
├── weights/                 # Put cnn_best.pt and transfer_best.pt here after training
├── tests/test_project.py    # Offline integration and regression tests
└── report.pdf               # Generated after training; not fabricated in advance
```

The original PDF and `train_market1501_kaggle.ipynb` are retained as references.
`data/`, `outputs/`, checkpoints, generated PDFs and development files in `tmp/`
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

## Train locally

Open `notebooks/train.ipynb`, check the data/output paths, and run all cells.
The default Windows notebook loader uses `num_workers=0` to avoid multiprocessing
problems. A GPU is strongly recommended; the code also supports CPU execution.

| Setting | M1 | M2 |
| --- | --- | --- |
| Epoch limit | 80 | 40 |
| AdamW learning rate | 0.001 | head 0.001, backbone 0.0001 |
| Weight decay | 0.0001 | 0.0001 |
| Early-stopping patience | 18 | 12 |
| Label smoothing | 0.05 | 0.05 |
| Batch size (local default) | 64 | 64 |
| Scheduler | cosine | cosine |
| Checkpoint selection | validation accuracy, then macro F1 | same |

Gradient clipping limits norm to 5. CUDA mixed precision is enabled when available.
`train_loss` and `val_loss` are plain cross entropy, making the plots interpretable;
the weighted/smoothed loss used for optimization is also saved as `train_objective_loss`.
Training measurements include augmentation and dropout, while validation does not.
The test set is not used for early stopping, class weights, or hyperparameter selection.

Every call to `fit` starts a new experiment and replaces files in its output
directory. Change `OUTPUT_ROOT` to preserve a previous run. `best.pt` and `last.pt`
are inference checkpoints, not complete optimizer-resume snapshots.

## Train on Kaggle

1. Upload this project to your Git repository, including `src/` and requirements.
2. Upload `train_kaggle.ipynb` as a Kaggle notebook.
3. Replace `REPO_URL` in its first code cell with the real repository HTTPS URL.
   `REPO_REF` optionally selects an existing branch/tag on the first clone.
4. Enable Internet and a GPU accelerator. Add the linked FER-2013 dataset using
   **Add Input**. The notebook finds the folder below `/kaggle/input`; if none is
   attached it optionally downloads through [kagglehub](https://github.com/Kaggle/kagglehub).
   If more than one candidate exists, set `DATA_ROOT_OVERRIDE` explicitly.
5. Run all cells. Both models train through the cloned `src` helpers. The
   notebook records the Git commit and writes results under `/kaggle/working`.
6. Download `fer2013_results.zip`. Copy its `weights/cnn_best.pt` and
   `weights/transfer_best.pt` into this project's `weights/` directory.

Kaggle defaults to batch size 128 and two data-loader workers. Reduce batch size
if memory is insufficient. A second GPU is not required. On rerun, the notebook
reuses the clone and verifies its origin; it does not silently pull or reset files.
Start a fresh session or explicitly update the clone to use a newer commit.

The PDF requires an executed `notebooks/train.ipynb` with outputs. Downloading the
executed Kaggle notebook does not execute the separate local notebook. Run the
local notebook too, or upload that notebook to Kaggle with its setup paths adapted,
and retain the executed version at the required submission path.

## Evaluation artifacts

```text
outputs/
├── class_distribution.png
├── split_manifest.json
├── comparison.csv / comparison.json
├── cnn/
│   ├── best.pt / last.pt
│   ├── config.json / training_summary.json
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

The notebooks generate a six-page `report.pdf` using real experiment outputs.
Without a real demo screenshot, page six is explicitly marked pending. Add
your own image-specific error discussion in `src/report.py`, then regenerate:

```powershell
python -m src.report --results outputs --output report.pdf --screenshot outputs/demo_screenshot.jpg --author "Your name / student ID"
```

For a Kaggle run, extract its `outputs/` folder into this project first. Include
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
Synthetic-image test scores are not FER-2013 results. When CUDA is available,
an additional test checks mixed-precision training for both models on the same
generated images.
