# Facial emotion recognition on FER-2013

Readable PyTorch training and real-time phone-camera inference with two models:
M1 is the supplied four-block CNN trained from scratch; M2 is DDAMFN initialized
from the official DDAMFN++ AffectNet-7 checkpoint.

The selected checkpoints are from run 4. They are ready to use through
`config.yaml`; no training or checkpoint-path override is required for inference.

| Selected model | Test accuracy | Test macro F1 | Parameters | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| M1: custom CNN | 66.41% | 0.6415 | 980,711 | 67 |
| M2: DDAMFN | 72.05% | 0.7059 | 4,187,421 | 32 |

Both exceed the assignment's accuracy targets of 60% and 65%. Scores cover all
7,178 FER test images. The complete measurements are in
[reports/results/comparison.json](reports/results/comparison.json).
M2 uses external expression-labeled pretraining; that matters when comparing
it with a model trained from scratch.

## Project layout

```text
.
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- assets/                  YuNet detector, license and source information
|-- docs/assignment.pdf      Original assignment specification
|-- models/
|   |-- cnn_best.pt          Selected M1 checkpoint
|   |-- transfer_best.pt     Selected M2 checkpoint
|   |-- manifest.json       Source run, epochs, scores and SHA-256 checksums
|   `-- README.md
|-- reports/
|   |-- report.pdf           Run-4 report
|   |-- training.log         Original Kaggle training log
|   |-- results/             Plots, metrics, histories and split manifests
|   |-- history/             Earlier reports and notes, without obsolete models
|   `-- README.md
|-- notebooks/train.ipynb    Local training notebook
|-- train_kaggle.ipynb       Kaggle notebook that clones and uses src/ helpers
|-- src/
|   |-- model.py             M1 and model builders
|   |-- ddamfn.py            DDAMFN architecture and verified pretraining
|   |-- data.py              Splits, preprocessing, augmentation, class weights
|   |-- train.py             PyTorch training loop
|   |-- experiments.py       Validation export and optional experiment helpers
|   |-- evaluate.py          Test metrics, plots, error examples, CPU benchmark
|   |-- detector.py          YuNet face detector and optional Haar backend
|   |-- predict.py           Checkpoint loading and emotion inference
|   |-- app.py               Camera, image and video application
|   `-- report.py            Six-page PDF report generator
`-- tests/                   Offline regression and integration checks
```

The selected models, final report and supporting results are retained in the
repository. Raw datasets, fresh training runs, caches and temporary files are
ignored. Older experiment archives and synthetic development outputs are not
needed to run the application.

## Install

Activate your Python environment, then run from the project directory:

```cmd
python -m pip install -r requirements.txt
```

Use a matching PyTorch/torchvision pair for your CPU or CUDA installation.
OpenCV 4.10 or newer is required. The local implementation was verified with
Python 3.12, PyTorch 2.5.1, torchvision 0.20.1 and OpenCV 4.10. A GUI-capable
OpenCV installation is needed to show the camera window.

## Phone-camera inference

Connect the phone and computer to the same Wi-Fi network, start the phone's
IP-camera server, and replace the example URL below with its actual video-stream
URL. Confirm the stream works on the computer first.

M1:

```cmd
python src\app.py --source "http://192.168.1.100:8080/video" --model cnn --device cpu
```

M2:

```cmd
python src\app.py --source "http://192.168.1.100:8080/video" --model transfer --device cpu
```

The app selects `models/cnn_best.pt` or `models/transfer_best.pt` automatically.
Run one command at a time. Press **q** to quit, or **s** to save
`reports/demo_screenshot.jpg`.

YuNet is the default pretrained face detector. Its 233 KB ONNX model is bundled
under `assets/` with its MIT license, so detection works offline. Frames are
letterboxed to 320x320 without stretching; detected boxes are mapped back to the
original frame with a 10% crop margin. Emotion labels use large white text on dark
panels, with confidence below the emotion name and FPS at the bottom-left.

Useful options to append to either command:

| Option | Effect |
| --- | --- |
| `--label-scale 1.3` | Make emotion labels 30% larger |
| `--detector-size 640` | Use more detail for smaller faces, at extra CPU cost |
| `--detector-confidence 0.65` | Lower the face-detection threshold from 0.75 |
| `--confidence 0.5` | Set the emotion-confidence threshold |
| `--cpu-threads 4` | Limit PyTorch and OpenCV CPU threads |
| `--detector haar` | Select the older Haar detector explicitly |
| `--weights models/another_model.pt` | Override the selected emotion checkpoint |

Low-confidence predictions show `UNCERTAIN` and their top guess. Confidence
scores are not calibrated probabilities. The displayed FPS includes capture,
face detection, preprocessing and classification; it excludes display and video
writing. Verify the required live CPU FPS on the demonstration computer. Run 4's
11.75 FPS for M2 measured the classifier alone on Kaggle.

Other inputs:

```cmd
python src\app.py --source 0 --model cnn
python src\app.py --source test.jpg --model transfer
python src\app.py --source demo.mp4 --model transfer --output reports\predictions\annotated.mp4
python src\app.py --source test.jpg --model cnn --headless --output reports\predictions\test.jpg
```

The default static-image output is `reports/predictions/prediction.jpg`.
Camera failures trigger bounded retries. Settings and selected model paths are
in `config.yaml`; relative paths there resolve against the config file.

## Data and augmentation

Download [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and extract it
under `data/fer2013/`, containing `train/` and `test/` with the seven emotion
subfolders. Class order is angry, disgust, fear, happy, neutral, sad, surprise.

Validation is a stratified 10% holdout from the original training folder, using
seed 42. The selected run contains 25,838 training, 2,871 validation and 7,178 test
images. Both models use identical file splits. Test images do not select epochs.
Checkpoint selection uses validation accuracy, with macro F1 breaking exact ties.

| Training augmentation | M1: gentle_combined | M2: fer_moderate |
| --- | --- | --- |
| Horizontal flip | Probability 0.5 | Probability 0.5 |
| Position changes | Reflect-pad 2 pixels and random crop back to 48x48 | Shifts up to 5% |
| Rotation | Up to 8 degrees | Up to 10 degrees |
| Scale | 0.95-1.05 | 0.95-1.05 |
| Brightness/contrast | Up to 8%, probability 0.5 | Up to 15%, probability 0.7 |
| Erasing | None | Probability 0.15, area 1-4% |

Training augmentation is sampled when images are loaded. Validation, test and
live inference have no random augmentation. M1 uses grayscale 48x48 tensors
normalized with mean/std 0.5. M2 repeats grayscale into three channels, resizes to
112x112, and uses means (0.485, 0.456, 0.406) and standard deviations
(0.229, 0.224, 0.225). Resizing adds no new image detail. M2 erasing occurs before
normalization. Class imbalance uses square-root inverse training-class-frequency
loss weights, without oversampling.

## Models and training

M1 has four Conv3x3-BatchNorm-ReLU-MaxPool2-Dropout blocks with channels
32/64/128/256 and dropout 0.10/0.15/0.20/0.25. The head flattens 2304 features,
then uses Linear256-BatchNorm-ReLU-Dropout0.50-Linear7. It returns logits for
cross entropy and applies softmax only during inference.

M2 uses mixed-kernel depthwise face features and two coordinate attention heads.
The [official DDAMFN++ AffectNet-7 initialization](https://github.com/SainingZhang/DDAMFN)
is pinned and checksum-verified in `src/ddamfn.py`. Its seven-class head is
remapped into this project's class order before fine-tuning all layers. We use
weighted cross entropy rather than reproducing upstream SAM or the reciprocal
attention-diversity loss. Inference from a selected checkpoint needs no download.

Each notebook performs exactly two fits, without archived baseline outputs:

| Setting | M1 | M2 |
| --- | --- | --- |
| Epoch limit | 80 | 40 |
| Batch size | 128 | Up to 64 |
| Learning rate | 1e-3 | Head 1e-4, backbone 1e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Label smoothing | 0.05 | 0.02 |
| Patience | 18 | 10 |

Both use AdamW, epoch-based cosine LR decay, gradient clipping and training-only
class weights. M2 ramps its backbone LR over two epochs. GPU training uses AMP;
validation and inference use float32. Logs print one summary per epoch, and
histories record successful and skipped updates. `NUM_WORKERS=0` avoids notebook
worker-cleanup errors while keeping GPU training enabled.

For local training, run `notebooks/train.ipynb` with your active environment as
the kernel and check `DATA_ROOT`. Fresh runs stay under `runs/RUN_NAME/` so the
selected root-level models are not overwritten automatically. Each completed
run exports `models/` with its two selected checkpoints and `reports/` with the
report and results. Duplicate training checkpoints are removed after export.
Fits start fresh; optimizer-resume state is not saved.

For Kaggle, upload `train_kaggle.ipynb`, enable Internet and a GPU, set its
repository URL, and attach the FER dataset. The notebook clones the helpers,
verifies the official pretrained weights before training M1, records the source
commit and creates `fer2013_ddamfn.zip`. After reviewing a new run, copy its
`models/` and `reports/` into the corresponding project directories.

## Report and checks

The run-4 PDF is at [reports/report.pdf](reports/report.pdf). Plots, per-class
reports, prediction CSVs, histories and split manifests are in `reports/results/`.
The PDF is a training-run draft until you add the real phone-camera screenshot,
your name/student ID and your own visual observations about the error examples.

After saving the phone screenshot, regenerate it with:

```cmd
python -m src.report --results reports/results --output reports/report.pdf --screenshot reports/demo_screenshot.jpg --author "Your name / student ID"
```

The original specification is preserved as [docs/assignment.pdf](docs/assignment.pdf).
Keep an executed training notebook for submission. If the submission portal
requires the original PDF's exact layout, assemble a separate submission copy.

Run the offline checks from an activated environment:

```cmd
python -m unittest discover -s tests -v
```

Synthetic test images verify implementation behavior, not FER accuracy. The
selected model hashes and result integrity are checked against the retained
real run. Optional upstream-weight and sample-image tests skip when their
local development fixtures are absent.
