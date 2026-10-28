"""Build a six-page report from real experiment outputs, without inventing results.

python -m src.report --results reports/results --output reports/report.pdf --screenshot reports/demo_screenshot.jpg
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_report(results_dir, output_path="reports/report.pdf", screenshot=None, author="Student name / ID"):
    results_dir, output_path = Path(results_dir), Path(output_path)
    metrics = {name: read_json(results_dir / name / "metrics.json") for name in ["cnn", "transfer"]}
    configs = {name: read_json(results_dir / name / "config.json") for name in metrics}
    counts = read_json(results_dir / "cnn" / "class_counts.json")
    summaries = {name: read_json(results_dir / name / "training_summary.json") for name in metrics}
    is_ddam = configs["transfer"].get("architecture", "resnet18") == "ddamfn"
    pretrained = configs["transfer"].get("pretrained", False)
    initialization = (
        "M2 starts from the official DDAMFN++ AffectNet-7 checkpoint; this uses external "
        "expression-labeled pretraining. No extra data is mixed into FER fine-tuning."
        if is_ddam and pretrained else
        "M2 uses ImageNet-1K pretraining." if pretrained else
        "M2 was initialized without pretrained weights in this run."
    )
    transfer_preprocessing = (
        "M2: repeat grayscale to three channels, resize to 112x112; use ImageNet mean/std "
        "to match the pretrained model. Upsampling adds no image detail."
        if is_ddam else "M2 uses the same 48x48 grayscale normalization as M1."
    )
    transfer_method = (
        "M2: DDAMFN with mixed-kernel depthwise face features and two coordinate attention "
        "heads, then a seven-class linear output. AffectNet classifier rows are remapped "
        "to our emotion order. Fine-tune all layers with weighted cross entropy; this is "
        "not a reproduction of the upstream SAM/attention-diversity training recipe."
        if is_ddam else
        "M2: adapted ResNet-18 with a single-channel stride-one stem, no initial max pool, "
        "and Dropout0.30-Linear7 head. Fine-tune every layer."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def page(title, number):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, title, fontsize=17, weight="bold", color="#19324d")
        fig.text(0.08, 0.025, f"FER-2013 | {author}", fontsize=8, color="0.4")
        fig.text(0.91, 0.025, str(number), fontsize=8, color="0.4")
        return fig

    def paragraph(fig, text, y, size=10):
        lines = []
        renderer = fig.canvas.get_renderer()
        font = FontProperties(size=size)
        available_width = fig.bbox.width * 0.84
        for line in text.split("\n"):
            current = ""
            for word in line.split():
                candidate = f"{current} {word}".strip()
                pixels = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
                if pixels > available_width and current:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        fig.text(0.08, y, "\n".join(lines), va="top", fontsize=size, linespacing=1.4)
        return y - len(lines) * (size / 72 / 11.69) * 1.4

    def picture(fig, path, bounds):
        axis = fig.add_axes(bounds)
        if Path(path).is_file():
            axis.imshow(plt.imread(path))
        else:
            axis.text(0.5, 0.5, f"Missing artifact: {Path(path).name}", ha="center", va="center", wrap=True)
        axis.axis("off")

    with PdfPages(output_path) as pdf:
        fig = page("Facial emotion recognition - methods", 1)
        y = paragraph(fig,
            "Data: msambare/fer2013 on Kaggle. Seven emotions; grayscale 48 x 48 pixels. "
            f"{initialization}\n\n"
            "Split protocol: validation is stratified from the original training folder; the supplied "
            "test folder is reserved for final evaluation. The saved split manifests record every file "
            "and label. Model selection uses validation accuracy, with macro F1 as a tie-breaker.", 0.90)
        y = paragraph(fig, "\nSplit sizes: " + ", ".join(f"{split}={sum(values.values()):,}" for split, values in counts.items()) +
                      ".\nClass order: angry, disgust, fear, happy, neutral, sad, surprise.", y - 0.01)
        picture(fig, results_dir / "class_distribution.png", [0.08, 0.48, 0.84, 0.22])
        paragraph(fig,
            "Preprocessing M1: grayscale 48x48, scale to [0,1], normalize mean/std 0.5. "
            f"{transfer_preprocessing} Validation, test and realtime inference use deterministic "
            "preprocessing. Square-root inverse training class frequencies weight the loss. "
            "M1 gentle combined: flip, reflect-padded crops, small affine changes and mild jitter. "
            "M2 fer_moderate: flip, small affine changes, occasional lighting jitter and small erasures. "
            f"Selected profiles: M1={configs['cnn'].get('augmentation', 'baseline')}, "
            f"M2={configs['transfer'].get('augmentation', 'baseline')}.\n\n"
            "M1: four Conv3x3-BatchNorm-ReLU-MaxPool2x2-Dropout blocks, with "
            "32/64/128/256 channels and dropout 0.10/0.15/0.20/0.25. Flatten 2304, "
            "Linear256-BatchNorm-ReLU-Dropout0.50, then Linear7.\n\n"
            f"{transfer_method}\n\n"
            "Both models return logits for cross entropy; softmax is applied at inference.", 0.46, size=9.5)
        pdf.savefig(fig); plt.close(fig)

        fig = page("Training and quantitative comparison", 2)
        text = []
        for name in metrics:
            config, result = configs[name], metrics[name]
            text.append(f"{name.upper()}: AdamW lr={config['learning_rate']}, "
                        f"backbone lr={config['backbone_lr'] if name == 'transfer' else 'n/a'}, "
                        f"weight decay={config['weight_decay']}, smoothing={config['label_smoothing']}, "
                        f"epochs <= {config['epochs']}, patience={config['patience']}, seed={config['seed']}. "
                        f"Cosine LR schedule. Best epoch={result['best_epoch']}. "
                        f"Training device={summaries[name]['device']}. "
                        f"Augmentation={config.get('augmentation', 'baseline')}.")
        paragraph(fig, "\n\n".join(text), 0.90, size=9)
        axis = fig.add_axes([0.08, 0.62, 0.84, 0.11]); axis.axis("off")
        rows = [[name, f"{m['test_accuracy']:.2%}", f"{m['test_macro_f1']:.4f}",
                 f"{m['parameters']:,}", f"{m['cpu_model_fps']:.1f}", "yes" if m["target_met"] else "no"]
                for name, m in metrics.items()]
        table = axis.table(cellText=rows, colLabels=["Model", "Accuracy", "Macro F1", "Params", "CPU FPS", "Target met"],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
        paragraph(fig, "Targets: M1 >= 60%, M2 >= 65%. CPU FPS is batch-one model forward only "
                  f"(float32; M1 1x48x48, M2 {'3x112x112' if is_ddam else '1x48x48'}); "
                  "it excludes preprocessing, face detection and camera I/O. "
                  "Realtime throughput must be measured separately on the demonstration laptop. "
                  f"Benchmark CPU: {metrics['cnn']['cpu']}, threads={metrics['cnn']['cpu_threads']}.", 0.61, size=9)
        picture(fig, results_dir / "cnn" / "training_curves.png", [0.08, 0.31, 0.84, 0.23])
        picture(fig, results_dir / "transfer" / "training_curves.png", [0.08, 0.07, 0.84, 0.23])
        pdf.savefig(fig); plt.close(fig)

        fig = page("Confusion matrices on held-out test data", 3)
        picture(fig, results_dir / "cnn" / "confusion_matrix.png", [0.15, 0.49, 0.70, 0.41])
        picture(fig, results_dir / "transfer" / "confusion_matrix.png", [0.15, 0.07, 0.70, 0.41])
        pdf.savefig(fig); plt.close(fig)

        for page_number, name in [(4, "cnn"), (5, "transfer")]:
            fig = page(f"{name.upper()} - misclassified faces and analysis", page_number)
            picture(fig, results_dir / name / "misclassified.png", [0.06, 0.56, 0.88, 0.34])
            with (results_dir / name / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                errors = [row for row in csv.DictReader(stream) if row["comment"]]
            errors.sort(key=lambda row: int(row["example_number"]))
            paragraphs = [f"{i+1}. {row['true']} -> {row['predicted']}. {row['comment']}"
                          for i, row in enumerate(errors)]
            paragraph(fig, "\n\n".join(paragraphs) if errors else "No test errors were observed.",
                      0.55, size=8.2)
            paragraph(fig, "Observations above describe score errors and suggest visual checks. Add your "
                      "own image-specific observations before submission; score patterns alone cannot "
                      "establish why a face was misclassified.", 0.085, size=8)
            pdf.savefig(fig); plt.close(fig)

        fig = page("Realtime system and demonstration", 6)
        paragraph(fig,
            "Pipeline: OpenCV VideoCapture -> pretrained YuNet face detection (320x320 letterbox) -> crop each face with "
            "a small margin -> shared grayscale preprocessing -> batched PyTorch inference -> "
            "softmax -> bounding boxes, large emotion labels on dark panels, confidence and FPS.\n\n"
            "The command line selects cnn/transfer and accepts a phone URL, webcam, image or file "
            "video. Connection failures trigger bounded retries and a clear error. Low-confidence "
            "predictions are labelled uncertain. Press s to save the phone-camera demonstration.\n\n"
            "The on-screen FPS includes capture and processing, but excludes display and output "
            "encoding. It depends on image resolution, number of faces, CPU threads and Wi-Fi. "
            "The 10 FPS CPU requirement must be checked on the student's own laptop.", 0.90)
        if screenshot and Path(screenshot).is_file():
            picture(fig, screenshot, [0.08, 0.29, 0.84, 0.35])
            paragraph(fig, "Phone-camera demonstration screenshot supplied by the student.", 0.28, size=9)
        else:
            paragraph(fig, "DEMONSTRATION PENDING: no real phone-camera screenshot was supplied. "
                      "Run the app, press s, then regenerate this report with --screenshot. "
                      "This report is a draft until the demonstration and visual error analysis are completed.", 0.55)
        paragraph(fig,
            "Limitations: FER labels can be ambiguous; a visible expression does not reliably reveal "
            "a person's internal emotional state. Strong profiles, small faces and occlusion can hinder detection. "
            "Lighting, pose, occlusion and the FER-to-phone domain shift can reduce accuracy. "
            "Softmax scores are model confidence values, not calibrated probabilities.\n\n"
            "Sources: kaggle.com/datasets/msambare/fer2013; OpenCV Zoo YuNet "
            "(github.com/opencv/opencv_zoo), with model provenance/license in assets/. "
            + ("github.com/SainingZhang/DDAMFN (DDAMFN++ AffectNet-7); pinned source revision "
               "and weight SHA-256 are recorded in pretrained_source.json. " if is_ddam else
               "torchvision ImageNet ResNet-18 documentation. ") +
            "Checkpoints, split manifests, class counts, run configuration, CSV metrics and error "
            "examples accompany the experiment for reproducibility.", 0.23, size=9)
        pdf.savefig(fig); plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("reports/results"))
    parser.add_argument("--output", type=Path, default=Path("reports/report.pdf"))
    parser.add_argument("--screenshot", type=Path, default=Path("reports/demo_screenshot.jpg"))
    parser.add_argument("--author", default="Student name / ID")
    args = parser.parse_args()
    print(build_report(args.results, args.output, args.screenshot, args.author))


if __name__ == "__main__":
    main()
