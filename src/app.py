"""Run from the repository root: python src/app.py --source test.jpg --model cnn."""

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Support both required `python src/app.py` and `python -m src.app` invocations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import yaml

from src.detector import FaceDetector
from src.predict import EmotionPredictor


LOG = logging.getLogger("fer")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path):
    # cv2.imread/imwrite do not reliably handle Unicode filenames on Windows.
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, frame):
    ok, encoded = cv2.imencode(Path(path).suffix.lower(), frame)
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Facial emotion recognition from images, videos or IP cameras.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--source", help="Phone URL, webcam index, image path, or video path.")
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    parser.add_argument("--weights", type=Path, help="Override the configured checkpoint.")
    parser.add_argument("--detector", choices=["yunet", "haar"], help="Face detector (default: pretrained YuNet).")
    parser.add_argument("--detector-size", type=int, help="YuNet input size: 320 for speed, 640 for smaller faces.")
    parser.add_argument("--detector-confidence", type=float, help="Minimum YuNet face-detection confidence.")
    parser.add_argument("--label-scale", type=float, help="Emotion label size multiplier (default: 1.0).")
    parser.add_argument("--device", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--output", type=Path, help="Annotated image or .mp4/.avi video.")
    parser.add_argument("--headless", action="store_true", help="Run without a display window.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed video frames.")
    parser.add_argument("--cpu-threads", type=int)
    return parser.parse_args(argv)


def draw_emotion_label(frame, box, result, threshold, label_scale=1.0):
    """Draw a large, high-contrast two-line label, keeping the panel inside the image."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    confident = result["confidence"] >= threshold
    color = (50, 210, 80) if confident else (0, 180, 255)
    title = result["label"].upper() if confident else "UNCERTAIN"
    detail = (f"{result['confidence']:.1%} confidence" if confident else
              f"Top guess: {result['label']} {result['confidence']:.1%}")
    # Keep text readable at 480p and grow it on higher-resolution camera frames.
    scale = label_scale * max(1.0, min(height / 480, 2.0))
    padding, gap = 8, 6
    title_scale, detail_scale = 1.05 * scale, 0.62 * scale
    font = cv2.FONT_HERSHEY_SIMPLEX

    def measure():
        title_size, title_base = cv2.getTextSize(title, font, title_scale, 2)
        detail_size, detail_base = cv2.getTextSize(detail, font, detail_scale, 1)
        panel_w = max(title_size[0], detail_size[0]) + padding * 2
        panel_h = title_size[1] + title_base + detail_size[1] + detail_base + padding * 2 + gap
        return title_size, title_base, detail_size, detail_base, panel_w, panel_h

    # Text must also fit near frame edges and on small image inputs.
    available_w, available_h = max(1, width - 8), max(1, height - 8)
    for _ in range(8):
        title_size, title_base, detail_size, detail_base, panel_w, panel_h = measure()
        fit = min(1.0, available_w / panel_w, available_h / panel_h)
        if fit >= 1:
            break
        title_scale *= fit * 0.95
        detail_scale *= fit * 0.95
        padding = max(1, round(padding * fit))
        gap = max(1, round(gap * fit))
    left = max(0, min(x1, width - panel_w))
    if y1 >= panel_h + 8:
        top = y1 - panel_h - 8
    elif y2 + panel_h + 8 <= height:
        top = y2 + 8
    else:
        top = max(0, min(y1 + 4, height - panel_h))
    right, bottom = min(width - 1, left + panel_w), min(height - 1, top + panel_h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), (18, 18, 18), cv2.FILLED)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
    title_y = top + padding + title_size[1]
    detail_y = title_y + title_base + gap + detail_size[1]
    cv2.putText(frame, title, (left + padding, title_y), font, title_scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (left + padding, detail_y), font, detail_scale,
                (225, 225, 225), 1, cv2.LINE_AA)
    return left, top, right, bottom


def annotate_frame(frame, detector, predictor, threshold, label_scale=1.0):
    boxes = detector.detect(frame)
    faces = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
    results = predictor.predict_faces(faces)
    for box, result in zip(boxes, results):
        draw_emotion_label(frame, box, result, threshold, label_scale)
    return frame, results


def draw_fps(frame, fps, faces):
    # Bottom-left avoids the emotion labels normally positioned above faces.
    text = f"FPS: {fps:.1f} | Faces: {faces}"
    height, width = frame.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.60
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    if text_w + 16 > width:
        scale *= max(0.05, (width - 16) / (text_w + 1))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    top = max(0, height - text_h - baseline - 16)
    cv2.rectangle(frame, (0, top), (min(width - 1, text_w + 16), height - 1), (18, 18, 18), cv2.FILLED)
    cv2.putText(frame, text, (8, top + text_h + 5), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def open_capture(source, live, camera_config):
    if live and isinstance(source, str):
        # FFmpeg timeout properties bound failed network opens and reads.
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(camera_config.get("open_timeout_ms", 5000)),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(camera_config.get("read_timeout_ms", 5000)),
        ])
    else:
        capture = cv2.VideoCapture(source)
    if live:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def run(args):
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping.")
    source = args.source if args.source is not None else config.get("source", 0)
    source = str(source)
    live = source.isdecimal() or urlparse(source).scheme.lower() in {"http", "https", "rtsp", "rtmp", "udp", "tcp"}
    if source.isdecimal():
        source = int(source)
    elif not live:
        if args.source is None:
            source = str(config_path.parent / source)
        if not Path(source).is_file():
            raise FileNotFoundError(f"Source does not exist: {source}")
    threshold = args.confidence if args.confidence is not None else config.get("confidence_threshold", 0.4)
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    threads = args.cpu_threads if args.cpu_threads is not None else config.get("cpu_threads", 4)
    if threads < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("cpu-threads and max-frames must be positive.")
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)  # Bound CPU parallelism in the YuNet detector too.
    model_paths = config.get("models", config.get("weights", {}))
    weights = args.weights or config_path.parent / model_paths[args.model]
    predictor = EmotionPredictor(weights, args.device or config.get("device", "cpu"))
    if predictor.checkpoint["model_name"] != args.model:
        raise ValueError("The checkpoint model does not match --model.")
    detector_settings = dict(config.get("face_detection", {}))
    if args.detector:
        detector_settings["backend"] = args.detector
    if args.detector_size is not None:
        detector_settings["input_size"] = args.detector_size
    if args.detector_confidence is not None:
        detector_settings["score_threshold"] = args.detector_confidence
    if detector_settings.get("model_path"):
        detector_settings["model_path"] = config_path.parent / detector_settings["model_path"]
    label_scale = args.label_scale if args.label_scale is not None else config.get("display", {}).get("label_scale", 1.0)
    if not np.isfinite(label_scale) or label_scale <= 0:
        raise ValueError("label-scale must be a positive finite number.")
    detector = FaceDetector(**detector_settings)
    LOG.info("Face detector: %s | emotion model: %s | label scale: %.2f",
             detector.backend, args.model, label_scale)
    output = args.output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, str) and not live and output.resolve() == Path(source).resolve():
            raise ValueError("Output must differ from the input file.")
    is_image = isinstance(source, str) and not live and Path(source).suffix.lower() in IMAGE_EXTENSIONS
    if is_image:
        frame = read_image(source)
        if frame is None:
            raise ValueError(f"OpenCV could not decode image: {source}")
        start = time.perf_counter()
        frame, results = annotate_frame(frame, detector, predictor, threshold, label_scale)
        draw_fps(frame, 1 / max(time.perf_counter() - start, 1e-6), len(results))
        output = output or config_path.parent / "reports" / "predictions" / "prediction.jpg"
        if output.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Image output must have an image extension such as .jpg or .png.")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_image(output, frame)
        LOG.info("%d face(s). Saved %s", len(results), output)
        if not args.headless:
            try:
                cv2.imshow("Emotion - press any key to close", frame)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
        return 0
    if output and output.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError("Video output must end with .mp4 or .avi.")
    camera_config = config.get("camera", {})
    max_retries = int(camera_config.get("reconnect_attempts", 5))
    delay = float(camera_config.get("reconnect_delay", 1.0))
    if max_retries < 0 or delay < 0:
        raise ValueError("Reconnect attempts and delay must be nonnegative.")
    capture, writer = None, None
    failures, processed, smoothed_fps = 0, 0, 0.0
    try:
        while True:
            start = time.perf_counter()
            if capture is None:
                try:
                    capture = open_capture(source, live, camera_config)
                except cv2.error as error:
                    LOG.warning("Camera open failed: %s", error)
                    capture = None
            opened = capture is not None and capture.isOpened()
            ok, frame = capture.read() if opened else (False, None)
            if not ok:
                if not live:
                    if not opened or processed == 0:
                        raise ValueError(f"Could not decode video: {source}")
                    LOG.info("End of video after %d frames.", processed)
                    break
                failures += 1
                if capture is not None:
                    capture.release()
                    capture = None
                if failures > max_retries:
                    LOG.error("Camera unavailable after %d retries. Check the URL and Wi-Fi connection.", max_retries)
                    return 2
                LOG.warning("Camera disconnected/unavailable. Retrying %d/%d...", failures, max_retries)
                # Keep the UI responsive to q during the reconnect delay.
                until = time.perf_counter() + delay
                while time.perf_counter() < until:
                    if not args.headless and cv2.waitKey(50) & 0xFF == ord("q"):
                        return 0
                    if args.headless:
                        time.sleep(min(0.05, max(0, until - time.perf_counter())))
                continue
            failures = 0
            frame, results = annotate_frame(frame, detector, predictor, threshold, label_scale)
            fps = 1 / max(time.perf_counter() - start, 1e-6)
            smoothed_fps = fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * fps
            draw_fps(frame, smoothed_fps, len(results))
            if output:
                if writer is None:
                    input_fps = capture.get(cv2.CAP_PROP_FPS)
                    input_fps = input_fps if 0 < input_fps < 240 else 20.0
                    codec = "mp4v" if output.suffix.lower() == ".mp4" else "MJPG"
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), input_fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise OSError(f"Cannot open output video: {output}")
                # Camera resolution can change after reconnection; writer dimensions cannot.
                # The first frame's output size is stored rather than relying on backend get().
                if processed == 0:
                    output_size = (frame.shape[1], frame.shape[0])
                if (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(frame, output_size)
                writer.write(frame)
            processed += 1
            if not args.headless:
                cv2.imshow("Emotion - q: quit, s: screenshot", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    screenshot = config_path.parent / "reports" / "demo_screenshot.jpg"
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                    write_image(screenshot, frame)
                    LOG.info("Screenshot saved: %s", screenshot)
            if args.max_frames and processed >= args.max_frames:
                break
        LOG.info("Processed %d frames. Smoothed capture + processing FPS: %.1f (excludes display/write).",
                 processed, smoothed_fps)
        return 0
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        LOG.info("Stopped.")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, cv2.error, yaml.YAMLError) as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
