# Pretrained YuNet face detector

`face_detection_yunet_2023mar.onnx` is the unmodified pretrained detector from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet).
It is distributed under the included [MIT license](YUNET_LICENSE.txt).

- Source commit: `48b10fa505aeb51fbd127fc60a5c660040e823b7`
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Size: 232,589 bytes
- [Pinned download](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)

The 2023 model supports this project's OpenCV 4.x runtime. The app uses the
FP32 detector on CPU at 320x320 by default. Preprocessing preserves aspect ratio
with letterboxing, and returned boxes are mapped back to the original frame.
These detector weights are separate from the trained FER emotion checkpoints.
