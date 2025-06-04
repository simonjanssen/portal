# Portal - Python-to-ORT

End-to-end reproduction of neural network predictions in Rust using ONNX checkpoints trained with Python.

> [!NOTE]
> The idea of this repository is to achieve exact reproduction of Python-trained deep learning model predictions in Rust using [ORT crate](). Though ONNX allows reproduction of neural networks, input and output processing is (often) not included but essential for identical results. Main focus of this repository is therefore on proper pre- and postprocessing of images.

This is a hybrid repository containing python scripts to export trained models as ONNX-files, and to run predictions with these ONNX-models in Rust.

## Quickstart
```bash
cargo run --release -- --image images/bus.jpg --model checkpoints/model.onnx
```
Prediction task and matching models are automatically infered from ONNX files (if possible).

## Supported/Tested Models
| Category/Task | Model-Family | Python-Reference / ONNX Export Script | Tested |
| --- | --- | --- | --- |
| 2D Object Detection | D-Fine | [D-Fine]() | ✅ |
| 2D Object Detection | Yolo | [Yolo]() | ✅ |
| Image Classification | Pytorch Image Models / TIMM | [Timm]() | ✅ |
