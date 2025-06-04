# Portal - Python-to-ORT

End-to-end reproduction of neural network predictions in Rust using ONNX checkpoints trained with Python.

> [!NOTE]
> The idea of this repository is to achieve exact reproduction of Python-trained deep learning model predictions in Rust using [ORT crate](https://github.com/pykeio/ort). Though ONNX allows reproduction of neural networks, input and output processing is (often) not included but essential for identical results. Main focus of this repository is therefore on proper preprocessing of input images and postprocessing of (raw) predictions.

This is a hybrid repository containing python scripts to export trained models as ONNX-files, and to run predictions with these ONNX-models in Rust.

## Quickstart

### Step 1: Create ONNX Export
Choosing from one of the available providers, create an ONNX model file. Export scripts from the offiical repositories / docs are linked in the table below.

### Step 2: Run Predictions in Rust
```bash
cargo run --release -- --image images/bus.jpg --model checkpoints/model.onnx
```
Prediction task and matching models are automatically infered from ONNX files (if possible).

## Supported/Tested Models
| Task | Model-Family | Python / ONNX Export Script | Tested |
| --- | --- | --- | --- |
| Object Detection | D-FINE | [link](https://github.com/Peterande/D-FINE/blob/master/tools/deployment/export_onnx.py) | ✅ |
| Object Detection | Ultralytics / YOLO | [link](https://docs.ultralytics.com/integrations/onnx/) | ✅ |
| Classification | HuggingFace / Pytorch Image Models | [link](https://github.com/huggingface/pytorch-image-models/blob/main/onnx_export.py) | ✅ |
