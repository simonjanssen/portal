

import onnx 

model = onnx.load_model("./checkpoints/mobilenet/mobilenetv3_large_100.onnx")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")