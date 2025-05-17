
# https://huggingface.co/docs/timm/quickstart
# https://github.com/huggingface/pytorch-image-models/blob/main/onnx_validate.py

from PIL import Image
import timm 
import torch 

def main():
    img = Image.open("./images/bus.jpg")
    model = timm.create_model('mobilenetv3_large_100', pretrained=True).eval()
    config = timm.data.resolve_data_config(model.pretrained_cfg)
    print(config)
    transform = timm.data.create_transform(**config)
    print(transform)
    img_tensor = transform(img)
    print(img_tensor.shape)

    output = model(img_tensor.unsqueeze(0))
    print(output.shape)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    print(indices, values)


if __name__ == "__main__":
    main()
