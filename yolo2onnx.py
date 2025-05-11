
import argparse 
from ultralytics import YOLO
from pprint import pprint
import json 

def main(args):
    model = YOLO(f"{args.model}.pt")
    result = model.predict(args.image)
    result[0].save(filename="result_ul.jpg")
    class_names = {int(k): v for k, v in model.names.items()}
    with open(f"{args.model}.json", "w") as fp:
        json.dump(class_names, fp, indent=4)
    model.export(format="onnx")
    model_onnx = YOLO(f"{args.model}.onnx", task="detect")
    result_onnx = model_onnx.predict(args.image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    main(args)
