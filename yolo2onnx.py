
import argparse 
from ultralytics import YOLO
from pprint import pprint
import json 

def main(args):
    model = YOLO(f"{args.model}.pt")
    result = model.predict("./images/bus.jpg")
    class_names = {int(k): v for k, v in model.names.items()}
    with open(f"{args.model}.json", "w") as fp:
        json.dump(class_names, fp, indent=4)
    model.export(format="onnx")
    model_onnx = YOLO(f"{args.model}.onnx")
    result_onnx = model.predict("./images/bus.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)
