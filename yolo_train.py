# object_detection_yolo/yolo_train.py
"""
Train YOLOv8 on your pothole dataset.
Requirements: pip install ultralytics

Dataset layout (under data_yaml path):
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt   (YOLO format: class x_center y_center width height normalized)
  labels/val/*.txt
"""

from ultralytics import YOLO
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_yaml", type=str, default="data.yaml", help="Path to dataset YAML")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLOv8 model to start from (yolov8n.pt/yolov8s.pt...)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--project", type=str, default="../runs/yolo", help="Where to save runs")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.project, exist_ok=True)
    print(f"Training YOLOv8: data={args.data_yaml}, model={args.model}")
    model = YOLO(args.model)  # loads pretrained weights
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name="pothole_yolov8",
        exist_ok=True,
        device=0  # change or set to 'cpu' if no GPU
    )
    print("Training finished. Check runs in:", os.path.join(args.project, "pothole_yolov8"))

if __name__ == "__main__":
    main()
