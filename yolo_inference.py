# object_detection_yolo/yolo_inference.py
"""
Run YOLOv8 inference.
Usage examples:
  python yolo_inference.py --weights runs/yolo/pothole_yolov8/weights/best.pt --source ../examples --out outputs_yolo
  python yolo_inference.py --weights yolov8n.pt --source ../examples/test1.jpg
"""

from ultralytics import YOLO
import argparse
import os
import shutil

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt")
    p.add_argument("--source", type=str, required=True, help="Image or folder of images")
    p.add_argument("--out", type=str, default="yolo_outputs", help="Output folder for annotated images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return p.parse_args()

def main():
    args = parse_args()
    # prepare output folder
    if os.path.exists(args.out):
        try:
            shutil.rmtree(args.out)
        except Exception:
            pass
    os.makedirs(args.out, exist_ok=True)

    print("Loading model:", args.weights)
    model = YOLO(args.weights)

    # run predict
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,         # saves annotated images to 'runs/detect' by default
        save_conf=True     # include confidence in label
    )

    # move saved results to the chosen out folder (results saved under 'runs/detect/exp*')
    # ultralytics saves results into runs/detect unless you pass save_dir; move them for convenience
    run_root = "runs/detect"
    if os.path.exists(run_root):
        # find the latest exp folder
        exps = sorted([os.path.join(run_root, d) for d in os.listdir(run_root)], key=os.path.getmtime)
        if exps:
            latest = exps[-1]
            for f in os.listdir(latest):
                src = os.path.join(latest, f)
                dst = os.path.join(args.out, f)
                try:
                    shutil.move(src, dst)
                except Exception:
                    pass
            print("Saved YOLO annotated outputs to:", args.out)
        else:
            print("No runs/detect/exp* folder found; check ultralytics output.")
    else:
        print("No runs/detect folder found; prediction may have failed or saved elsewhere.")

if __name__ == "__main__":
    main()
