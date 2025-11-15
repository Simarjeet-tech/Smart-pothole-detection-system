# webapp/utils.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Optional import for YOLO; if not available, YOLO functions gracefully return None
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

def load_cnn_model():
    """
    Try to load a Keras model from common locations.
    Raise exception if none found.
    """
    possible = [
        os.path.join("models", "best_model.h5"),
        os.path.join("models", "final_model.h5"),
        "models/best_model.h5",
        "models/final_model.h5"
    ]
    for p in possible:
        if os.path.exists(p):
            model = load_model(p)
            return model
    raise FileNotFoundError("No Keras model found in models/. Put best_model.h5 or final_model.h5 there.")

def detect_regions_opencv(image_bgr):
    """
    Same heuristic as src/inference_cv_and_cnn.detect_regions
    Returns list of boxes (x,y,w,h)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = gray.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < 10 or hh < 10 or ww > 0.9*w or hh > 0.9*h:
            continue
        boxes.append((x, y, ww, hh))
    return boxes

def classify_crop(model, crop_bgr, target_size=224):
    """
    Preprocess and classify a crop using MobileNetV2-style preprocessing.
    Returns probability (float).
    """
    crop = cv2.resize(crop_bgr, (target_size, target_size))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    arr = img_to_array(crop_rgb)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    return float(pred[0][0])

def run_cnn_opencv_detection(model, image_path, out_path="output.jpg", prob_threshold=0.5, show_boxes=True, img_size=224):
    """
    Run the heuristic OpenCV localization + CNN classification on one image.
    Saves annotated image to out_path and returns out_path (or None if nothing found).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    boxes = detect_regions_opencv(img)
    annotated = img.copy()
    any_box = False
    for (x,y,w,h) in boxes:
        crop = img[y:y+h, x:x+w]
        prob = classify_crop(model, crop, target_size=img_size)
        label = f"Pothole: {prob:.2f}" if prob >= prob_threshold else f"No: {prob:.2f}"
        color = (0,0,255) if prob >= prob_threshold else (0,255,0)
        if show_boxes:
            cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
            cv2.putText(annotated, label, (x, max(y-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        any_box = True

    # Also add whole-image probability (optional): average over boxes or classify whole image
    if not any_box:
        # Save unchanged or a note
        cv2.putText(annotated, "No candidate regions detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_path, annotated)
    return out_path

def run_yolo_detection_if_available(source, out_dir="yolo_outputs", weights_path=None):
    """
    Run YOLOv8 inference if ultralytics is installed and weights are available.
    Returns path to saved annotated image (first annotated file) or None.
    """
    if not ULTRALYTICS_AVAILABLE:
        return None

    # default weights location: try to find best.pt under object_detection_yolo runs
    if weights_path is None:
        # common ultralytics structure: object_detection_yolo/runs/train/pothole_yolov8/weights/best.pt
        candidate = os.path.join("object_detection_yolo", "runs", "yolo", "pothole_yolov8", "weights", "best.pt")
        # also check runs/yolo
        candidate2 = os.path.join("object_detection_yolo", "runs", "yolov8", "pothole_yolov8", "weights", "best.pt")
        candidate3 = "best.pt"
        for c in [candidate, candidate2, candidate3]:
            if os.path.exists(c):
                weights_path = c
                break

    if weights_path is None or not os.path.exists(weights_path):
        # let user know weights not found
        return None

    model = YOLO(weights_path)
    # ultralytics saves results in runs/detect by default; we set save to True
    results = model.predict(source=source, save=True)
    # find latest runs/detect/exp* folder
    run_root = "runs/detect"
    if not os.path.exists(run_root):
        return None
    exps = sorted([os.path.join(run_root, d) for d in os.listdir(run_root)], key=os.path.getmtime)
    if not exps:
        return None
    latest = exps[-1]
    # Pick first annotated image found there
    for f in os.listdir(latest):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            out_img = os.path.join(latest, f)
            # Move or copy to out_dir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, os.path.basename(out_img))
            try:
                # copy file
                import shutil
                shutil.copy(out_img, dst)
                return dst
            except Exception:
                return out_img
    return None
