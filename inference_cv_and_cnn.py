import cv2
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        boxes.append((x,y,ww,hh))

    return boxes

def classify_crop(model, crop, size):
    crop = cv2.resize(crop, (size, size))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = img_to_array(crop)
    crop = preprocess_input(crop)
    crop = np.expand_dims(crop, axis=0)
    pred = model.predict(crop)[0][0]
    return float(pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    model = load_model(args.model_path)
    image = cv2.imread(args.input)

    boxes = detect_regions(image)
    annotated = image.copy()

    for (x,y,w,h) in boxes:
        crop = image[y:y+h, x:x+w]
        prob = classify_crop(model, crop, args.img_size)
        label = f"Pothole: {prob:.2f}" if prob >= 0.5 else f"No: {prob:.2f}"
        color = (0,0,255) if prob >= 0.5 else (0,255,0)

        cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
        cv2.putText(annotated, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(args.output + "/output.jpg", annotated)
    print("Saved:", args.output + "/output.jpg")

if __name__ == "__main__":
    main()
