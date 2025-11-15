# YOLOv8 Pothole Detection

Requirements:
pip install ultralytics



Prepare dataset in YOLO format (images + labels) and update `data.yaml`.

Train:
python yolo_train.py --data_yaml data.yaml --model yolov8n.pt --epochs 50

Infer:

python yolo_inference.py --weights runs/yolo/pothole_yolov8/weights/best.pt --source ../examples --out ../yolo_outputs

Notes & Tips

. Use labelImg or Roboflow to label images in YOLO format. Each label file must match its image filename and contain lines: class x_center y_center width height (normalized).

. ultralytics requires recent Python (3.8+). If you run into GPU issues, try device='cpu' in model.train() / model.predict().

. The data.yaml path field should be the parent folder that contains images/ and labels/ subfolders. Example:

data/yolo_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/


How to run the Streamlit app

1. From repository root, install requirements (if not already):

pip install -r requirements.txt

2. Run:

streamlit run webapp/app.py


3. Place your trained Keras model at:

models/best_model.h5
# or
models/final_model.h5


How to build & run (local, CPU)

1. Build the image
docker build -t smart-pothole-app:latest .


2. Run the container (mount your local models/ and examples/ so you can update models without rebuilding)

docker run --rm -it \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/data:/app/data \
  --name smart-pothole \
  smart-pothole-app:latest


Open your browser at [http://localhost:8501](http://localhost:8501).
