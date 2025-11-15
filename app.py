# webapp/app.py
import streamlit as st
from PIL import Image
import os
import io
import time
import tempfile

from utils import (
    load_cnn_model,
    run_cnn_opencv_detection,
    run_yolo_detection_if_available,
)

# App config
st.set_page_config(page_title="Smart Pothole Detection", layout="centered")

st.title("Smart Pothole Detection System")
st.markdown("**Author:** Simerjeet Tech")
st.markdown(
    """
    Upload a road image and choose a detection method.
    - **CNN + OpenCV**: primary method (region proposals with OpenCV + MobileNetV2 classifier)
    - **YOLOv8**: object-detection (if you have trained YOLO weights and ultralytics installed)
    """
)

# Sidebar options
st.sidebar.header("Options")
method = st.sidebar.radio("Detection method", ("CNN + OpenCV (Primary)", "YOLOv8 (if available)"))
threshold = st.sidebar.slider("CNN probability threshold", 0.1, 0.9, 0.5, step=0.05)
show_boxes = st.sidebar.checkbox("Show bounding boxes on output", value=True)

uploaded_file = st.file_uploader("Upload an image (.jpg/.png)", type=["jpg", "jpeg", "png"])

# Model load area
model_load_state = st.sidebar.empty()
if method.startswith("CNN"):
    model_load_state.info("Loading CNN model (MobileNetV2)...")
    try:
        cnn_model = load_cnn_model()  # tries to load models/best_model.h5 or models/final_model.h5
        model_load_state.success("CNN model loaded.")
    except Exception as e:
        model_load_state.error(f"Failed to load CNN model: {e}")
        cnn_model = None
else:
    model_load_state.info("YOLOv8 will be used (if ultralytics & weights present)")

# Inference button
if uploaded_file is not None:
    # convert uploaded file to image
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run Detection"):
        timestamp = int(time.time())
        tmpdir = tempfile.mkdtemp(prefix=f"pothole_{timestamp}_")
        input_path = os.path.join(tmpdir, "input.jpg")
        image.save(input_path)

        with st.spinner("Running detection..."):
            if method.startswith("CNN"):
                if cnn_model is None:
                    st.error("CNN model not loaded. Please place a Keras model in `models/best_model.h5` or `models/final_model.h5`.")
                else:
                    out_path = os.path.join(tmpdir, "cnn_output.jpg")
                    try:
                        results = run_cnn_opencv_detection(
                            model=cnn_model,
                            image_path=input_path,
                            out_path=out_path,
                            prob_threshold=threshold,
                            show_boxes=show_boxes,
                            img_size=224
                        )
                        if results:
                            st.image(out_path, caption="CNN + OpenCV Detection", use_column_width=True)
                            st.success("Detection finished.")
                        else:
                            st.warning("No candidate regions detected by OpenCV.")
                    except Exception as e:
                        st.exception(e)
            else:
                # YOLO
                try:
                    yolo_out = run_yolo_detection_if_available(source=input_path, out_dir=tmpdir)
                    if yolo_out:
                        st.image(yolo_out, caption="YOLOv8 Detection", use_column_width=True)
                        st.success("YOLO inference finished.")
                    else:
                        st.warning("YOLO is not available or no output saved. Make sure `ultralytics` is installed and `weights` path is configured in utils.")
                except Exception as e:
                    st.exception(e)

        # Clean-up note
        st.markdown(f"Temporary files saved to `{tmpdir}` (for debugging).")
else:
    st.info("Upload an image to get started. You can also place a trained Keras model in `models/` and YOLO weights in `object_detection_yolo/runs/...`")

st.markdown("---")
st.markdown(
    "## Notes\n"
    "- Place your trained Keras model as `models/best_model.h5` or `models/final_model.h5` at the repository root.\n"
    "- If you want to use YOLOv8 in the web app, install `ultralytics` and provide weights path in `webapp/utils.py` or put the best `.pt` under `object_detection_yolo/runs/...`.\n"
    "- This app is intended for demo/testing. For production, containerize and add security checks.\n"
)
