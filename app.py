import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="PPE Detection System",
    layout="centered"
)

st.title("🦺 PPE Detection System")
st.write("Detect Personal Protective Equipment in images, videos, or live webcam feed")

# ---------------- Load YOLO Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video", "Webcam"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

# ================= IMAGE MODE =================
if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

        with st.spinner("Detecting PPE..."):
            results = model(img_array, conf=conf_threshold)
            annotated_img = results[0].plot()

        st.subheader("✅ Detection Result")
        st.image(annotated_img, use_container_width=True)

# ================= VIDEO MODE =================
elif mode == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("❌ Unable to open video")
        else:
            st.info("▶️ Processing video...")
            while cap.isOpened():
                ret, frame = cap.read()

                # ---- SAFETY CHECKS ----
                if not ret or frame is None or frame.size == 0:
                    break

                frame = cv2.resize(frame, (640, 480))

                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()

                if annotated_frame is None or annotated_frame.size == 0:
                    continue

                annotated_frame = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )

                stframe.image(
                    annotated_frame,
                    use_container_width=True
                )

                time.sleep(0.03)  # limit FPS

            cap.release()
            st.success("✅ Video processing completed")

# ================= WEBCAM MODE =================
elif mode == "Webcam":
    st.warning("⚠️ Click Stop to end webcam streaming")

    run = st.checkbox("Start Webcam")
    frame_window = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
        else:
            while run:
                ret, frame = cap.read()

                # ---- SAFETY CHECKS ----
                if not ret or frame is None or frame.size == 0:
                    continue

                frame = cv2.resize(frame, (640, 480))

                results = model(frame, conf=conf_threshold)
                annotated_frame = results[0].plot()

                if annotated_frame is None or annotated_frame.size == 0:
                    continue

                annotated_frame = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )

                frame_window.image(
                    annotated_frame,
                    use_container_width=True
                )

                time.sleep(0.03)

            cap.release()
