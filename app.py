import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Construction Safety – YOLOv11",
    layout="wide",
    page_icon="🏗️"
)

# ==================================================
# GLOBAL STYLE + BACKGROUND
# ==================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    width: 100%;
    overflow-x: hidden;
}

/* BACKGROUND */
.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
    url("https://static.vecteezy.com/system/resources/thumbnails/040/988/694/small/ai-generated-construction-worker-s-safety-helmet-and-work-tools-with-sunset-sky-in-the-background-photo.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* REMOVE STREAMLIT DEFAULT UI */
header, footer, #MainMenu {
    visibility: hidden;
}

/* GLASS CARD */
.glass {
    background: rgba(20, 25, 30, 0.82);
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 30px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffb74d, #f57c00, #1c1c1c);
}
section[data-testid="stSidebar"] * {
    color: black !important;
    font-weight: 600;
}

/* TITLES */
h1 { text-align:center; color:#FFD369; font-weight:800; }
h2 { color:#FFCC80; }

/* IMAGES / VIDEO */
img, video {
    border-radius: 16px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}

/* ===========================
   CUSTOM UPLOAD BOX
=========================== */
.upload-box {
    border: 2px dashed rgba(255,255,255,0.4);
    border-radius: 18px;
    padding: 35px;
    text-align: center;
    color: white;
    background: rgba(20,25,30,0.65);
    margin-bottom: 20px;
    transition: 0.3s;
}
.upload-box:hover {
    border-color: #FFD369;
    background: rgba(20,25,30,0.85);
}
.upload-icon {
    font-size: 42px;
    margin-bottom: 10px;
}
.upload-text {
    font-size: 17px;
    font-weight: bold;
}
.upload-hint {
    font-size: 13px;
    opacity: 0.7;
}

/* ===========================
   HIDE WEBRTC WHITE BAR ONLY
=========================== */
.webrtc-media-container button {
    display: none !important;
}

/* ===========================
   FORCE SIDEBAR TO BE VISIBLE
=========================== */
button[kind="header"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.image(
    "https://static.vecteezy.com/system/resources/thumbnails/055/660/493/small/construction-safety-vest-helmet-tools-equipment-png.png",
    width=120
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Image", "🎥 Video", "📷 Webcam"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return YOLO("best.onnx", task="detect")

model = load_model()

# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="glass" style="text-align:center;">
        <h1 style="font-size:42px;">👷 AI-Powered Construction Safety</h1>
        <p style="font-size:18px; max-width:900px; margin:auto;">
        This system enhances construction site safety by automatically detecting whether workers
        are wearing required Personal Protective Equipment (PPE) using advanced computer vision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h2>🔍 Project Summary</h2>
        <p>
        This project improves safety compliance at construction sites by detecting workers
        who are not wearing required PPE such as helmets and safety vests.
        </p>
        <p>
        It uses a <b>YOLOv11m</b> object detection model trained on
        <b>2.2k images</b> from a Roboflow dataset with YOLO default augmentations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h2>🎯 Objectives</h2>
        <ul>
            <li>Ensure PPE compliance (Helmet & Vest detection)</li>
            <li>Provide real-time alerts and monitoring</li>
            <li>Build a lightweight, deployable AI solution</li>
            <li>Support image, video, and live webcam detection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# IMAGE PAGE
# ==================================================
elif page == "🔍 Image":

    st.markdown("""
    <div class="glass">
        <h2>📸 Image Detection</h2>
        <p>Upload an image to detect helmets and safety vests.</p>
    </div>
    """, unsafe_allow_html=True)

    img_file = st.file_uploader("", ["jpg", "png", "jpeg"], label_visibility="collapsed")
    if img_file:
        img = np.array(Image.open(img_file))
        st.image(model(img, conf=confidence)[0].plot(), use_column_width=True)

# ==================================================
# VIDEO PAGE
# ==================================================
elif page == "🎥 Video":

    st.markdown("""
    <div class="glass">
        <h2>🎥 Video Detection</h2>
        <p>Upload a video and preview PPE detection (Cloud-safe mode).</p>
    </div>
    """, unsafe_allow_html=True)

    vid = st.file_uploader("Upload Video", ["mp4", "avi", "mov"])
    if vid:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(vid.read())
        cap = cv2.VideoCapture(tmp.name)

        frame_box = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence, verbose=False)
            frame_box.image(results[0].plot(), channels="BGR")
        cap.release()

# ==================================================
# WEBCAM PAGE
# ==================================================
elif page == "📷 Webcam":

    st.markdown("""
    <div class="glass">
        <h2>📷 Real-Time Webcam Detection</h2>
        <p>Allow camera access to start live PPE detection.</p>
    </div>
    """, unsafe_allow_html=True)

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            return model(img, conf=confidence)[0].plot()

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
