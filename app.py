import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==================================================
# PAGE CONFIG (SIDEBAR ALWAYS VISIBLE)
# ==================================================
st.set_page_config(
    page_title="Construction Safety – YOLOv11",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
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

/* FIX PNG VISIBILITY */
img {
    background: rgba(255,255,255,0.95);
    padding: 10px;
    border-radius: 14px;
}

/* UPLOAD BOX */
.upload-box {
    border: 2px dashed rgba(255,255,255,0.4);
    border-radius: 18px;
    padding: 35px;
    text-align: center;
    color: white;
    background: rgba(20,25,30,0.65);
    margin-bottom: 20px;
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
    "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
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
    <div class="glass">
        <h1>👷 AI-Powered Construction Safety</h1>
        <p>
        Automated detection of safety helmets and vests at construction sites using YOLOv11.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# IMAGE PAGE
# ==================================================
elif page == "🔍 Image":
    st.markdown("""
    <div class="glass">
        <h2>📸 Image Detection</h2>
        <p>Upload an image to analyze PPE compliance.</p>
    </div>
    """, unsafe_allow_html=True)

    img_file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if img_file:
        img = np.array(Image.open(img_file))
        results = model(img, conf=confidence, verbose=False)
        st.image(results[0].plot(), use_column_width=True)

# ==================================================
# VIDEO PAGE (CLOUD SAFE – PREVIEW MODE)
# ==================================================
elif page == "🎥 Video":
    st.markdown("""
    <div class="glass">
        <h2>🎥 Video Detection</h2>
        <p>Cloud-safe preview mode (no video saving).</p>
    </div>
    """, unsafe_allow_html=True)

    vid = st.file_uploader("Upload Video", ["mp4", "avi", "mov"])

    if vid:
        st.info("▶️ Processing video preview…")

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(vid.read())
        cap = cv2.VideoCapture(tmp.name)

        frame_box = st.empty()
        SKIP_FRAMES = 5
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % SKIP_FRAMES != 0:
                continue

            frame = cv2.resize(frame, (416, 416))
            results = model(frame, conf=confidence, verbose=False)
            frame_box.image(results[0].plot(), channels="BGR")

        cap.release()
        st.success("✅ Video preview finished")

        st.warning(
            "Full video export is available only in local mode "
            "due to codec limitations on Streamlit Cloud."
        )

# ==================================================
# WEBCAM PAGE
# ==================================================
elif page == "📷 Webcam":
    st.markdown("""
    <div class="glass">
        <h2>📷 Real-Time Webcam Detection</h2>
        <p>Desktop browsers recommended.</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ Webcam detection works best on desktop browsers. "
        "Mobile devices may show camera preview only."
    )

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=confidence, verbose=False)
            return results[0].plot()

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
