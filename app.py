import streamlit as st
from ultralytics import YOLO
import cv2
import time
import tempfile
import numpy as np
from datetime import datetime
import pandas as pd

# ------------------ Page Config ------------------
st.set_page_config(page_title="🔥 Fire Detection", layout="wide")
st.title("🔥 Real-Time Fire Detection Dashboard")
st.markdown("Powered by **YOLOv8 + Streamlit**")

# ------------------ Session State Initialization ------------------
if 'model' not in st.session_state:
    st.session_state.model = YOLO("best.pt")
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# ------------------ Sidebar Controls ------------------
st.sidebar.title("🧭 Control Panel")
source_option = st.sidebar.selectbox("📥 Input Source", ["Webcam", "Upload Image", "Upload Video"])
confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
play_sound = st.sidebar.toggle("🔊 Enable Alert Sound", value=True)
show_logs = st.sidebar.toggle("📜 Show Detection Logs", value=True)
theme_toggle = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

st.session_state.theme = theme_toggle.lower()

start_detection = st.sidebar.button("🚀 Start Detection")
stop_detection = st.sidebar.button("🛑 Stop Detection")

# ------------------ Custom Styling ------------------
def set_theme():
    css = """
        <style>
        body {
            background-color: %s;
            color: %s;
        }
        .log-box {
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            background-color: %s;
            color: %s;
            font-size: 16px;
        }
        </style>
    """ % (
        "#0E1117" if st.session_state.theme == 'dark' else "#FFFFFF",
        "#FFFFFF" if st.session_state.theme == 'dark' else "#000000",
        "#1E1E1E" if st.session_state.theme == 'dark' else "#F0F0F0",
        "#FFFFFF" if st.session_state.theme == 'dark' else "#000000"
    )
    st.markdown(css, unsafe_allow_html=True)

set_theme()

# ------------------ Placeholders ------------------
status_placeholder = st.empty()
image_placeholder = st.empty()
progress = st.progress(0)

col1, col2 = st.columns([2, 1])

# ------------------ Utility Functions ------------------
def detect_fire(frame):
    results = st.session_state.model.predict(source=frame, imgsz=640, conf=confidence_threshold, verbose=False)
    annotated = results[0].plot()
    fire_detected = any(cls == 0 for cls in results[0].boxes.cls)
    return fire_detected, annotated

def play_alert():
    if play_sound:
        st.components.v1.html("""
            <script>
                const beep = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
                beep.play();
            </script>
        """, height=0)

def log_event(fire, timestamp):
    message = "🔥 Fire Detected" if fire else "✅ No Fire"
    st.session_state.detection_log.append({"timestamp": timestamp, "event": message})

def export_log_to_csv():
    if st.session_state.detection_log:
        df = pd.DataFrame(st.session_state.detection_log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Log as CSV",
            data=csv,
            file_name="fire_detection_log.csv",
            mime="text/csv"
        )

def display_log():
    if show_logs and st.session_state.detection_log:
        with col2:
            st.markdown("<div class='log-box'>📋 <b>Detection Log</b></div>", unsafe_allow_html=True)
            df = pd.DataFrame(st.session_state.detection_log).sort_values(by="timestamp", ascending=False)
            st.dataframe(df, use_container_width=True)
            export_log_to_csv()

# ------------------ Detection Logic ------------------
if start_detection and not stop_detection:
    if source_option == "Webcam":
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Failed to read webcam frame.")
                break

            fire, annotated = detect_fire(frame)
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_event(fire, timestamp)

            if fire:
                status_placeholder.warning("🔥 Fire Detected!")
                play_alert()
                progress.progress(100)
            else:
                status_placeholder.success("✅ No Fire Detected.")
                progress.progress(0)

            with col1:
                image_placeholder.image(annotated, channels="BGR", use_container_width=True)

            if stop_detection:
                break
            time.sleep(0.03)

        cap.release()
        st.success("✅ Webcam detection stopped.")

    elif source_option == "Upload Image":
        image_file = st.sidebar.file_uploader("🖼 Upload Image", type=["jpg", "jpeg", "png"])
        if image_file:
            image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, 1)
            fire, annotated = detect_fire(frame)
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_event(fire, timestamp)

            with col1:
                image_placeholder.image(annotated, channels="BGR", use_container_width=True)

            if fire:
                status_placeholder.warning("🔥 Fire Detected in Image!")
                play_alert()
            else:
                status_placeholder.success("✅ No Fire in Image.")

    elif source_option == "Upload Video":
        video_file = st.sidebar.file_uploader("🎞 Upload Video", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                fire, annotated = detect_fire(frame)
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_event(fire, timestamp)

                if fire:
                    status_placeholder.warning("🔥 Fire Detected!")
                    play_alert()
                    progress.progress(100)
                else:
                    status_placeholder.success("✅ No Fire Detected.")
                    progress.progress(0)

                with col1:
                    image_placeholder.image(annotated, channels="BGR", use_container_width=True)

                if stop_detection:
                    break
                time.sleep(0.03)

            cap.release()
            st.success("✅ Video processing completed.")

# ------------------ Show Detection Log ------------------
display_log()
