# REQUIREMENTS

# 1. ultralytics
# 2. streamlit
# 3. moviepy
# 4. streamlit-webrtc


import streamlit as st # To use Streamlit, we need to import it
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import tempfile
import glob, os
from moviepy import VideoFileClip
import shutil
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# First we need to load our best.pt model

@st.cache_resource # This loads model in memory, so it wont reload the model each time
def load_model():
    model = YOLO("electronics_yolo_final_v2.pt")
    return model

model = load_model()


# Now for basic page settings,

st.set_page_config(page_title="Electronic Component Detector", page_icon="🎥" , layout="wide")
st.title("Electronic Component Detector Dashboard")

st.sidebar.header("⚙️ Controls")

st.sidebar.write("---")

mode = st.sidebar.radio("Input Source:", ["Webcam", "Image Upload", "Video Upload"])

st.sidebar.write("---")

confidence = st.sidebar.slider("Confidence Threshold", 0.0 , 1.0, 0.5)

st.sidebar.write("---")


# Next is counting present objects,

def count_objects(result):
    counts = {}

    for box in result.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        counts[name] = counts.get(name, 0) + 1
    return counts


# For video per frame count,

def process_video_with_overlay(input_path, output_path, model, conf):

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Predicting using the best.pt model

    results_stream = model.predict(source=input_path, conf=conf, stream=True)

    # Counting the components and writing them on the video

    for result in results_stream:
        frame = result.plot()
        frame_counts = count_objects(result)

        y_offset = 30
        line_height = 25

        for cls, count in frame_counts.items():
            line = f"{cls}: {count}"
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += line_height

        out.write(frame)

    out.release()

    # Converting because Streamlit cant show .mp4v, we need proper .mp4

    h264_output = output_path.replace(".mp4", "_h264.mp4")

    clip = VideoFileClip(output_path)
    clip.write_videofile(
        h264_output,
        codec="libx264", # Encode method
        audio=False,
        logger=None
    )
    clip.close()

    return h264_output


# Now for image upload,

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.subheader("Uploaded Image")
        st.image(img, use_column_width=True) # shows the uploaded image

        results = model.predict(img_np, conf=confidence)
        plotted = results[0].plot()
        # plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Result")
        st.image(plotted, use_column_width=True) # shows the resulted image

        # Show stats
        st.subheader("Total Detections in Image")
        counts = count_objects(results[0])
        st.dataframe(counts, width="content") # Gives us a nice table!
        

# For video upload,

elif mode == "Video Upload":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if video_file:

        # Removing past video each time there is a new one

        for f in glob.glob("*.mp4"):
            os.remove(f)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") # Save temporary input video (YOLO needs a file path)
        tfile.write(video_file.read())

        st.video(tfile.name)
        st.subheader("Processing Video...")

        overlay_output = "output_overlay.mp4"

        st.info("Generating video with per-frame counts...")
        h264_path = process_video_with_overlay(
            input_path=tfile.name,
            output_path=overlay_output,
            model=model,
            conf=confidence
        )

        st.subheader("YOLO Detection Output")
        st.write("This Video will be deleted when uploading another video, Be sure to move or make a copy!")
        st.video(h264_path, autoplay=True)


# Lastly, Live video/webcam feed

elif mode == "Webcam":
    st.write("Real-time detection... ")

    class YOLOTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model.predict(img, conf=confidence)
            annotated = results[0].plot()

            # Count objects
            counts = count_objects(results)
            st.session_state["counts"] = counts

            return annotated

    # Start webcam
    webrtc_streamer(
        key="example",
        video_transformer_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Live stats
    st.subheader("Real-Time Object Count")

    if "counts" in st.session_state:
        st.write(st.session_state["counts"])
    else:
        st.write("Waiting for detections...")