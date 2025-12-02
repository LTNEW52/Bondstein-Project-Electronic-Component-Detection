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

def count_objects(results):
    counts = {}

    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        counts[name] = counts.get(name, 0) + 1
    return counts


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
        st.subheader("Objects Detected")
        counts = count_objects(results)
        st.write(counts)


# For video upload,

elif mode == "Video Upload":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if video_file:
        if os.path.exists("runs"):
            shutil.rmtree("runs")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") # Save temporary input video (YOLO needs a file path)
        tfile.write(video_file.read())

        st.video(tfile.name)
        st.subheader("Processing Video...")

        # Run YOLO on the uploaded video
        results = model.predict(
            source=tfile.name,
            conf=confidence,
            save=True,
            project="runs/detect",
            name="streamlit_video",
            exist_ok=True,
        )

        # Find YOLO output (could be mp4 or avi)

        mp4_files = glob.glob("runs/detect/streamlit_video/*.mp4")
        avi_files = glob.glob("runs/detect/streamlit_video/*.avi")

        output_path = None

        if mp4_files:
            # YOLO already produced an MP4
            output_path = mp4_files[0]
            st.info("YOLO output is already MP4.")
        elif avi_files:
            # Need to convert AVI → MP4 using moviepy
            avi_path = avi_files[0]
            mp4_path = avi_path.replace(".avi", ".mp4")

            st.info("Converting AVI → MP4 with moviepy...")

            clip = VideoFileClip(avi_path) # New version have from moviepy import instead of moviepy.editor
    
            clip.write_videofile(
                mp4_path,
                codec="libx264",
                audio=False, # No need for audio in this case
                logger=None,  # hides ffmpeg progress inside Streamlit
            )
            clip.close()

            output_path = mp4_path
            st.success("Converted to MP4 successfully!")
        else:
            st.error("No YOLO output video found (neither .mp4 nor .avi).")
            st.stop()

        # Show processed video
        if output_path and os.path.exists(output_path):
            st.subheader("YOLO Detection Output")
            st.video(output_path , autoplay=True)
        else:
            st.error("Converted MP4 not found on disk.")
            st.stop()

        # Stats for last frame / run
        st.subheader("Last Frame Object Count")
        counts = count_objects(results)
        st.write(counts)