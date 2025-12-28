import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Autonomous Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Autonomous Vehicle Detection with YOLOv11")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Load model
@st.cache_resource
def load_model(model_path):
    """Load YOLO model"""
    model = YOLO(model_path)
    return model

# Model path
model_path = st.sidebar.text_input("Model Path", "weights/best.pt")

if os.path.exists(model_path):
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.error("Model file not found. Please add your .pt file to the weights folder.")
    st.stop()

# Confidence threshold
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Upload option
upload_type = st.sidebar.radio("Upload Type", ["Image", "Video"])

if upload_type == "Image":
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Run inference
        with col2:
            st.subheader("Detection Results")
            results = model(image, conf=confidence)
            
            # Plot results
            annotated_image = results[0].plot()
            st.image(annotated_image, channels="BGR", use_container_width=True)
        
        # Display detection details
        st.subheader("Detection Details")
        detections = results[0].boxes
        if len(detections) > 0:
            st.write(f"Total detections: {len(detections)}")
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"Detection {i+1}: {model.names[cls]} (Confidence: {conf:.2f})")
        else:
            st.info("No objects detected.")

elif upload_type == "Video":
    # Video upload
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, dir="temp", suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Run Detection"):
            # Process video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Output video path
            output_path = os.path.join("temp", "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                
                # Write frame
                out.write(annotated_frame)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            st.success("Video processing completed!")
            st.video(output_path)
            
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image or video to start detection.")
