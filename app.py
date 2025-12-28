import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Bangladesh Autonomous Vehicle Perception",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🇧🇩 Bangladesh Autonomous Vehicle Perception System (YOLOv11)")
st.markdown("### Real-time Vehicle Detection for Autonomous Driving")
st.markdown("---")

# Load model with error handling
@st.cache_resource
def load_model():
    """Load YOLO model with error handling"""
    model_path = "weights/best.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("📁 Please place your trained YOLOv11 model (best.pt) in the 'weights' folder.")
        st.stop()
    
    try:
        model = YOLO(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load the model
model = load_model()

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# File uploader
st.sidebar.subheader("📤 Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video",
    type=["jpg", "jpeg", "png", "mp4"],
    help="Supported formats: JPG, PNG, MP4"
)

# Confidence threshold
confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Lower values detect more objects but may include false positives"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Upload an image or video to start vehicle detection")

# Main content
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Image Processing
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("📸 Image Processing")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### Detection Results")
            
            # Run inference
            with st.spinner("🔍 Running detection..."):
                results = model(image, conf=confidence)
                
                # Get annotated image
                annotated_image = results[0].plot()
                
                # Convert BGR to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_column_width=True)
        
        # Display detection statistics
        st.markdown("---")
        st.subheader("📊 Detection Statistics")
        
        detections = results[0].boxes
        if len(detections) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detections", len(detections))
            
            # Count detections by class
            class_counts = {}
            for box in detections:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            with col2:
                st.metric("Unique Classes", len(class_counts))
            
            with col3:
                highest_conf = max([float(box.conf[0]) for box in detections])
                st.metric("Highest Confidence", f"{highest_conf:.2%}")
            
            # Detailed detection list
            st.markdown("#### 🔍 Detected Objects:")
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                st.write(f"**{i+1}.** {class_name} - Confidence: {conf:.2%}")
        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
    
    # Video Processing
    elif file_extension == 'mp4':
        st.subheader("🎥 Video Processing")
        
        # Save uploaded video to temp folder
        temp_input_path = os.path.join("temp", "input_video.mp4")
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display original video
        st.markdown("#### Original Video")
        st.video(temp_input_path)
        
        # Process button
        if st.button("▶️ Run Detection", type="primary"):
            st.markdown("#### Processed Video")
            
            # Open video
            cap = cv2.VideoCapture(temp_input_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video path
            temp_output_path = os.path.join("temp", "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()
            
            frame_count = 0
            detection_summary = []
            
            # Process video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                
                # Write to output video
                out.write(annotated_frame)
                
                # Store detection count for this frame
                detection_summary.append(len(results[0].boxes))
                
                # Update progress every 10 frames for web optimization
                frame_count += 1
                if frame_count % 10 == 0 or frame_count == total_frames:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"⏳ Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                    
                    # Display current frame (for preview)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_column_width=True)
            
            # Release resources
            cap.release()
            out.release()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            frame_placeholder.empty()
            
            # Show processed video
            st.success("✅ Video processing completed!")
            st.video(temp_output_path)
            
            # Video statistics
            st.markdown("---")
            st.subheader("📊 Video Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Frames", total_frames)
            
            with col2:
                avg_detections = sum(detection_summary) / len(detection_summary)
                st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")
            
            with col3:
                max_detections = max(detection_summary)
                st.metric("Max Detections", max_detections)
            
            # Cleanup temporary input file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

else:
    # Welcome screen
    st.info("👈 Please upload an image or video from the sidebar to begin detection.")
    
    st.markdown("### 🎯 Features:")
    st.markdown("""
    - **Real-time Object Detection** using YOLOv11
    - **Image Analysis** with bounding boxes and confidence scores
    - **Video Processing** with frame-by-frame detection
    - **Adjustable Confidence** threshold for detection sensitivity
    - **Detailed Statistics** for all detections
    """)
    
    st.markdown("### 🚀 Getting Started:")
    st.markdown("""
    1. Ensure your trained YOLOv11 model (`best.pt`) is in the `weights/` folder
    2. Upload an image (JPG, PNG) or video (MP4) using the sidebar
    3. Adjust the confidence threshold if needed
    4. For videos, click "Run Detection" to process
    """)
