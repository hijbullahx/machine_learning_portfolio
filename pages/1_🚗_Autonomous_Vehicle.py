import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Set page configuration
st.set_page_config(
    page_title="Autonomous Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    .detection-list {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
    }
    
    .sidebar .element-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #999;
        text-align: center;
        padding: 15px 0;
        font-size: 13px;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .contributors-section {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .contributor-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .contributor-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">Bangladesh Autonomous Vehicle Obstacle Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Perception for Dense Traffic & Complex Road Conditions</p>', unsafe_allow_html=True)
st.markdown("---")

# Technologies & Frameworks
st.markdown('<h2 style="font-size: 1.8rem; font-weight: 700; margin: 1.5rem 0;">🛠️ Technologies & Frameworks</h2>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); transition: transform 0.3s ease;">
        <h3 style="color: white !important; font-weight: 600; margin-bottom: 1rem;">🧠 Deep Learning</h3>
        <ul style="list-style: none; padding: 0;">
            <li>✓ PyTorch</li>
            <li>✓ TensorFlow</li>
            <li>✓ Ultralytics YOLO</li>
            <li>✓ OpenCV</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); transition: transform 0.3s ease;">
        <h3 style="color: white !important; font-weight: 600; margin-bottom: 1rem;">👁️ Computer Vision</h3>
        <ul style="list-style: none; padding: 0;">
            <li>✓ Object Detection</li>
            <li>✓ Image Segmentation</li>
            <li>✓ Video Analysis</li>
            <li>✓ Real-time Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); transition: transform 0.3s ease;">
        <h3 style="color: white !important; font-weight: 600; margin-bottom: 1rem;">💻 Development</h3>
        <ul style="list-style: none; padding: 0;">
            <li>✓ Python</li>
            <li>✓ Streamlit</li>
            <li>✓ Git/GitHub</li>
            <li>✓ Docker</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); transition: transform 0.3s ease;">
        <h3 style="color: white !important; font-weight: 600; margin-bottom: 1rem;">🔬 Research Areas</h3>
        <ul style="list-style: none; padding: 0;">
            <li>✓ Autonomous Vehicles</li>
            <li>✓ Traffic Analysis</li>
            <li>✓ Edge AI</li>
            <li>✓ Model Optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Load model with error handling
@st.cache_resource
def load_model():
    """Load YOLO model with error handling"""
    model_path = "weights/best.pt"
    
    if not os.path.exists(model_path):
        st.error("❌ Model weights not found. Please add **best.pt** to the **weights/** folder.")
        st.info("📁 Expected path: `weights/best.pt`")
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

# Sidebar Configuration
st.sidebar.header("⚙️ Model Settings")
st.sidebar.markdown("---")

# File uploader
st.sidebar.subheader("📤 Upload Media")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video",
    type=["jpg", "jpeg", "png", "mp4"],
    help="Supported formats: JPG, PNG, MP4"
)

# Model parameters
st.sidebar.subheader("🎯 Detection Parameters")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for detections"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for NMS"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Upload media to start obstacle detection")

# Main content
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Image Processing
    if file_extension in ['jpg', 'jpeg', 'png']:
        st.subheader("📸 Image Analysis")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### Detected Objects")
            
            # Run inference with both thresholds
            with st.spinner("🔍 Running detection..."):
                results = model(image, conf=confidence, iou=iou_threshold)
                
                # Get annotated image
                annotated_image = results[0].plot()
                
                # Convert BGR to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
        
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
            st.warning("⚠️ No objects detected. Try adjusting the confidence or IOU threshold.")
    
    # Video Processing
    elif file_extension == 'mp4':
        st.subheader("🎥 Video Analysis")
        
        # Save uploaded video to temp folder
        os.makedirs("temp", exist_ok=True)
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
                results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)
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
                    frame_placeholder.image(frame_rgb, use_container_width=True)
            
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
                avg_detections = sum(detection_summary) / len(detection_summary) if detection_summary else 0
                st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")
            
            with col3:
                max_detections = max(detection_summary) if detection_summary else 0
                st.metric("Max Detections", max_detections)
            
            # Cleanup temporary input file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

else:
    # Welcome screen
    st.info("👈 Please upload an image or video from the sidebar to begin detection.")
    
    st.markdown("### 🎯 System Capabilities:")
    st.markdown("""
    - **Real-time Object Detection** using YOLOv11 architecture
    - **Image Analysis** with bounding boxes and confidence scores
    - **Video Processing** with frame-by-frame detection
    - **Adjustable Parameters** for detection sensitivity and accuracy
    - **Comprehensive Statistics** for all detections
    - **Optimized for Bangladesh** road conditions and traffic patterns
    """)
    
    st.markdown("### 🚀 How to Use:")
    st.markdown("""
    1. Upload an image (JPG, PNG) or video (MP4) using the sidebar
    2. Adjust the **Confidence Threshold** to control detection sensitivity
    3. Adjust the **IOU Threshold** for overlapping object handling
    4. For videos, click **"Run Detection"** to process all frames
    5. View detection results and statistics below
    """)
    
    st.markdown("### 📋 Detected Object Classes:")
    if model:
        st.write("Model can detect the following classes:")
        classes_list = ", ".join([f"`{name}`" for name in model.names.values()])
        st.markdown(classes_list)

# Contributors Section
st.markdown("---")
st.markdown('<h2 style="font-size: 1.8rem; font-weight: 700; margin: 1.5rem 0;">👥 Project Contributors</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="contributor-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎓 Md. Taher Bin Omar Hijbullah</h3>
            <p style="color: #6b7280;">
                📧 22303142@iubat.edu
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="contributor-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎓 Md. Rony Mia</h3>
            <p style="color: #6b7280;">
                📧 22303296@iubat.edu
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Md. Taher Bin Omar Hijbullah. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
