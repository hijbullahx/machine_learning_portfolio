import streamlit as st
import json
import os

# Set page configuration
st.set_page_config(
    page_title="ML Portfolio - Md. Taher Bin Omar Hijbullah",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .project-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        background-color: #f9f9f9;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #666;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 Machine Learning Portfolio</h1>
        <h3>Md. Taher Bin Omar Hijbullah</h3>
        <p>Computer Vision | Deep Learning | AI Research</p>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("## 👋 Welcome to My ML Portfolio")
st.markdown("""
I am **Md. Taher Bin Omar Hijbullah**, a passionate Machine Learning Engineer and AI Researcher 
specializing in Computer Vision and Deep Learning applications. This portfolio showcases my research 
projects and implementations in autonomous systems, object detection, and intelligent perception systems.

### 🎯 Research Focus:
- **Autonomous Vehicle Perception** for complex traffic scenarios
- **Real-time Object Detection** using state-of-the-art YOLO architectures
- **Deep Learning** applications for Bangladesh's unique road conditions
- **AI-powered solutions** for emerging market challenges
""")

st.markdown("---")

# Load projects from JSON
projects_file = "projects.json"
if os.path.exists(projects_file):
    with open(projects_file, 'r') as f:
        projects_data = json.load(f)
else:
    # Default project data
    projects_data = {
        "projects": [
            {
                "name": "🚗 Autonomous Vehicle Perception",
                "description": "Real-time obstacle detection system designed for Bangladesh's dense traffic conditions using YOLOv11. Detects vehicles, pedestrians, and obstacles with high accuracy.",
                "status": "Active",
                "github": "https://github.com/yourusername/BD_Autonomous_YOLOv11",
                "page": "1_🚗_Autonomous_Vehicle"
            }
        ]
    }
    # Save default data
    with open(projects_file, 'w') as f:
        json.dump(projects_data, f, indent=4)

# Display Projects
st.markdown("## 🚀 Active Projects")
st.markdown("Explore my current research and development projects:")
st.markdown("")

# Create project grid
projects = projects_data.get("projects", [])
if projects:
    # Display 2 projects per row
    for i in range(0, len(projects), 2):
        cols = st.columns(2)
        
        for idx, col in enumerate(cols):
            if i + idx < len(projects):
                project = projects[i + idx]
                
                with col:
                    st.markdown(f"""
                        <div class="project-card">
                            <h3>{project['name']}</h3>
                            <p><strong>Status:</strong> <span style="color: #10b981;">●</span> {project['status']}</p>
                            <p>{project['description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if project.get('page'):
                            st.info(f"👉 Navigate to **{project['name']}** page from the sidebar")
                    with col2:
                        if project.get('github'):
                            st.markdown(f"[🔗 GitHub]({project['github']})")
                    
                    st.markdown("")
else:
    st.info("No projects available yet. Check the Admin Panel to add projects.")

st.markdown("---")

# Skills and Technologies
st.markdown("## 🛠️ Technologies & Frameworks")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### Deep Learning")
    st.markdown("""
    - PyTorch
    - TensorFlow
    - Ultralytics YOLO
    - OpenCV
    """)

with col2:
    st.markdown("### Computer Vision")
    st.markdown("""
    - Object Detection
    - Image Segmentation
    - Video Analysis
    - Real-time Processing
    """)

with col3:
    st.markdown("### Development")
    st.markdown("""
    - Python
    - Streamlit
    - Git/GitHub
    - Docker
    """)

with col4:
    st.markdown("### Research Areas")
    st.markdown("""
    - Autonomous Vehicles
    - Traffic Analysis
    - Edge AI
    - Model Optimization
    """)

st.markdown("---")

# Contact Information
st.markdown("## 📫 Get in Touch")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📧 Email")
    st.markdown("contact@example.com")

with col2:
    st.markdown("### 💼 LinkedIn")
    st.markdown("[Connect on LinkedIn](#)")

with col3:
    st.markdown("### 🐙 GitHub")
    st.markdown("[View GitHub Profile](#)")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Md. Taher Bin Omar Hijbullah. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom to prevent content from being hidden by fixed footer
st.markdown("<br><br>", unsafe_allow_html=True)
