import streamlit as st
import json
import os

# Set page configuration
st.set_page_config(
    page_title="ML Portfolio - Md. Taher Bin Omar Hijbullah",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header h3 {
        font-size: 1.8rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
        opacity: 0.95;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .intro-section {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #1a1a1a;
    }
    
    .intro-section h3 {
        color: #1a1a1a;
        margin-top: 1rem;
    }
    
    .intro-section ul {
        color: #2d3748;
    }
    
    .intro-section strong {
        color: #000000;
    }
    
    .project-card {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid transparent;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .project-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .project-card:hover::before {
        transform: scaleX(1);
    }
    
    .project-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .project-card h3 {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    .tech-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .tech-card:hover {
        transform: translateY(-5px);
    }
    
    .tech-card h3 {
        color: white !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .contact-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .contact-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 2rem 0 1.5rem 0;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
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
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
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
st.markdown('<h2 class="section-title">👋 Welcome to My ML Portfolio</h2>', unsafe_allow_html=True)
st.markdown("""
<<div class="intro-section">
  <p>
    I am <strong>Md. Taher Bin Omar Hijbullah</strong>, a Computer Science student and aspiring 
    <strong>Machine Learning enthusiast</strong> currently learning and exploring the fundamentals 
    of ML, Deep Learning, and AI-driven systems. This portfolio reflects my learning journey, practice 
    projects, and experiments as I build a strong foundation in intelligent systems and data-driven 
    problem solving.
  </p>

  <h3>🎯 Current Learning Focus:</h3>
  <ul>
    <li><strong>Machine Learning fundamentals</strong> using Python and common ML libraries</li>
    <li><strong>Deep Learning concepts</strong> and neural network basics</li>
    <li><strong>Computer Vision fundamentals</strong> through small experiments and practice projects</li>
    <li><strong>Applied AI</strong> for solving practical, real-world problems</li>
  </ul>

  <p>
    I am actively learning through coursework, self-study, and hands-on projects, with the goal of 
    gradually advancing toward more complex AI and ML applications in the future.
  </p>
</div>

""", unsafe_allow_html=True)

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
st.markdown('<h2 class="section-title">🚀 Active Research Projects</h2>', unsafe_allow_html=True)
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
                            <div class="status-badge">{project['status']}</div>
                            <p style="color: #4a5568; line-height: 1.6;">{project['description']}</p>
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
st.markdown('<h2 class="section-title">🛠️ Technologies & Frameworks</h2>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="tech-card">
        <h3>🧠 Deep Learning</h3>
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
    <div class="tech-card">
        <h3>👁️ Computer Vision</h3>
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
    <div class="tech-card">
        <h3>💻 Development</h3>
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
    <div class="tech-card">
        <h3>🔬 Research Areas</h3>
        <ul style="list-style: none; padding: 0;">
            <li>✓ Autonomous Vehicles</li>
            <li>✓ Traffic Analysis</li>
            <li>✓ Edge AI</li>
            <li>✓ Model Optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Contact Information
st.markdown('<h2 class="section-title">📫 Get in Touch</h2>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="https://www.linkedin.com/in/hijbullah/" target="_blank" style="margin: 0 15px; text-decoration: none;">
            <img src="https://img.icons8.com/color/96/000000/linkedin.png" alt="LinkedIn" style="width: 60px; height: 60px; transition: transform 0.3s; cursor: pointer;" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">
        </a>
        <a href="https://github.com/hijbullahx" target="_blank" style="margin: 0 15px; text-decoration: none;">
            <img src="https://img.icons8.com/glyph-neue/96/000000/github.png" alt="GitHub" style="width: 60px; height: 60px; transition: transform 0.3s; cursor: pointer;" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">
        </a>
        <a href="https://www.facebook.com/h6781/" target="_blank" style="margin: 0 15px; text-decoration: none;">
            <img src="https://img.icons8.com/color/96/000000/facebook-new.png" alt="Facebook" style="width: 60px; height: 60px; transition: transform 0.3s; cursor: pointer;" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">
        </a>
    </div>
    <p style="text-align: center; color: #6b7280; margin-top: 1rem;">
        📧 hijbullah119445@gmail.com | 22303142@iubat.edu
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Md. Taher Bin Omar Hijbullah. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom to prevent content from being hidden by fixed footer
st.markdown("<br><br>", unsafe_allow_html=True)
