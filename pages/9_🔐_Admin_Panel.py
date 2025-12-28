import streamlit as st
import json
import os
import shutil
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Admin Panel",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .login-container {
        max-width: 450px;
        margin: 5rem auto;
        padding: 3rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        text-align: center;
    }
    
    .admin-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .admin-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .admin-card:hover {
        border-color: #667eea;
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.2);
    }
    
    .success-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
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
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stat-card h3 {
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication
if not st.session_state.authenticated:
    st.markdown("""
        <div class="login-container">
            <h1 style="font-size: 3rem; margin-bottom: 0;">🔐</h1>
            <h2 class="admin-header">Admin Panel</h2>
            <p style="color: #6b7280; margin-bottom: 2rem;">Enter your credentials to continue</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Password", type="password", placeholder="Enter admin password")
        
        if st.button("🔓 Login", type="primary", use_container_width=True):
            if password == "hijbullah23":
                st.session_state.authenticated = True
                st.success("✅ Authentication successful! Redirecting...")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Invalid password. Please try again.")
    
else:
    # Admin Panel Content
    st.markdown('<h1 class="admin-header">🔐 Admin Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="text-align: center; color: #6b7280;">Manage Models, Projects, and System Settings</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["📦 Model Management", "📝 Project Management", "⚙️ System Info"])
    
    # TAB 1: Model Management
    with tab1:
        st.header("📦 Update YOLO Model")
        st.markdown("Upload a new trained model to replace the existing one.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Current model info
            model_path = "weights/best.pt"
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                model_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                st.markdown(f"""
                    <div class="admin-card">
                        <h3 style="color: #10b981; margin-bottom: 1rem;">✅ Model Status: Active</h3>
                        <p><strong>File:</strong> best.pt</p>
                        <p><strong>Size:</strong> {model_size:.2f} MB</p>
                        <p><strong>Last Updated:</strong> {model_modified.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="admin-card">
                        <h3 style="color: #ef4444;">⚠️ No Model Found</h3>
                        <p>Please upload a model file below</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Upload new model
            st.subheader("Upload New Model")
            uploaded_model = st.file_uploader(
                "Choose a .pt model file",
                type=["pt"],
                help="Upload your trained YOLOv11 model file"
            )
            
            if uploaded_model is not None:
                st.write(f"**Uploaded:** {uploaded_model.name}")
                st.write(f"**Size:** {uploaded_model.size / (1024 * 1024):.2f} MB")
                
                if st.button("🔄 Replace Current Model", type="primary"):
                    try:
                        # Create backup of old model
                        if os.path.exists(model_path):
                            backup_path = f"weights/best_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                            shutil.copy(model_path, backup_path)
                            st.info(f"📦 Backup created: {backup_path}")
                        
                        # Save new model
                        os.makedirs("weights", exist_ok=True)
                        with open(model_path, "wb") as f:
                            f.write(uploaded_model.read())
                        
                        st.success("✅ Model updated successfully! Please reload the Autonomous Vehicle page.")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error updating model: {str(e)}")
        
        with col2:
            st.markdown("""
                <div class="admin-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h3 style="color: white;">📋 Instructions</h3>
                    <ol style="text-align: left; padding-left: 1.5rem;">
                        <li>Train your YOLOv11 model</li>
                        <li>Export as .pt format</li>
                        <li>Upload using the form</li>
                        <li>Click Replace Current Model</li>
                        <li>Refresh the app to use new model</li>
                    </ol>
                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px;">
                        <strong>⚠️ Important:</strong><br>
                        A backup will be created automatically
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # TAB 2: Project Management
    with tab2:
        st.header("📝 Manage Portfolio Projects")
        
        # Load existing projects
        projects_file = "projects.json"
        if os.path.exists(projects_file):
            with open(projects_file, 'r') as f:
                projects_data = json.load(f)
        else:
            projects_data = {"projects": []}
        
        # Display existing projects
        st.subheader("📚 Current Projects")
        if projects_data["projects"]:
            for idx, project in enumerate(projects_data["projects"]):
                with st.expander(f"📌 {project['name']}"):
                    st.write(f"**Description:** {project['description']}")
                    st.write(f"**Status:** {project['status']}")
                    st.write(f"**GitHub:** {project.get('github', 'N/A')}")
                    st.write(f"**Page:** {project.get('page', 'N/A')}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{idx}"):
                        projects_data["projects"].pop(idx)
                        with open(projects_file, 'w') as f:
                            json.dump(projects_data, f, indent=4)
                        st.success("Project deleted!")
                        st.rerun()
        else:
            st.info("No projects yet. Add one below!")
        
        st.markdown("---")
        
        # Add new project
        st.subheader("➕ Add New Project")
        
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name", placeholder="e.g., 🎯 Sentiment Analysis System")
            project_description = st.text_area(
                "Description",
                placeholder="Brief description of the project...",
                height=100
            )
            project_status = st.selectbox("Status", ["Active", "In Development", "Completed", "Archived"])
            project_github = st.text_input("GitHub Link (optional)", placeholder="https://github.com/...")
            project_page = st.text_input("Page Path (optional)", placeholder="e.g., 2_🎯_Sentiment_Analysis")
            
            submitted = st.form_submit_button("✅ Add Project", type="primary")
            
            if submitted:
                if project_name and project_description:
                    new_project = {
                        "name": project_name,
                        "description": project_description,
                        "status": project_status,
                        "github": project_github if project_github else "",
                        "page": project_page if project_page else ""
                    }
                    
                    projects_data["projects"].append(new_project)
                    
                    with open(projects_file, 'w') as f:
                        json.dump(projects_data, f, indent=4)
                    
                    st.success("✅ Project added successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Please fill in at least Project Name and Description.")
    
    # TAB 3: System Info
    with tab3:
        st.header("⚙️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Directory Structure")
            directories = {
                "weights/": "Model files",
                "temp/": "Temporary uploads",
                "pages/": "Streamlit pages"
            }
            
            for dir_path, description in directories.items():
                exists = os.path.exists(dir_path)
                status = "✅" if exists else "❌"
                st.write(f"{status} **{dir_path}** - {description}")
        
        with col2:
            st.subheader("📊 File Statistics")
            
            # Count files in weights
            if os.path.exists("weights"):
                weight_files = len([f for f in os.listdir("weights") if f.endswith('.pt')])
                st.metric("Model Files", weight_files)
            
            # Count temp files
            if os.path.exists("temp"):
                temp_files = len(os.listdir("temp"))
                st.metric("Temporary Files", temp_files)
            
            # Count projects
            st.metric("Active Projects", len(projects_data.get("projects", [])))
        
        st.markdown("---")
        
        # Maintenance actions
        st.subheader("🧹 Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Temp Folder"):
                try:
                    if os.path.exists("temp"):
                        for file in os.listdir("temp"):
                            os.remove(os.path.join("temp", file))
                        st.success("✅ Temp folder cleared!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("📦 List Model Backups"):
                if os.path.exists("weights"):
                    backups = [f for f in os.listdir("weights") if "backup" in f]
                    if backups:
                        st.write("**Backups found:**")
                        for backup in backups:
                            st.write(f"- {backup}")
                    else:
                        st.info("No backups found.")
        
        with col3:
            st.info("More features coming soon!")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Md. Taher Bin Omar Hijbullah. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
