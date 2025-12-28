import streamlit as st
import json
import os
import shutil
from datetime import datetime
import sys

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import hf_manager

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
    
    # Load projects
    projects_file = "projects.json"
    if os.path.exists(projects_file):
        with open(projects_file, 'r') as f:
            projects_data = json.load(f)
    else:
        projects_data = {"projects": []}
    
    # Project selector
    st.subheader("📂 Select Project")
    if projects_data["projects"]:
        project_names = [p["name"] for p in projects_data["projects"]]
        selected_project = st.selectbox(
            "Choose a project to manage:",
            project_names,
            help="Select a project to manage its model, settings, and information"
        )
        
        # Get selected project details
        project_index = project_names.index(selected_project)
        current_project = projects_data["projects"][project_index]
        
        st.markdown(f"""
            <div class="admin-card">
                <h3>{current_project['name']}</h3>
                <p><strong>Status:</strong> {current_project['status']}</p>
                <p>{current_project['description']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for different admin functions for this project
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Model Management", "📝 Project Settings", "⚙️ System Info", "📤 Upload Project"])
    else:
        st.warning("⚠️ No projects found. Please add a project first.")
        st.info("💡 Use the 'Upload Project' tab to add your first project.")
        
        # Show all tabs even if no projects
        tab1, tab2, tab3, tab4 = st.tabs(["📦 Model Management", "📝 Project Settings", "⚙️ System Info", "📤 Upload Project"])
    
    # TAB 1: Model Management
    with tab1:
        if projects_data["projects"]:
            st.header(f"📦 Model Management - {current_project['name']}")
            st.markdown("Upload a new trained model for this project to Hugging Face.")
            
            # Get the model filename for this project
            model_filename = hf_manager.get_model_filename_for_project(current_project['name'])
            
            st.info(f"📄 Model file for this project: **{model_filename}**")
            
            # Display current model info
            model_info = hf_manager.get_model_info(model_filename)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Current Model Status")
                if model_info['exists']:
                    st.success(f"✅ Model exists locally")
                    st.metric("File Size", f"{model_info['size_mb']} MB")
                    st.metric("Location", "weights/" + model_filename)
                else:
                    st.warning("⚠️ Model not found locally")
                    st.info("Will download from Hugging Face when needed")
                
                # List available models in HF repo
                st.markdown("### 🗂️ Available Models in HF Repo")
                with st.spinner("Loading..."):
                    available_models = hf_manager.list_available_models()
                    if available_models:
                        for model in available_models:
                            if model == model_filename:
                                st.success(f"✓ {model} (current)")
                            else:
                                st.info(f"○ {model}")
                    else:
                        st.warning("No models found in repository")
            
            with col2:
                st.markdown("### 📤 Upload New Model")
                uploaded_model = st.file_uploader(
                    "Choose a model file",
                    type=["pt", "pth", "onnx"],
                    help=f"Upload a new model - will be saved as {model_filename}",
                    key="model_uploader"
                )
                
                if uploaded_model is not None:
                    st.write(f"**File:** {uploaded_model.name}")
                    st.write(f"**Size:** {uploaded_model.size / (1024 * 1024):.2f} MB")
                    st.write(f"**Will be saved as:** {model_filename}")
                    
                    if st.button("🚀 Upload to Hugging Face", type="primary"):
                        try:
                            # Save temporarily
                            os.makedirs("weights", exist_ok=True)
                            temp_path = f"weights/temp_{model_filename}"
                            
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_model.read())
                            
                            # Upload to Hugging Face
                            with st.spinner("Uploading to Hugging Face..."):
                                success = hf_manager.upload_model(temp_path, model_filename)
                            
                            if success:
                                # Rename to final name
                                final_path = f"weights/{model_filename}"
                                if os.path.exists(final_path):
                                    # Backup old model
                                    backup_path = f"weights/{model_filename}.backup"
                                    shutil.move(final_path, backup_path)
                                
                                shutil.move(temp_path, final_path)
                                
                                st.success("✅ Model uploaded to Hugging Face successfully!")
                                st.balloons()
                                st.info("🔄 Please reload the project page to use the new model.")
                            else:
                                os.remove(temp_path)
                                st.error("❌ Failed to upload model to Hugging Face")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
        else:
            st.header("📦 Model Management")
            st.info("Please add a project first to manage models.")
    
    # TAB 2: Project Settings
    with tab2:
        if projects_data["projects"]:
            st.header(f"📝 Project Settings - {current_project['name']}")
            
            # Edit current project
            st.subheader("✏️ Edit Current Project")
            with st.form("edit_project_form"):
                edit_name = st.text_input("Project Name", value=current_project['name'])
                edit_description = st.text_area("Description", value=current_project['description'], height=100)
                edit_status = st.selectbox("Status", ["Active", "In Development", "Completed", "Archived"], 
                                          index=["Active", "In Development", "Completed", "Archived"].index(current_project['status']))
                edit_github = st.text_input("GitHub Link", value=current_project.get('github', ''))
                edit_page = st.text_input("Page Path", value=current_project.get('page', ''))
                
                col1, col2 = st.columns(2)
                with col1:
                    update_submitted = st.form_submit_button("💾 Update Project", type="primary", use_container_width=True)
                with col2:
                    delete_submitted = st.form_submit_button("🗑️ Delete Project", use_container_width=True)
                
                if update_submitted:
                    projects_data["projects"][project_index] = {
                        "name": edit_name,
                        "description": edit_description,
                        "status": edit_status,
                        "github": edit_github,
                        "page": edit_page
                    }
                    with open(projects_file, 'w') as f:
                        json.dump(projects_data, f, indent=4)
                    st.success("✅ Project updated successfully!")
                    st.rerun()
                
                if delete_submitted:
                    projects_data["projects"].pop(project_index)
                    with open(projects_file, 'w') as f:
                        json.dump(projects_data, f, indent=4)
                    st.success("🗑️ Project deleted!")
                    st.rerun()
            
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
            
            submitted = st.form_submit_button("✅ Add Project", type="primary", use_container_width=True)
            
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
        if projects_data["projects"]:
            st.header(f"⚙️ System Information - {current_project['name']}")
        else:
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
    
    # TAB 4: Upload Project
    with tab4:
        st.header("📤 Upload Complete Project")
        st.markdown("Add a new project with all its details and files.")
        
        st.markdown("""
            <div class="admin-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <h3 style="color: white;">📋 Project Upload Guide</h3>
                <p>Upload a complete project including model files, documentation, and metadata.</p>
                <ul style="text-align: left; line-height: 1.8;">
                    <li>✓ Fill in project details below</li>
                    <li>✓ Upload trained model file (.pt)</li>
                    <li>✓ Optionally upload demo images/videos</li>
                    <li>✓ Project will appear on the home page</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Project upload form
        with st.form("upload_project_form"):
            st.subheader("📝 Project Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                upload_project_name = st.text_input("Project Name*", placeholder="e.g., 🎯 Traffic Sign Detection")
                upload_project_status = st.selectbox("Status", ["Active", "In Development", "Completed", "Archived"])
                upload_project_github = st.text_input("GitHub Repository", placeholder="https://github.com/username/repo")
            
            with col2:
                upload_project_page = st.text_input("Page Path", placeholder="e.g., 2_🎯_Traffic_Signs")
                upload_project_icon = st.text_input("Project Icon (emoji)", placeholder="🎯", max_chars=2)
                upload_project_tags = st.text_input("Tags (comma-separated)", placeholder="computer-vision, deep-learning")
            
            upload_project_description = st.text_area(
                "Project Description*",
                placeholder="Detailed description of your project, its goals, and achievements...",
                height=150
            )
            
            st.markdown("---")
            st.subheader("📦 Model & Files")
            
            col1, col2 = st.columns(2)
            
            with col1:
                upload_model_file = st.file_uploader(
                    "Upload Model File (.pt)*",
                    type=["pt"],
                    help="Upload your trained model file"
                )
            
            with col2:
                upload_demo_files = st.file_uploader(
                    "Upload Demo Files (optional)",
                    type=["jpg", "jpeg", "png", "mp4"],
                    accept_multiple_files=True,
                    help="Upload sample images or videos for demonstration"
                )
            
            st.markdown("---")
            
            # Submit button
            upload_submitted = st.form_submit_button("🚀 Upload Project", type="primary", use_container_width=True)
            
            if upload_submitted:
                if upload_project_name and upload_project_description and upload_model_file:
                    try:
                        # Create project folder structure
                        project_folder = f"projects/{upload_project_name.replace(' ', '_').replace('🎯', '').replace('🚗', '').strip()}"
                        os.makedirs(f"{project_folder}/models", exist_ok=True)
                        os.makedirs(f"{project_folder}/demos", exist_ok=True)
                        
                        # Save model file
                        model_filename = f"{project_folder}/models/{upload_model_file.name}"
                        with open(model_filename, "wb") as f:
                            f.write(upload_model_file.read())
                        
                        # Save demo files
                        demo_files_saved = []
                        if upload_demo_files:
                            for demo_file in upload_demo_files:
                                demo_filename = f"{project_folder}/demos/{demo_file.name}"
                                with open(demo_filename, "wb") as f:
                                    f.write(demo_file.read())
                                demo_files_saved.append(demo_filename)
                        
                        # Add to projects.json
                        new_project = {
                            "name": f"{upload_project_icon} {upload_project_name}" if upload_project_icon else upload_project_name,
                            "description": upload_project_description,
                            "status": upload_project_status,
                            "github": upload_project_github if upload_project_github else "",
                            "page": upload_project_page if upload_project_page else "",
                            "model_path": model_filename,
                            "demo_files": demo_files_saved,
                            "tags": [tag.strip() for tag in upload_project_tags.split(",")] if upload_project_tags else []
                        }
                        
                        projects_data["projects"].append(new_project)
                        
                        with open(projects_file, 'w') as f:
                            json.dump(projects_data, f, indent=4)
                        
                        st.success("✅ Project uploaded successfully!")
                        st.balloons()
                        st.info(f"📁 Project saved to: {project_folder}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error uploading project: {str(e)}")
                else:
                    st.error("❌ Please fill in required fields: Project Name, Description, and Model File.")

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Md. Taher Bin Omar Hijbullah. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
