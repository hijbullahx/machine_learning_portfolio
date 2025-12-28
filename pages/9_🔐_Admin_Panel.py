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

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication
if not st.session_state.authenticated:
    st.title("🔐 Admin Panel - Authentication Required")
    st.markdown("---")
    
    password = st.text_input("Enter Admin Password", type="password")
    
    if st.button("Login", type="primary"):
        if password == "admin123":
            st.session_state.authenticated = True
            st.success("✅ Authentication successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid password. Please try again.")
    
    st.info("💡 Default password: admin123")
    
else:
    # Admin Panel Content
    st.title("🔐 Admin Panel")
    st.markdown("### Manage Models, Projects, and System Settings")
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
                
                st.success(f"✅ Current model: **best.pt**")
                st.info(f"📊 Size: {model_size:.2f} MB | Last Updated: {model_modified.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("⚠️ No model file found!")
            
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
            st.markdown("### 📋 Instructions")
            st.markdown("""
            1. Train your YOLOv11 model
            2. Export as `.pt` format
            3. Upload using the form
            4. Click **Replace Current Model**
            5. Refresh the app to use new model
            """)
            
            st.markdown("### ⚠️ Important")
            st.warning("A backup of the old model will be created automatically.")
    
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
