# ML Portfolio - Auto-Discovery Project System

## 📁 Project Structure

```
BD_Autonomous_YOLOv11_web_app-1/
├── Home.py                          # Main landing page (auto-discovers projects)
├── requirements.txt                 # Python dependencies
├── packages.txt                     # System packages
├── temp/                            # Temporary files
└── pages/                           # Projects directory
    ├── 1_🚗_Bangladesh_Traffic.py  # Wrapper page for autonomous_vehicle
    └── autonomous_vehicle/          # Example ML Project
        ├── config.py                # Project metadata (auto-loaded by Home.py)
        ├── main.py                  # Actual Streamlit application
        ├── README.md                # Project documentation
        └── weights/                 # Model weights
            └── best.pt              # Trained model
```

## 🚀 How It Works

### Auto-Discovery System
The home page **automatically discovers** all projects in the `pages/` directory:
- Scans each subdirectory in `pages/`
- Looks for `config.py` file
- Loads project metadata dynamically
- Displays project cards on home page

### Adding New Projects

1. **Create project folder:**
   ```
   pages/your_new_project/
   ```

2. **Add config.py:**
   ```python
   PROJECT_CONFIG = {
       "id": "your_project_id",
       "title": "Your Project Title",
       "status": "Active",  # or "Completed", "In Progress"
       "icon": "🎯",
       "page_name": "2_🎯_Your_Project.py",  # Wrapper page filename
       "description": """Your project description with HTML formatting""",
       "technologies": ["Tech1", "Tech2", "Tech3"],
       "github_link": "https://github.com/...",
       "model_path": "pages/your_new_project/weights/model.pt"
   }
   ```

3. **Create main.py:**
   ```python
   import streamlit as st
   import sys
   import os
   sys.path.append(os.path.dirname(__file__))
   from config import PROJECT_CONFIG
   
   st.set_page_config(
       page_title=PROJECT_CONFIG['title'],
       page_icon=PROJECT_CONFIG['icon']
   )
   
   # Your project code here...
   ```

4. **Create wrapper page in pages/ directory:**
   ```python
   # pages/2_🎯_Your_Project.py
   import streamlit as st
   import sys
   import os
   
   project_dir = os.path.join(os.path.dirname(__file__), "your_new_project")
   sys.path.insert(0, project_dir)
   
   main_file = os.path.join(project_dir, "main.py")
   with open(main_file) as f:
       code = f.read()
       exec(code)
   ```

5. **Add your model/weights:**
   ```
   pages/your_new_project/weights/
   ```

6. **Done!** The home page will automatically display your project.

## ✏️ Updating Project Information

To update a project's information (title, description, status, etc.):

1. Navigate to the project's `config.py`:
   ```
   pages/autonomous_vehicle/config.py
   ```

2. Edit the `PROJECT_CONFIG` dictionary:
   ```python
   PROJECT_CONFIG = {
       "title": "New Title Here",      # Update this
       "description": "New desc...",    # Update this
       "status": "Completed",           # Update this
       # ...
   }
   ```

3. Save the file

4. Refresh browser → Changes appear automatically on both:
   - Home page project card
   - Project page itself

## 🎯 Benefits

✅ **Zero Duplication** - Each project's info stored once in its own `config.py`
✅ **Auto-Discovery** - Add a folder with `config.py` and it appears on home page
✅ **Isolated Projects** - Each project is self-contained with its own files
✅ **Easy Maintenance** - Update one file, everything syncs
✅ **Scalable** - Add unlimited projects without modifying Home.py

## 📊 Current Projects

### 1. Bangladesh Traffic Perception System
- **Location**: `pages/autonomous_vehicle/`
- **Config**: `pages/autonomous_vehicle/config.py`
- **Model**: `pages/autonomous_vehicle/weights/best.pt`
- **Tech**: YOLOv11 + CBAM Attention

## 🔧 Running the Application

```bash
# Activate virtual environment
source venv/Scripts/activate  # On Windows Git Bash
# or
venv\Scripts\activate         # On Windows CMD

# Run Streamlit
streamlit run Home.py
```

## 📝 Project Config Reference

```python
PROJECT_CONFIG = {
    "id": str,              # Unique identifier (required)
    "title": str,           # Project name (required)
    "status": str,          # "Active", "Completed", "In Progress"
    "icon": str,            # Emoji icon
    "page_name": str,       # Wrapper page filename (e.g., "1_🚗_Project.py")
    "description": str,     # HTML-formatted description
    "technologies": list,   # List of technologies used
    "github_link": str,     # GitHub repository URL
    "model_path": str,      # Path to model weights (optional)
    # Add custom fields as needed
}
```

## 🎨 Customization

Each project can have:
- Custom styling in `main.py`
- Own dependencies (document in project README)
- Multiple Python files (import from same directory)
- Assets folder (images, data, etc.)

## 🚨 Important Notes

- Each project folder **must** contain `config.py` with `PROJECT_CONFIG`
- Each project folder **must** contain `main.py` (the actual Streamlit app)
- Each project **must** have a wrapper page in `pages/` directory (e.g., `1_🚗_Project.py`)
- The wrapper page executes the `main.py` from the project subfolder
- Model weights go in `pages/project_name/weights/`
- Don't put project files in root directory
- Home.py automatically finds all projects - no manual registration needed

## 📚 Example: Complete Project Structure

```
pages/
├── 1_🚗_Bangladesh_Traffic.py      # Wrapper page (Streamlit requirement)
└── autonomous_vehicle/              # Actual project folder
    ├── config.py                    # Project metadata
    ├── main.py                      # Main Streamlit application
    ├── README.md                    # Project docs
    ├── utils.py                     # Helper functions (optional)
    ├── weights/                     # Model weights
    │   └── best.pt
    └── assets/                      # Images, data (optional)
        └── sample_image.jpg
```

## 🎓 For Future You

When you add new ML projects:
1. Create folder in `pages/project_name/`
2. Add `config.py` and `main.py` in that folder
3. Create wrapper page in `pages/` (e.g., `2_🎯_Project.py`)
4. That's it! Home page updates automatically.

No need to:
- ❌ Modify Home.py
- ❌ Register projects manually
- ❌ Duplicate project info
- ❌ Worry about syncing

## 🔍 Why This Structure?

**Streamlit Limitation**: Streamlit only recognizes pages directly in the `pages/` directory, not in subdirectories.

**Our Solution**:
- ✅ Project files live in organized subfolders: `pages/project_name/`
- ✅ Lightweight wrapper pages in `pages/`: `1_🚗_Project.py`
- ✅ Wrapper loads and executes the actual `main.py` from subfolder
- ✅ Best of both worlds: organization + Streamlit compatibility
