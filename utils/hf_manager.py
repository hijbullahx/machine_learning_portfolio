"""
Hugging Face Model Manager
Universal downloader and uploader for all project models
"""

import os
import streamlit as st
from huggingface_hub import hf_hub_download, upload_file, list_repo_files
from pathlib import Path


def get_hf_credentials():
    """Get Hugging Face credentials from secrets"""
    try:
        hf_repo_id = st.secrets["hf_repo_id"]
        hf_token = st.secrets["hf_token"]
        return hf_repo_id, hf_token
    except Exception as e:
        st.error(f"❌ Error loading HF credentials: {str(e)}")
        return None, None


def download_model(filename, local_dir="weights"):
    """
    Download a specific model file from Hugging Face repo
    
    Args:
        filename: Name of the model file (e.g., 'vehicle_best.pt', 'cancer_best.pt')
        local_dir: Local directory to save the model
    
    Returns:
        str: Path to the downloaded model file, or None if failed
    """
    hf_repo_id, hf_token = get_hf_credentials()
    
    if not hf_repo_id or not hf_token:
        return None
    
    local_path = os.path.join(local_dir, filename)
    
    # Check if file already exists locally
    if os.path.exists(local_path):
        print(f"✅ Model already exists locally: {local_path}")
        return local_path
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download from Hugging Face
        print(f"📥 Downloading {filename} from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=filename,
            token=hf_token,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        st.error(f"❌ Failed to download {filename}: {str(e)}")
        return None


def upload_model(file_path, filename):
    """
    Upload a model file to Hugging Face repo
    
    Args:
        file_path: Local path to the file to upload
        filename: Name to save the file as in HF repo (e.g., 'vehicle_best.pt')
    
    Returns:
        bool: True if successful, False otherwise
    """
    hf_repo_id, hf_token = get_hf_credentials()
    
    if not hf_repo_id or not hf_token:
        return False
    
    try:
        print(f"📤 Uploading {filename} to Hugging Face...")
        
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=hf_repo_id,
            token=hf_token,
            commit_message=f"Update {filename}"
        )
        
        print(f"✅ Model uploaded successfully: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {str(e)}")
        st.error(f"❌ Failed to upload {filename}: {str(e)}")
        return False


def list_available_models():
    """
    List all model files available in the Hugging Face repo
    
    Returns:
        list: List of model filenames
    """
    hf_repo_id, hf_token = get_hf_credentials()
    
    if not hf_repo_id or not hf_token:
        return []
    
    try:
        files = list_repo_files(repo_id=hf_repo_id, token=hf_token)
        # Filter for model files (.pt, .pth, .onnx, etc.)
        model_files = [f for f in files if f.endswith(('.pt', '.pth', '.onnx', '.h5', '.pkl'))]
        return model_files
        
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return []


def get_model_info(filename):
    """
    Get information about a specific model file
    
    Args:
        filename: Name of the model file
    
    Returns:
        dict: Model information (size, last modified, etc.)
    """
    local_path = os.path.join("weights", filename)
    
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        return {
            "exists": True,
            "path": local_path,
            "size_mb": round(size_mb, 2)
        }
    else:
        return {
            "exists": False,
            "path": None,
            "size_mb": 0
        }


# Project-specific model filenames mapping
PROJECT_MODELS = {
    "🚗 Autonomous Vehicle Perception": "vehicle_best.pt",
    "🏥 Breast Cancer Detection": "cancer_best.pt",
    "🎯 Traffic Sign Recognition": "traffic_sign_best.pt",
    "😊 Sentiment Analysis": "sentiment_best.pt"
}


def get_model_filename_for_project(project_name):
    """
    Get the model filename for a specific project
    
    Args:
        project_name: Name of the project
    
    Returns:
        str: Model filename for this project
    """
    return PROJECT_MODELS.get(project_name, "default_model.pt")
