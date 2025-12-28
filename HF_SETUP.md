# Hugging Face Setup Instructions

## 1. Create Hugging Face Repository

1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up or log in
3. Click "New" → "Model"
4. Name it: `ml_portfolio`
5. Set to Public or Private
6. Click "Create model"

## 2. Get Your Access Token

1. Go to Settings → Access Tokens
2. Click "New token"
3. Name: "Portfolio Upload"
4. Type: Write
5. Copy the token

## 3. Update secrets.toml

Edit `.streamlit/secrets.toml`:
```toml
[default]
hf_repo_id = "YourUsername/ml_portfolio"
hf_token = "hf_xxxxxxxxxxxxxxxxxxxxx"
```

## 4. Upload Initial Models

### Option A: Using Hugging Face CLI (Recommended)

```bash
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Upload your model
huggingface-cli upload Hijbullah/ml_portfolio weights/vehicle_best.pt vehicle_best.pt
```

### Option B: Using Python Script

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="weights/best.pt",
    path_in_repo="vehicle_best.pt",
    repo_id="Hijbullah/ml_portfolio",
    token="your_token_here"
)
```

### Option C: Web Interface

1. Go to your repo: `https://huggingface.co/Hijbullah/ml_portfolio`
2. Click "Files and versions"
3. Click "Add file" → "Upload file"
4. Upload your model as `vehicle_best.pt`

## 5. Model Naming Convention

Each project has a specific model filename:
- Autonomous Vehicle → `vehicle_best.pt`
- Breast Cancer → `cancer_best.pt`
- Traffic Signs → `traffic_sign_best.pt`
- Sentiment Analysis → `sentiment_best.pt`

## 6. Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New app"
4. Select your repo
5. Add secrets in "Advanced settings":
   ```toml
   [default]
   hf_repo_id = "Hijbullah/ml_portfolio"
   hf_token = "hf_xxxxxxxxxxxxxxxxxxxxx"
   ```
6. Click "Deploy"

## Architecture Overview

```
Hugging Face Repo (Hijbullah/ml_portfolio)
├── vehicle_best.pt          # Autonomous Vehicle model
├── cancer_best.pt           # Breast Cancer model
├── traffic_sign_best.pt     # Traffic Sign model
└── sentiment_best.pt        # Sentiment Analysis model

Your App
├── utils/hf_manager.py      # Downloads models on demand
├── pages/1_🚗_Autonomous_Vehicle.py  # Uses vehicle_best.pt
├── pages/2_🏥_Breast_Cancer.py       # Uses cancer_best.pt
└── pages/9_🔐_Admin_Panel.py         # Uploads new models
```

## How It Works

1. **On Page Load**: App checks if model exists locally
2. **If Not**: Downloads from Hugging Face automatically
3. **First Time**: User downloads model (cached for future)
4. **Admin Upload**: New model uploaded to HF, replaces old one
5. **Next Visit**: New model downloaded automatically

## Benefits

✅ No Git LFS needed
✅ Models not in GitHub repo (stays small)
✅ Easy model updates via Admin Panel
✅ Automatic downloads when needed
✅ Support multiple projects easily
✅ Free hosting on HF (up to 300GB)
