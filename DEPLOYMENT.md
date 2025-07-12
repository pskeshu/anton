# Anton Streamlit App Deployment Guide

## 🚀 Streamlit Cloud Deployment

Your Anton microscopy analysis app is ready for deployment on Streamlit Cloud!

### Prerequisites
1. GitHub repository with your code
2. Streamlit Cloud account (free at https://share.streamlit.io/)
3. API keys for VLM providers (Google/Anthropic)

### Step-by-Step Deployment

#### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare Anton app for Streamlit Cloud deployment"
git push origin main
```

#### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub repository
4. Set these deployment settings:
   - **Repository**: `your-username/anton`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose your custom URL

#### 3. Configure Secrets
In your Streamlit Cloud app dashboard:
1. Go to "Settings" → "Secrets"
2. Add your API keys:
```toml
GOOGLE_API_KEY = "your-google-gemini-api-key"
ANTHROPIC_API_KEY = "your-anthropic-claude-api-key"
```

#### 4. Deploy!
Click "Deploy" - your app will be live in a few minutes!

### 📁 Files Created for Deployment

- **streamlit_app.py**: Main entry point for Streamlit Cloud
- **.streamlit/config.toml**: App configuration (upload limits, theme)
- **.streamlit/secrets.toml**: Template for secrets (don't put real keys here)
- **packages.txt**: System dependencies for OpenCV
- **requirements.txt**: Python dependencies (already existed)

### 🔧 Local Testing

Test your deployment setup locally:
```bash
streamlit run streamlit_app.py
```

### 🔑 API Key Setup

The app supports both Google Gemini and Anthropic Claude:
- **Google Gemini**: Get key from https://makersuite.google.com/app/apikey
- **Anthropic Claude**: Get key from https://console.anthropic.com/

The app will use mock responses if no API keys are provided.

### 🖼️ Image Sources

The app can work with:
1. **BBBC013 Dataset** (if you include it in your repo)
2. **User uploads** (supports PNG, JPG, TIFF, BMP)
3. **Mock analysis** (works without real images)

### ⚡ Performance Notes

For best performance on Streamlit Cloud:
- Large image datasets should be stored externally
- Consider using smaller sample images in the repo
- Mock mode provides instant results for demos

### 🛠️ Troubleshooting

**Common Issues:**
- Import errors: Check requirements.txt includes all dependencies
- OpenCV issues: packages.txt should resolve system dependencies
- Memory limits: Streamlit Cloud has resource limits for free tier

**Debug Mode:**
Add this to your app for debugging:
```python
import streamlit as st
st.write("Debug info:", st.experimental_get_query_params())
```

### 🌐 Example Deployment

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

Users can:
- Upload microscopy images
- Run VLM analysis (with API keys)
- View CMPO phenotype classifications
- Download results

---

🔬 **Your Anton app is ready for the cloud!**