#!/usr/bin/env python3
"""
Minimal Anton Streamlit App - Crash-Safe Version
"""

import streamlit as st
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import traceback
import gc  # Garbage collection

# Configure PIL to handle large images better
Image.MAX_IMAGE_PIXELS = None  # Remove PIL size limit

# Setup page
st.set_page_config(
    page_title="Anton Microscopy Analysis", 
    page_icon="🔬", 
    layout="wide"
)

# Header
st.title("🔬 Anton Microscopy Analysis")
st.markdown("**Simple Interface**: Upload image → See basic analysis")

# Sidebar
st.sidebar.header("🎛️ Controls")

# Check API status
api_status = []
if os.getenv('GOOGLE_API_KEY'):
    api_status.append("✅ Google API Key")
elif os.getenv('ANTHROPIC_API_KEY'):
    api_status.append("✅ Anthropic API Key")
else:
    api_status.append("⚠️ No API key - demo mode")

for status in api_status:
    st.sidebar.write(status)

# Simple file upload with unique key
st.sidebar.subheader("📁 Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Upload microscopy image",
    key="image_uploader"  # Add unique key
)

# Analysis button with unique key
analyze_btn = st.sidebar.button("🚀 Analyze", type="primary", key="analyze_button")

# Main content
col1, col2 = st.columns([1, 1])

# Left: Image display
with col1:
    st.subheader("🖼️ Image")
    
    if uploaded_file is not None:
        try:
            # Reset file pointer to beginning (important!)
            uploaded_file.seek(0)
            
            # Simple PIL loading - most reliable
            image = Image.open(uploaded_file)
            
            # Resize if too large (prevent memory issues)
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                st.info(f"📏 Image resized to {image.size} for display")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Store in session state to prevent reprocessing
            if 'current_image' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
                # Clear old image from memory
                if 'current_image' in st.session_state:
                    del st.session_state.current_image
                    gc.collect()
                
                st.session_state.current_image = image
                st.session_state.uploaded_filename = uploaded_file.name
            
            # Display image
            st.image(st.session_state.current_image, caption=f"Uploaded: {uploaded_file.name}", width=400)
            
            # Basic info
            st.caption(f"Size: {st.session_state.current_image.size} | Mode: {st.session_state.current_image.mode}")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
            # Don't show full traceback to users - just log it
            print(f"Image loading error: {traceback.format_exc()}")
    else:
        st.info("👆 Upload an image to start")

# Right: Analysis results
with col2:
    st.subheader("🧠 Analysis Results")
    
    if analyze_btn and uploaded_file is not None:
        try:
            # Simple mock analysis to test if basic functionality works
            st.success("✅ Analysis Started!")
            
            with st.spinner("Processing..."):
                # Mock processing
                import time
                time.sleep(2)
            
            # Mock results
            st.markdown("### 📊 Mock Analysis Results")
            
            st.write("**Stage 1: Global Analysis**")
            st.text_area("Description:", 
                "Mock analysis: This appears to be a microscopy image with cellular structures. "
                "The image shows good contrast and appears suitable for analysis.", 
                height=100)
            
            st.write("**Stage 2: Object Detection**")
            st.text_area("Objects:", 
                "Mock detection: Multiple cellular objects detected. "
                "Estimated cell count: 15-25 cells visible.", 
                height=100)
            
            st.success("✅ Mock analysis complete!")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.code(traceback.format_exc())
    
    elif analyze_btn:
        st.warning("Please upload an image first!")
    else:
        st.info("👈 Upload image and click Analyze")

# Footer
st.markdown("---")
st.markdown("🔬 **Anton Framework** - Minimal Demo Version")