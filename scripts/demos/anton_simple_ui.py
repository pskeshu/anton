#!/usr/bin/env python3
"""
Anton Simple UI - Streamlit Web Interface

Simple interface: Load image → Click analyze → See results
Perfect for quick microscopy analysis with VLM and CMPO annotations.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import random

# Setup page
st.set_page_config(
    page_title="Anton Microscopy Analysis", 
    page_icon="🔬", 
    layout="wide"
)

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

# Import Anton components
try:
    from anton.core.pipeline import AnalysisPipeline
    from anton.utils.image_io import ImageLoader
    from anton.cmpo.mapping import map_to_cmpo
    from anton.cmpo.ontology import CMPOOntology
    anton_available = True
except ImportError as e:
    anton_available = False
    import_error = str(e)

# Header
st.title("🔬 Anton Microscopy Analysis")
st.markdown("**Simple Interface**: Load image → Click analyze → See VLM analysis + CMPO phenotypes")

# Check if Anton is available
if not anton_available:
    st.error(f"❌ Anton not available: {import_error}")
    st.stop()

# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# API Key check
api_status = []
vlm_provider = "mock"

if os.getenv('GOOGLE_API_KEY'):
    api_status.append("✅ Google API Key")
    vlm_provider = "gemini"
elif os.getenv('ANTHROPIC_API_KEY'):
    api_status.append("✅ Anthropic API Key")
    vlm_provider = "claude"
else:
    api_status.append("⚠️ No API key - using mock")

for status in api_status:
    st.sidebar.write(status)

st.sidebar.write(f"**VLM Provider**: {vlm_provider.upper()}")

# Dataset selection
st.sidebar.subheader("📁 Image Source")

# Check for BBBC013 dataset
bbbc013_path = Path("data/bbbc013/BBBC013_v1_images_bmp")
dataset_available = bbbc013_path.exists()

if dataset_available:
    # Get BBBC013 images
    bbbc013_images = sorted(list(bbbc013_path.glob("Channel1-*.BMP")))
    st.sidebar.success(f"✅ BBBC013 Dataset ({len(bbbc013_images)} images)")
    
    use_dataset = st.sidebar.radio("Select Source:", ["BBBC013 Dataset", "Upload Image"])
    
    if use_dataset == "BBBC013 Dataset":
        # Image selector
        image_names = [img.name for img in bbbc013_images]
        selected_image = st.sidebar.selectbox("Choose Image:", image_names)
        
        if st.sidebar.button("🎲 Random Image"):
            selected_image = random.choice(image_names)
            st.rerun()
        
        image_path = bbbc013_path / selected_image
        uploaded_file = None
    else:
        image_path = None
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload your own microscopy image"
        )
else:
    st.sidebar.warning("⚠️ BBBC013 dataset not found")
    image_path = None
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload your own microscopy image"
    )

# Analysis controls
st.sidebar.subheader("🔬 Analysis")
analyze_btn = st.sidebar.button("🚀 Analyze Image", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1, 1])

# Left column: Image display
with col1:
    st.subheader("🖼️ Image")
    
    # Load and display image
    current_image = None
    image_to_analyze = None
    
    if image_path and image_path.exists():
        # BBBC013 image
        try:
            loader = ImageLoader()
            current_image = loader.load(str(image_path))
            image_to_analyze = str(image_path)
            
            # Image display controls
            col_img1, col_img2 = st.columns([3, 1])
            
            with col_img2:
                st.markdown("**🔍 Display Options:**")
                zoom_level = st.select_slider(
                    "Zoom Level",
                    options=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                    value=1.0,
                    help="Adjust image size"
                )
                
                crop_enabled = st.checkbox("Enable Crop View", help="Show a cropped region of the image")
                
                if crop_enabled:
                    st.markdown("**Crop Region:**")
                    crop_x = st.slider("X Start", 0, current_image.shape[1]-100, 0, step=10)
                    crop_y = st.slider("Y Start", 0, current_image.shape[0]-100, 0, step=10)
                    crop_size = st.slider("Crop Size", 100, min(current_image.shape), 200, step=10)
            
            with col_img1:
                # Apply zoom and crop
                display_image = current_image.copy()
                
                if crop_enabled:
                    # Crop the image
                    x_end = min(crop_x + crop_size, current_image.shape[1])
                    y_end = min(crop_y + crop_size, current_image.shape[0])
                    display_image = current_image[crop_y:y_end, crop_x:x_end]
                    caption_text = f"BBBC013: {image_path.name} (Cropped)"
                else:
                    caption_text = f"BBBC013: {image_path.name}"
                
                # Calculate display width based on zoom
                base_width = 400
                display_width = int(base_width * zoom_level)
                
                st.image(
                    display_image, 
                    caption=caption_text, 
                    width=display_width,
                    use_column_width=False
                )
            
            # Image info
            if crop_enabled:
                st.caption(f"Original Shape: {current_image.shape} | Crop Shape: {display_image.shape}")
                st.caption(f"Crop Region: [{crop_x}:{crop_x+crop_size}, {crop_y}:{crop_y+crop_size}]")
            else:
                st.caption(f"Shape: {current_image.shape} | Type: {current_image.dtype}")
                st.caption(f"Range: [{current_image.min():.0f}, {current_image.max():.0f}]")
            
            # Full resolution download
            if st.button("💾 Download Full Resolution"):
                # Convert to PIL Image for download
                if current_image.dtype != np.uint8:
                    # Normalize to 0-255 for 8-bit
                    normalized = ((current_image - current_image.min()) / 
                                (current_image.max() - current_image.min()) * 255).astype(np.uint8)
                else:
                    normalized = current_image
                
                pil_img = Image.fromarray(normalized)
                st.download_button(
                    label="📁 Download PNG",
                    data=pil_img.tobytes(),
                    file_name=f"{image_path.stem}_full_res.png",
                    mime="image/png"
                )
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
            
    elif uploaded_file:
        # Uploaded image
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = ImageLoader()
            current_image = loader.load(temp_path)
            image_to_analyze = temp_path
            
            # Image display controls (same as above)
            col_img1, col_img2 = st.columns([3, 1])
            
            with col_img2:
                st.markdown("**🔍 Display Options:**")
                zoom_level = st.select_slider(
                    "Zoom Level",
                    options=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                    value=1.0,
                    help="Adjust image size"
                )
                
                crop_enabled = st.checkbox("Enable Crop View", help="Show a cropped region of the image")
                
                if crop_enabled:
                    st.markdown("**Crop Region:**")
                    crop_x = st.slider("X Start", 0, current_image.shape[1]-100, 0, step=10, key="upload_crop_x")
                    crop_y = st.slider("Y Start", 0, current_image.shape[0]-100, 0, step=10, key="upload_crop_y")
                    crop_size = st.slider("Crop Size", 100, min(current_image.shape), 200, step=10, key="upload_crop_size")
            
            with col_img1:
                # Apply zoom and crop
                display_image = current_image.copy()
                
                if crop_enabled:
                    # Crop the image
                    x_end = min(crop_x + crop_size, current_image.shape[1])
                    y_end = min(crop_y + crop_size, current_image.shape[0])
                    display_image = current_image[crop_y:y_end, crop_x:x_end]
                    caption_text = f"Uploaded: {uploaded_file.name} (Cropped)"
                else:
                    caption_text = f"Uploaded: {uploaded_file.name}"
                
                # Calculate display width based on zoom
                base_width = 400
                display_width = int(base_width * zoom_level)
                
                st.image(
                    display_image, 
                    caption=caption_text, 
                    width=display_width,
                    use_column_width=False
                )
            
            # Image info
            if crop_enabled:
                st.caption(f"Original Shape: {current_image.shape} | Crop Shape: {display_image.shape}")
                st.caption(f"Crop Region: [{crop_x}:{crop_x+crop_size}, {crop_y}:{crop_y+crop_size}]")
            else:
                st.caption(f"Shape: {current_image.shape} | Type: {current_image.dtype}")
                st.caption(f"Range: [{current_image.min():.0f}, {current_image.max():.0f}]")
            
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    else:
        st.info("👆 Select or upload an image to analyze")

# Right column: Analysis results
with col2:
    st.subheader("🧠 Analysis Results")
    
    # Run analysis when button is clicked
    if analyze_btn and image_to_analyze:
        
        # Progress indicator
        progress_bar = st.progress(0, text="Starting analysis...")
        
        try:
            # Configure pipeline
            bbbc013_context = {
                "experiment_type": "protein_translocation",
                "cell_line": "U2OS_osteosarcoma", 
                "protein": "FKHR-GFP",
                "drugs": ["Wortmannin", "LY294002"],
                "readout": "nuclear_vs_cytoplasmic_localization",
                "channels": ["FKHR-GFP", "DNA_DRAQ"]
            }
            
            config = {
                "vlm_provider": vlm_provider, 
                "channels": [0],
                "biological_context": bbbc013_context
            }
            
            progress_bar.progress(20, text="Initializing pipeline...")
            
            # Initialize and run pipeline
            pipeline = AnalysisPipeline(config)
            
            progress_bar.progress(40, text="Running VLM analysis...")
            results = pipeline.run_pipeline_sync(image_to_analyze)
            
            progress_bar.progress(80, text="Processing CMPO mappings...")
            
            # Display results
            progress_bar.progress(100, text="Analysis complete!")
            
            # Clear progress bar
            progress_bar.empty()
            
            # Create tabs for results
            tab1, tab2 = st.tabs(["🧠 VLM Analysis", "🧬 CMPO Phenotypes"])
            
            with tab1:
                st.markdown("### VLM Analysis Results")
                
                # Display each stage
                stages = [
                    ("Stage 1: Global Scene", "stage_1_global"),
                    ("Stage 2: Object Detection", "stage_2_objects"), 
                    ("Stage 3: Feature Analysis", "stage_3_features"),
                    ("Stage 4: Population Insights", "stage_4_population")
                ]
                
                for stage_name, stage_key in stages:
                    if stage_key in results and results[stage_key]:
                        with st.expander(f"📋 {stage_name}", expanded=(stage_key == "stage_1_global")):
                            stage_data = results[stage_key]
                            
                            # Extract content for display
                            if 'description' in stage_data:
                                content = stage_data['description']
                            elif 'segmentation_guidance' in stage_data:
                                content = stage_data['segmentation_guidance']
                            elif 'population_summary' in stage_data:
                                content = stage_data['population_summary']
                            else:
                                content = f"Stage completed. Keys: {list(stage_data.keys())}"
                            
                            # Clean up content for display
                            if content.startswith('```'):
                                lines = content.split('\n')
                                content = '\n'.join([line for line in lines if not line.strip().startswith('```')])
                            
                            st.text_area(
                                f"{stage_name} Results:",
                                content[:2000] + "\n\n[Truncated for display...]" if len(content) > 2000 else content,
                                height=200,
                                key=f"text_{stage_key}"
                            )
            
            with tab2:
                st.markdown("### CMPO Phenotype Classifications")
                
                try:
                    cmpo_mapper = CMPOOntology()
                    all_cmpo_results = []
                    
                    # Test CMPO mapping on all stages
                    for stage_name, stage_key in stages:
                        if stage_key in results and results[stage_key]:
                            stage_data = results[stage_key]
                            
                            # Extract text for CMPO mapping
                            stage_text = ""
                            if 'description' in stage_data:
                                stage_text = stage_data['description']
                            elif 'segmentation_guidance' in stage_data:
                                stage_text = stage_data['segmentation_guidance']
                            elif 'population_summary' in stage_data:
                                stage_text = stage_data['population_summary']
                            
                            if stage_text and len(stage_text) > 50:
                                # Clean JSON formatting
                                if '```' in stage_text:
                                    lines = stage_text.split('\n')
                                    stage_text = '\n'.join([line for line in lines if not line.strip().startswith('```')])
                                
                                cmpo_results = map_to_cmpo(stage_text, cmpo_mapper, context='protein_localization')
                                
                                if cmpo_results:
                                    st.markdown(f"**{stage_name}:**")
                                    
                                    for i, mapping in enumerate(cmpo_results[:3]):  # Top 3 per stage
                                        term_name = mapping.get('term_name', 'Unknown')
                                        confidence = mapping.get('confidence', 0)
                                        cmpo_id = mapping.get('CMPO_ID', 'Unknown')
                                        
                                        # Color code by confidence
                                        if confidence >= 4.5:
                                            color = "green"
                                            icon = "🟢"
                                        elif confidence >= 3.5:
                                            color = "orange"
                                            icon = "🟡"
                                        else:
                                            color = "red"
                                            icon = "🟠"
                                        
                                        st.markdown(f"{icon} **{term_name}**")
                                        st.caption(f"Confidence: {confidence:.2f} | ID: {cmpo_id}")
                                    
                                    all_cmpo_results.extend(cmpo_results)
                                    st.markdown("---")
                    
                    # Summary
                    if all_cmpo_results:
                        st.markdown("### 📊 Summary")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Total CMPO Terms", len(all_cmpo_results))
                        with col_b:
                            unique_terms = len(set(m.get('CMPO_ID') for m in all_cmpo_results))
                            st.metric("Unique Terms", unique_terms)
                        
                        # Top terms overall
                        st.markdown("**🏆 Top Phenotypes Overall:**")
                        top_terms = sorted(all_cmpo_results, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
                        
                        for i, term in enumerate(top_terms, 1):
                            name = term.get('term_name', 'Unknown')
                            conf = term.get('confidence', 0)
                            st.write(f"{i}. **{name}** ({conf:.2f})")
                    
                    else:
                        st.warning("No CMPO phenotypes identified in the analysis.")
                        
                except Exception as e:
                    st.error(f"CMPO mapping failed: {e}")
            
            # Clean up temp file if it exists
            if uploaded_file and os.path.exists(f"temp_{uploaded_file.name}"):
                os.remove(f"temp_{uploaded_file.name}")
                
        except Exception as e:
            progress_bar.empty()
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    elif analyze_btn:
        st.warning("Please select or upload an image first!")
    
    else:
        st.info("👈 Click 'Analyze Image' to start analysis")

# Footer
st.markdown("---")
st.markdown("🔬 **Anton Framework** - Hybrid Microscopy Analysis with VLM + Computer Vision")