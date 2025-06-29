# streamlit_image_matcher_auto.py

import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import io

st.set_page_config(page_title="PixelMatch", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .query-preview {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .match-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .download-section {
        background: rgba(76, 175, 80, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .preview-button > button {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .results-header {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Deep Model
@st.cache_resource
def load_model():
    try:
        model = resnet50(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Deep Feature Extraction
def extract_features(image_file_or_path):
    try:
        if isinstance(image_file_or_path, str):
            img = Image.open(image_file_or_path).convert("RGB")
        else:
            img = Image.open(image_file_or_path).convert("RGB")
        
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor).squeeze().numpy()
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Create ZIP with Custom Names
def create_zip_from_matches(matched_images, ref_files_dict, custom_name="match"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, (file_name, score) in enumerate(matched_images):
            if file_name in ref_files_dict:
                file_data = ref_files_dict[file_name]
                extension = file_name.rsplit('.', 1)[1] if '.' in file_name else 'jpg'
                new_filename = f"{custom_name}_{i+1}.{extension}"
                zipf.writestr(new_filename, file_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Extract Images from ZIP folder
def extract_images_from_zip(zip_file):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            image_files = []
            file_data_dict = {}
            
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    filename = file_info.filename
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        try:
                            file_data = zip_ref.read(filename)
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = filename.split('/')[-1]
                            image_files.append(file_obj)
                            file_data_dict[file_obj.name] = file_data
                        except Exception as e:
                            st.warning(f"Could not read {filename}: {e}")
                            continue
            
            return image_files, file_data_dict
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
        return [], {}

# Main Application
def main():
    st.markdown('<h1 class="main-title">PixelMatch</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    # Query Image Section
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: white; text-align: center;">🎯 Query Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_query = st.file_uploader("Upload the image to find matches for", type=['jpg', 'jpeg', 'png'])
        
        # Always show query image if uploaded
        if uploaded_query:
            st.markdown("""
            <div class="query-preview">
                <h4 style="color: white;">Query Image Preview</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(uploaded_query, use_column_width=True, caption="Your Query Image")
    
    # Reference Images Section
    with col2:
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: white; text-align: center;">📚 Reference Images</h3>
        </div>
        """, unsafe_allow_html=True)
        
        upload_method = st.radio("Choose upload method:", ["Multiple Images", "ZIP Folder"])
        
        if upload_method == "Multiple Images":
            uploaded_refs = st.file_uploader("Upload reference images", 
                                            type=['jpg', 'jpeg', 'png'], 
                                            accept_multiple_files=True)
            ref_files_dict = {}
        else:
            uploaded_zip = st.file_uploader("Upload ZIP folder", type=['zip'])
            if uploaded_zip:
                uploaded_refs, ref_files_dict = extract_images_from_zip(uploaded_zip)
                if uploaded_refs:
                    st.success(f"✅ Extracted {len(uploaded_refs)} images")
                else:
                    uploaded_refs = []
            else:
                uploaded_refs = []
                ref_files_dict = {}
        
        # Preview reference images button
        if uploaded_refs:
            st.markdown('<div class="preview-button">', unsafe_allow_html=True)
            if st.button(f"👁️ Preview Reference Images ({len(uploaded_refs)} files)"):
                st.session_state.show_refs = not st.session_state.get('show_refs', False)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show reference images preview if button clicked
    if uploaded_refs and st.session_state.get('show_refs', False):
        st.markdown("---")
        st.markdown('<p style="color: white; text-align: center; font-size: 1.2rem;">📖 Reference Images Preview</p>', unsafe_allow_html=True)
        
        # Show in grid
        cols_per_row = 4
        num_images = len(uploaded_refs)
        
        for i in range(0, num_images, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_images:
                    with cols[j]:
                        ref_img = uploaded_refs[i + j]
                        img_name = ref_img.name if hasattr(ref_img, 'name') else f"Image {i+j+1}"
                        st.image(ref_img, caption=img_name, use_column_width=True)
    
    # Processing Section
    if uploaded_query and uploaded_refs:
        st.markdown("---")
        
        # Custom naming section
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: white; text-align: center;">🏷️ Set Download Name</h3>
        </div>
        """, unsafe_allow_html=True)
        
        custom_name = st.text_input("Enter base name for matched images:", 
                                   value="", 
                                   placeholder="e.g., model, person, object...")
        
        # Only show process button when name is entered
        if custom_name.strip():
            if st.button("🚀 Start Processing & Find Matches", type="primary"):
                st.session_state.processing = True
                st.session_state.custom_name = custom_name.strip()
        else:
            st.info("💡 Enter a name above to start processing")
        
        # Processing and Results
        if st.session_state.get('processing', False):
            with st.spinner("🔄 AI is analyzing images..."):
                # Extract query features
                query_features = extract_features(uploaded_query)
                
                if query_features is None:
                    st.error("Failed to extract features from query image")
                    st.session_state.processing = False
                    return
                
                # Process reference images
                similarities = []
                
                # Populate ref_files_dict for individual files
                if upload_method == "Multiple Images":
                    ref_files_dict = {}
                    for ref_file in uploaded_refs:
                        ref_file.seek(0)
                        ref_files_dict[ref_file.name] = ref_file.read()
                        ref_file.seek(0)
                
                # Calculate similarities
                progress_bar = st.progress(0)
                for i, ref_file in enumerate(uploaded_refs):
                    file_name = ref_file.name if hasattr(ref_file, 'name') else f"image_{i+1}.jpg"
                    ref_features = extract_features(ref_file)
                    
                    if ref_features is not None:
                        similarity = cosine_similarity([query_features], [ref_features])[0][0]
                        similarities.append((file_name, similarity))
                    
                    progress_bar.progress((i + 1) / len(uploaded_refs))
                
                progress_bar.empty()
                
                # Filter matches above 0.8 threshold
                matched_images = [(name, score) for name, score in similarities if score >= 0.8]
                matched_images = sorted(matched_images, key=lambda x: x[1], reverse=True)
                
                # Store results in session state
                st.session_state.matched_images = matched_images
                st.session_state.ref_files_dict = ref_files_dict
                st.session_state.processing = False
                st.session_state.results_ready = True
        
        # Show Results
        if st.session_state.get('results_ready', False):
            matched_images = st.session_state.get('matched_images', [])
            ref_files_dict = st.session_state.get('ref_files_dict', {})
            custom_name = st.session_state.get('custom_name', 'match')
            
            st.markdown("---")
            
            if matched_images:
                st.markdown(f'<p class="results-header">🎯 Found {len(matched_images)} High-Quality Matches (≥80% similarity)</p>', unsafe_allow_html=True)
                
                # Show matches in a beautiful grid
                cols_per_row = 3
                for i in range(0, len(matched_images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(matched_images):
                            file_name, score = matched_images[i + j]
                            with cols[j]:
                                st.markdown(f"""
                                <div class="match-card">
                                    <h4 style="color: white;">Match {i+j+1}</h4>
                                    <p style="color: #4CAF50; font-weight: bold;">Similarity: {score:.1%}</p>
                                    <p style="color: white; font-size: 0.9rem;">Will be: {custom_name}_{i+j+1}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Find and show the reference image
                                for ref_img in uploaded_refs:
                                    ref_name = ref_img.name if hasattr(ref_img, 'name') else f"image_{uploaded_refs.index(ref_img)+1}"
                                    if ref_name == file_name:
                                        st.image(ref_img, use_column_width=True)
                                        break
                
                # Download Section
                st.markdown("""
                <div class="download-section">
                    <h3 style="color: white;">📦 Download Your Matches</h3>
                    <p style="color: white;">All matched images will be renamed and packaged for you!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create and offer download
                zip_data = create_zip_from_matches(matched_images, ref_files_dict, custom_name)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label=f"📥 Download {len(matched_images)} matches as {custom_name}_*.zip",
                        data=zip_data,
                        file_name=f"{custom_name}_matches_{len(matched_images)}_images.zip",
                        mime="application/zip",
                        type="primary"
                    )
                
                st.success(f"✅ Ready to download {len(matched_images)} high-quality matches!")
                
            else:
                st.markdown('<p class="results-header">❌ No matches found above 80% similarity</p>', unsafe_allow_html=True)
                st.info("Try uploading different reference images or a different query image.")
    
    elif uploaded_query or uploaded_refs:
        st.info("📋 Upload both query and reference images to start matching")
    
    else:
        st.markdown("""
        <div style="text-align: center; color: white; padding: 3rem;">
            <h3>🚀 Welcome to AI Image Matching!</h3>
            <p style="font-size: 1.1rem; opacity: 0.8;">
                Upload a query image and reference images to find matches with 80%+ similarity using deep learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if model is None:
        st.error("Model failed to load. Please check PyTorch installation.")
    else:
        main()