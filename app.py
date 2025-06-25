# streamlit_image_matcher.py (Fixed: Progress bar + Visible UI)

import streamlit as st
st.set_page_config(page_title="Model Image Matcher", layout="wide", initial_sidebar_state="collapsed")

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

# ----------------------- Custom CSS for Beautiful UI -----------------------
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: rgba(255,255,255,0.8);
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Upload section styling */
    .upload-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .upload-card:hover {
        transform: translateY(-5px);
    }
    
    .upload-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .upload-subtitle {
        color: rgba(255,255,255,0.7);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* File uploader custom styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #4CAF50 !important;
        background: rgba(76, 175, 80, 0.1) !important;
    }
    
    /* Image containers */
    .image-preview {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .image-preview:hover {
        transform: scale(1.05);
    }
    
    .image-preview img {
        border-radius: 10px;
        width: 100%;
        height: auto;
    }
    
    /* Match result cards */
    .match-result {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .match-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .score-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        width: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.8);
        font-weight: 400;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50, #45a049) !important;
        border-radius: 10px !important;
    }
    
    /* Info/warning boxes */
    .stInfo {
        background: rgba(33, 150, 243, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(33, 150, 243, 0.3) !important;
        color: white !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        color: white !important;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Welcome card */
    .welcome-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    
    /* Loading animation */
    .loading-text {
        text-align: center;
        font-size: 1.3rem;
        color: white;
        margin: 2rem 0;
        font-weight: 500;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        border: none;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------- Load Deep Model -----------------------
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

# ----------------------- Deep Feature Extraction -----------------------
def extract_features(image_file_or_path):
    try:
        if isinstance(image_file_or_path, str):
            img = Image.open(image_file_or_path).convert("RGB")
        else:
            # Handle uploaded file
            img = Image.open(image_file_or_path).convert("RGB")
        
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor).squeeze().numpy()
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# ----------------------- Create ZIP of Matches -----------------------
def create_zip_from_matches(matched_images, ref_files_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, (file_name, score) in enumerate(matched_images):
            if file_name in ref_files_dict:
                file_data = ref_files_dict[file_name]
                # Create a meaningful filename
                base_name = file_name.rsplit('.', 1)[0]
                extension = file_name.rsplit('.', 1)[1] if '.' in file_name else 'jpg'
                new_filename = f"match_{i+1}_{base_name}_score_{score:.4f}.{extension}"
                zipf.writestr(new_filename, file_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ----------------------- Main UI -----------------------
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-title">🧠 AI Image Matcher</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find similar images using deep learning and computer vision</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ **Settings**")
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.80, 0.01)
        show_scores = st.checkbox("Show Similarity Scores", True)
        max_matches = st.number_input("Max Matches to Show", 1, 50, 10)
        
        st.markdown("---")
        st.markdown("### 📖 **How it works**")
        st.markdown("""
        1. Upload a query image
        2. Upload reference images  
        3. AI analyzes features
        4. Get similarity matches
        """)
    
    # Upload sections
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="upload-card">
            <h3 class="upload-title">🎯 Query Image</h3>
            <p class="upload-subtitle">Upload the image you want to find matches for</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_query = st.file_uploader("Choose query image", type=['jpg', 'jpeg', 'png'], key="query")
    
    with col2:
        st.markdown("""
        <div class="upload-card">
            <h3 class="upload-title">📚 Reference Images</h3>
            <p class="upload-subtitle">Upload multiple images to search through</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_refs = st.file_uploader("Choose reference images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="refs")
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    if uploaded_query and uploaded_refs:
        # Show uploaded images preview
        st.markdown("### 🖼️ **Image Preview**")
        
        # Query image preview
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Query Image**")
            with st.container():
                st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                st.image(uploaded_query, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Reference Images ({len(uploaded_refs)} files)**")
            # Create a grid for reference images
            cols = st.columns(min(3, len(uploaded_refs)))
            for i, ref_img in enumerate(uploaded_refs):
                with cols[i % len(cols)]:
                    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                    st.image(ref_img, caption=ref_img.name, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        # Processing with progress bar
        st.markdown('<div class="loading-text">🔄 Processing images with AI...</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract query features
            status_text.text("Extracting features from query image...")
            progress_bar.progress(0.1)  # 10%
            query_features = extract_features(uploaded_query)
            
            if query_features is None:
                st.error("Failed to extract features from query image")
                return
            
            # Process reference images
            similarities = []
            ref_files_dict = {}  # Store file data for ZIP creation
            total_refs = len(uploaded_refs)
            
            for i, ref_file in enumerate(uploaded_refs):
                status_text.text(f"Processing reference image {i+1}/{total_refs}: {ref_file.name}")
                # Fixed progress calculation - ensure it stays between 0.1 and 0.9
                progress_value = 0.1 + (0.8 * (i+1) / total_refs)
                progress_bar.progress(min(progress_value, 0.9))
                
                # Store file data
                ref_file.seek(0)  # Reset file pointer
                ref_files_dict[ref_file.name] = ref_file.read()
                ref_file.seek(0)  # Reset again for feature extraction
                
                # Extract features
                ref_features = extract_features(ref_file)
                
                if ref_features is not None:
                    # Calculate similarity
                    similarity = cosine_similarity([query_features], [ref_features])[0][0]
                    similarities.append((ref_file.name, similarity))
            
            status_text.text("Analyzing matches...")
            progress_bar.progress(0.95)
            
            # Filter and sort matches
            matched_images = [(name, score) for name, score in similarities if score >= similarity_threshold]
            matched_images = sorted(matched_images, key=lambda x: x[1], reverse=True)
            matched_images = matched_images[:max_matches]  # Limit results
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            if matched_images:
                # Display results
                st.markdown(f"### 🎯 **Found {len(matched_images)} Matching Images**")
                
                # Create metrics row
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Total Matches</div>
                    </div>
                    """.format(len(matched_images)), unsafe_allow_html=True)
                
                with metric_cols[1]:
                    best_score = max(matched_images, key=lambda x: x[1])[1]
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">{:.3f}</div>
                        <div class="metric-label">Best Match</div>
                    </div>
                    """.format(best_score), unsafe_allow_html=True)
                
                with metric_cols[2]:
                    avg_score = sum(score for _, score in matched_images) / len(matched_images)
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">{:.3f}</div>
                        <div class="metric-label">Average Score</div>
                    </div>
                    """.format(avg_score), unsafe_allow_html=True)
                
                with metric_cols[3]:
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Threshold</div>
                    </div>
                    """.format(similarity_threshold), unsafe_allow_html=True)
                
                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                
                # Display each match
                for i, (file_name, score) in enumerate(matched_images):
                    st.markdown('<div class="match-result">', unsafe_allow_html=True)
                    
                    # Score badge and match info
                    st.markdown('<div class="match-header">', unsafe_allow_html=True)
                    if show_scores:
                        st.markdown(f'<div class="score-badge">Match {i+1} • Similarity: {score:.4f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="score-badge">Match {i+1}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Images side by side
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("**Query Image**")
                        st.image(uploaded_query, use_column_width=True)
                    
                    with img_col2:
                        st.markdown(f"**Matched: {file_name}**")
                        # Find the corresponding reference image
                        for ref_img in uploaded_refs:
                            if ref_img.name == file_name:
                                st.image(ref_img, use_column_width=True)
                                break
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download section
                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                st.markdown("### 📦 **Download Results**")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Create ZIP file
                    zip_data = create_zip_from_matches(matched_images, ref_files_dict)
                    st.download_button(
                        label="📥 Download All Matches as ZIP",
                        data=zip_data,
                        file_name=f"image_matches_{len(matched_images)}_images.zip",
                        mime="application/zip"
                    )
                
                with col2:
                    # Additional info
                    st.info(f"💡 {len(matched_images)} images ready for download with similarity scores in filenames!")
            
            else:
                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                st.warning(f"❌ No matches found above the similarity threshold of {similarity_threshold:.2f}")
                
                # Show all similarities for debugging
                if similarities:
                    with st.expander("🔍 View All Similarity Scores"):
                        all_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                        for name, score in all_similarities:
                            st.write(f"**{name}**: {score:.4f}")
                        
                        if all_similarities:
                            best_score = all_similarities[0][1]
                            st.info(f"💡 Try lowering the threshold to {best_score:.2f} to see the best match!")
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Welcome message when no files are uploaded
        st.markdown("""
        <div class="welcome-card">
            <h3 style="color: white; margin-bottom: 1.5rem;">🚀 Welcome to AI Image Matching!</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; line-height: 1.6;">
                Upload a query image and reference images to find the best matches using deep learning.<br><br>
                <strong>✨ Features:</strong><br>
                • ResNet-50 deep learning model<br>
                • Cosine similarity matching<br>
                • Batch processing & ZIP download<br>
                • Adjustable similarity thresholds
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if model is None:
        st.error("⚠️ Model failed to load. Please check your PyTorch installation.")
    else:
        main()