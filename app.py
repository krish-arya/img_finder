# streamlit_image_matcher.py (Enhanced with Deep Model Matching)

import streamlit as st
st.set_page_config(page_title="Model Image Matcher", layout="wide")

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

# ----------------------- Load Deep Model -----------------------
@st.cache_resource
def load_model():
    model = resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------- Deep Feature Extraction -----------------------
def extract_features(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = Image.open(img_path_or_pil).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).squeeze().numpy()
    return features

# ----------------------- Rename Function -----------------------
def rename_images(image_paths, base_name):
    renamed = []
    for i, img_path in enumerate(image_paths):
        new_name = f"{base_name}_{i+1}.jpg"
        new_path = os.path.join(os.path.dirname(img_path), new_name)
        os.rename(img_path, new_path)
        renamed.append((img_path, new_path))
    return renamed

# ----------------------- Streamlit UI -----------------------
st.title("🧠 Deep Learning-Based Model Image Matcher")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_query = st.file_uploader("Upload a query image", type=['jpg', 'jpeg', 'png'])
with col2:
    uploaded_refs = st.file_uploader("Upload reference images (multiple allowed)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

similarity_threshold = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.85, step=0.01)
top_n = st.slider("How many top matches to show?", 1, 30, 5)

if uploaded_query and uploaded_refs:
    st.markdown("### 🖼️ Uploaded Reference Images Preview:")
    image_cols = st.columns(min(5, len(uploaded_refs)))
    for i, img in enumerate(uploaded_refs):
        with image_cols[i % len(image_cols)]:
            st.image(img, caption=img.name, use_column_width=True)

    query_feat = extract_features(uploaded_query)

    ref_temp_paths = []
    similarities = []

    for ref_file in uploaded_refs:
        temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=ref_file.name[-4:])
        temp_ref.write(ref_file.read())
        temp_ref.close()
        ref_temp_paths.append(temp_ref.name)

        feat = extract_features(temp_ref.name)
        sim = cosine_similarity([query_feat], [feat])[0][0]
        similarities.append((temp_ref.name, sim))

    matched_images = [(path, score) for path, score in similarities if score >= similarity_threshold]
    matched_images = sorted(matched_images, key=lambda x: x[1], reverse=True)[:top_n]

    if matched_images:
        st.subheader("🎯 Top Matched Images (Deep Model):")
        match_cols = st.columns(min(5, len(matched_images)))
        for i, (path, score) in enumerate(matched_images):
            with match_cols[i % len(match_cols)]:
                st.image(str(path), caption=f"Similarity: {round(score, 4)}", use_column_width=True)
    else:
        st.info("❌ No strong matches found. Try adjusting the threshold or using clearer images.")

    with st.expander("📝 Rename matched images"):
        base_name = st.text_input("Enter base name for renaming matched images", "model")
        if st.button("Rename Images"):
            renamed_files = rename_images([str(p) for p, _ in matched_images], base_name)
            for old, new in renamed_files:
                st.write(f"✅ {os.path.basename(old)} renamed to {os.path.basename(new)}")

    for path in ref_temp_paths:
        os.remove(path)
else:
    st.warning("📂 Please upload both a query image and at least one reference image.")
