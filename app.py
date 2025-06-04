import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="Color Detection & Sorting", layout="centered")

def resize_image(image, max_size=600):
    height, width = image.shape[:2]
    scaling_factor = max_size / float(max(height, width))
    if scaling_factor < 1:
        image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
    return image

def extract_colors(image, num_colors=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    sorted_idx = np.argsort(-counts)
    sorted_colors = colors[sorted_idx]
    return sorted_colors

def display_palette(colors):
    st.subheader("Dominant Colors")
    col_blocks = []
    for color in colors:
        hex_code = "#{:02x}{:02x}{:02x}".format(*color)
        col_blocks.append(
            f"<div style='background-color:{hex_code}; width:60px; height:60px; display:inline-block; margin:5px; border-radius:5px;'></div><br><center>{hex_code}</center>"
        )
    st.markdown("".join(col_blocks), unsafe_allow_html=True)

st.title("🎨 Color Detection and Sorting App")

option = st.radio("Choose Input Source:", ("Upload Image", "Use Webcam"))

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
elif option == "Use Webcam":
    capture = st.camera_input("Take a picture")
    if capture:
        image = np.array(Image.open(capture))

if image is not None:
    st.image(image, caption="Original Image", use_column_width=True)
    resized = resize_image(image)
    colors = extract_colors(resized, num_colors=5)
    display_palette(colors)
