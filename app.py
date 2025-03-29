import streamlit as st
import cv2
import numpy as np
from segmentation_utils import load_image, load_video, apply_mask, resize_with_aspect_ratio

# Streamlit app title
st.title("Image and Video Segmentation App")

# Upload image or video
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    # Check if the uploaded file is an image or video
    if uploaded_file.type.startswith('image'):
        # Load and display the image
        image = load_image(uploaded_file)

        st.image(image, caption='Uploaded Image', use_container_width=True)


        # HSV sliders for segmentation
        st.sidebar.header("HSV Segmentation Parameters")
        lower_h = st.sidebar.slider("Lower Hue", 0, 179, 0)
        lower_s = st.sidebar.slider("Lower Saturation", 0, 255, 50)
        lower_v = st.sidebar.slider("Lower Value", 0, 255, 50)
        upper_h = st.sidebar.slider("Upper Hue", 0, 179, 179)
        upper_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
        upper_v = st.sidebar.slider("Upper Value", 0, 255, 255)

        # Apply mask
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask, result = apply_mask(image, lower_bound, upper_bound)

        # Display results
        st.image(mask, caption='Mask', use_container_width=True)
        st.image(result, caption='Segmented Result', use_container_width=True)


        # Save results
        if st.button("Save Results"):
            mask_filename = "mask.png"
            result_filename = "result.png"
            cv2.imwrite(mask_filename, mask)
            cv2.imwrite(result_filename, result)
            st.success(f"Results saved as {mask_filename} and {result_filename}")

    elif uploaded_file.type.startswith('video'):
        # Load and display the video
        video = load_video(uploaded_file.name)
        st.video(uploaded_file, use_container_width=True)


        # Similar HSV sliders for video segmentation
        st.sidebar.header("HSV Segmentation Parameters")
        lower_h = st.sidebar.slider("Lower Hue", 0, 179, 0)
        lower_s = st.sidebar.slider("Lower Saturation", 0, 255, 50)
        lower_v = st.sidebar.slider("Lower Value", 0, 255, 50)
        upper_h = st.sidebar.slider("Upper Hue", 0, 179, 179)
        upper_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
        upper_v = st.sidebar.slider("Upper Value", 0, 255, 255)

        # Process video frame by frame (simplified for demonstration)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            lower_bound = np.array([lower_h, lower_s, lower_v])
            upper_bound = np.array([upper_h, upper_s, upper_v])
            mask, result = apply_mask(frame, lower_bound, upper_bound)

            # Display results (this part can be improved for real-time display)
            st.image(result, caption='Segmented Video Frame', use_column_width=True)

