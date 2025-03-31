import streamlit as st
import cv2
import numpy as np
from segmentation_utils import load_image, load_video, apply_mask, resize_with_aspect_ratio

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #2E7D32; font-size: 4em; margin-bottom: 20px; transition: color 0.3s, transform 0.3s; text-transform: uppercase;'>Image And Video Segmentation App</h1>", unsafe_allow_html=True)

# Load custom CSS
st.markdown("""
<style>
/* Custom CSS for Streamlit App */

/* Title Style */
h1 {
    color: #2E7D32; /* Darker green color for better contrast */
    text-align: center;
    font-size: 4em; /* Increased font size for better visibility */
    margin-bottom: 20px; /* Space below title */
    transition: color 0.3s, transform 0.3s; /* Transition effects */
}

h1:hover {
    color: #66BB6A; /* Change color on hover */
    transform: scale(1.05); /* Slightly enlarge title on hover */
}

/* Body Background */
body {
    background-color: #F0F4F7; /* Light grey background for a softer look */
}

/* Button Style */
.stButton {
    background-color: #4CAF50; /* Green background for buttons */
    color: white; /* White text color */
    border: none; /* Remove border */
    padding: 12px 24px; /* Add padding */
    text-align: center; /* Center text */
    text-decoration: none; /* Remove underline */
    display: inline-block; /* Display as inline-block */
    font-size: 18px; /* Increase font size */
    margin: 10px 2px; /* Add margin */
    cursor: pointer; /* Pointer cursor on hover */
    transition: background-color 0.3s, transform 0.3s; /* Transition effects */
    border-radius: 5px; /* Rounded corners */
}

.stButton:hover {
    background-color: #66BB6A; /* Lighter green on hover */
    transform: scale(1.05); /* Slightly enlarge button on hover */
}

/* Image Style */
.stImage {
    border: 2px solid #4CAF50; /* Green border for images */
    border-radius: 10px; /* Rounded corners */
    max-width: 100%; /* Responsive image */
    height: auto; /* Maintain aspect ratio */
}

/* Slider Style */
.stSlider {
    margin: 20px 0; /* Add margin for sliders */
}

/* Tooltip Style */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position above the tooltip */
    left: 50%;
    margin-left: -60px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Media Queries for Responsiveness */
@media (max-width: 600px) {
    h1 {
        font-size: 2.5em; /* Smaller title on small screens */
    }
    .stButton {
        font-size: 16px; /* Smaller button text on small screens */
        padding: 10px 20px; /* Adjust padding */
    }
}
</style>
""", unsafe_allow_html=True)

# Upload image or video
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'], label_visibility="collapsed")

if uploaded_file is not None:
    # Check if the uploaded file is an image or video
    if uploaded_file.type.startswith('image'):
        # Load and display the image
        image = load_image(uploaded_file)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True, clamp=True)
        st.markdown("<p style='text-align: center;'>This is the uploaded image.</p>", unsafe_allow_html=True)

        # HSV sliders for segmentation
        st.sidebar.header("HSV Segmentation Parameters")
        st.sidebar.markdown("Adjust the sliders below to set the HSV segmentation parameters.", unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            lower_h = st.slider("Lower Hue", 0, 179, 0, help="Set the lower bound for hue.")
            lower_s = st.slider("Lower Saturation", 0, 255, 50, help="Set the lower bound for saturation.")
            lower_v = st.slider("Lower Value", 0, 255, 50, help="Set the lower bound for value.")
        with col2:
            upper_h = st.slider("Upper Hue", 0, 179, 179, help="Set the upper bound for hue.")
            upper_s = st.slider("Upper Saturation", 0, 255, 255, help="Set the upper bound for saturation.")
            upper_v = st.slider("Upper Value", 0, 255, 255, help="Set the upper bound for value.")

        # Apply mask
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask, result = apply_mask(image, lower_bound, upper_bound)

        # Display results
        st.image(mask, caption='Mask', use_container_width=True)
        st.image(result, caption='Segmented Result', use_container_width=True)

        # Save results
        if st.button("Save Results", key="save_results", help="Click to save the segmented results."):
            st.success("Results have been saved successfully!")
            mask_filename = "mask.png"
            result_filename = "result.png"
            cv2.imwrite(mask_filename, mask)
            cv2.imwrite(result_filename, result)
            st.success(f"Results saved as {mask_filename} and {result_filename}")

    elif uploaded_file.type.startswith('video'):
        # Load and display the video
        video = load_video(uploaded_file)  # Use the uploaded file object directly

        st.video(uploaded_file)

        # HSV sliders for video segmentation
        st.sidebar.header("HSV Segmentation Parameters")
        st.sidebar.markdown("Adjust the sliders below to set the HSV segmentation parameters.", unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            lower_h = st.slider("Lower Hue", 0, 179, 0, help="Set the lower bound for hue.")
            lower_s = st.slider("Lower Saturation", 0, 255, 50, help="Set the lower bound for saturation.")
            lower_v = st.slider("Lower Value", 0, 255, 50, help="Set the lower bound for value.")
        with col2:
            upper_h = st.slider("Upper Hue", 0, 179, 179, help="Set the upper bound for hue.")
            upper_s = st.slider("Upper Saturation", 0, 255, 255, help="Set the upper bound for saturation.")
            upper_v = st.slider("Upper Value", 0, 255, 255, help="Set the upper bound for value.")

        # Process video frame by frame (simplified for demonstration)
        if 'frame_index' not in st.session_state:
            st.session_state.frame_index = 0

        ret, frame = video.read()
        if ret:
            lower_bound = np.array([lower_h, lower_s, lower_v])
            upper_bound = np.array([upper_h, upper_s, upper_v])
            mask, result = apply_mask(frame, lower_bound, upper_bound)

            # Display results
            st.image(mask, caption='Mask Frame', use_container_width=True)  # Display mask frame
            st.image(result, caption='Segmented Video Frame', use_container_width=True)  # Display segmented result
