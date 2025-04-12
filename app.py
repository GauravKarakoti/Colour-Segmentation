import streamlit as st
import cv2
import numpy as np
from PIL import Image  # For image validation
import mimetypes  # For video MIME type validation
from segmentation_utils import load_image, load_video, apply_mask, resize_with_aspect_ratio
import tempfile  # Add this import for creating temporary files
import os  # For file operations

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #2E7D32; font-size: 4em; margin-bottom: 20px; transition: color 0.3s, transform 0.3s; text-transform: uppercase;'>Image & Video Segmentation with Color Palette</h1>", unsafe_allow_html=True)

nav_option = st.sidebar.radio("Navigation", ["Home", "Upload", "Color Palette", "Segmentation"])

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

def is_valid_image(file):
    """Validate if the uploaded file is a valid image."""
    try:
        img = Image.open(file)
        img.verify()  # Verify that it is an image
        return True
    except Exception:
        return False

def is_valid_video(file):
    """Validate if the uploaded file is a valid video."""
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type and mime_type.startswith('video')

if nav_option == "Home":
    st.markdown("<h2 style='text-align:center;'>Welcome to the Segmentation & Color Palette App 🎨</h2>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Use the sidebar to navigate different sections.</p>",unsafe_allow_html=True)
elif nav_option == "Upload":
    st.write("## Upload Your Image or Video")
    uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
    
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file  # Save the uploaded file in session state
        st.success("File uploaded successfully!")

elif nav_option == "Color Palette":
    st.write("## Adjust Your Color Palette 🎨")
    st.sidebar.header("HSV Segmentation")
    l_h = st.sidebar.slider("Lower Hue", 0, 179, 0)
    l_s = st.sidebar.slider("Lower Saturation", 0, 255, 50)
    l_v = st.sidebar.slider("Lower Value", 0, 255, 50)
    u_h = st.sidebar.slider("Upper Hue", 0, 179, 179)
    u_s = st.sidebar.slider("Upper Saturation", 0, 255, 255)
    u_v = st.sidebar.slider("Upper Value", 0, 255, 255)

    # RGB Palette
    st.sidebar.header("RGB Color Picker")
    r = st.sidebar.slider("Red", 0, 255, 0)
    g = st.sidebar.slider("Green", 0, 255, 0)
    b = st.sidebar.slider("Blue", 0, 255, 0)

    # Display Colors
    if l_h > u_h:  # Handle Hue wrap-around
        hue_range = list(range(l_h, 180)) + list(range(0, u_h + 1))
        mid_hue = (l_h + (u_h + 180)) // 2 % 180  # Correct midpoint calculation for wrap-around
    else:
        mid_hue = (l_h + u_h) // 2

    # Blend lower and upper HSV values
    blended_s = (l_s + u_s) // 2
    blended_v = (l_v + u_v) // 2
    
    # Create an HSV image and convert it to BGR
    hsv_color = np.full((300, 900, 3), (mid_hue, blended_s, blended_v), dtype=np.uint8)

    rgb_palette = np.zeros((300, 900, 3), np.uint8)
    rgb_palette[:] = [r, g, b]

    st.image(hsv_color, caption="HSV Palette")
    st.image(rgb_palette, caption="RGB Palette")

    if st.button("Save Palettes"):
        hsv_filename = "hsv_palette.png"
        rgb_filename = "rgb_palette.png"
        cv2.imwrite(hsv_filename, hsv_color)
        cv2.imwrite(rgb_filename, rgb_palette)
        st.success(f"Palettes saved as `{hsv_filename}` and `{rgb_filename}`!")

elif nav_option == "Segmentation":
    uploaded_file = st.session_state.get('uploaded_file')  # Retrieve the uploaded file from session state
    if uploaded_file is None:
        st.warning("Please upload a file in the 'Upload' section first.")
    else:
        try:
            # Reset file pointer to the beginning
            uploaded_file.seek(0)

            # Check if the uploaded file is an image or video
            if uploaded_file.type.startswith('image'):
                try:
                    # Read the uploaded file as bytes
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if image is None:  # Check if the image is empty
                        st.error("Failed to load the image. The file might be corrupted or not a valid image format.")
                    else:
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

                        # Add kernel size slider
                        kernel_size = st.sidebar.slider("Kernel Size", 1, 30, 5, step=2, help="Set the kernel size (odd values only).")
                        kernel_size = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)  # Ensure odd kernel size

                        # Apply mask
                        lower_bound = np.array([lower_h, lower_s, lower_v])
                        upper_bound = np.array([upper_h, upper_s, upper_v])
                        mask, result = apply_mask(image, lower_bound, upper_bound)

                        # Check if the mask is empty
                        if not np.any(mask):
                            st.warning("The segmentation mask is empty. Try adjusting the HSV parameters.")

                        # Apply morphological operations
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

                        # Display results
                        st.image(mask, caption='Mask', use_container_width=True)
                        st.image(result, caption='Segmented Result', use_container_width=True)

                        # Save results
                        if st.button("Save Results", key="save_results", help="Click to save the segmented results."):
                            mask_filename = "mask.png"
                            result_filename = "result.png"
                            cv2.imwrite(mask_filename, mask)
                            cv2.imwrite(result_filename, result)
                            st.success(f"Results saved as {mask_filename} and {result_filename}")

                            # Add download buttons for the results
                            with open(mask_filename, "rb") as mask_file:
                                st.download_button(
                                    label="Download Mask",
                                    data=mask_file,
                                    file_name=mask_filename,
                                    mime="image/png"
                                )
                            with open(result_filename, "rb") as result_file:
                                st.download_button(
                                    label="Download Segmented Result",
                                    data=result_file,
                                    file_name=result_filename,
                                    mime="image/png"
                                )
                except Exception as e:
                    st.error(f"An unexpected error occurred while processing the image: {e}")

            elif uploaded_file.type.startswith('video'):
                if not is_valid_video(uploaded_file):
                    st.error("Invalid video file. Please upload a valid video.")
                else:
                    # Save the uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                        temp_video_file.write(uploaded_file.read())
                        temp_video_filename = temp_video_file.name

                    # Ensure the temporary file is closed before processing
                    temp_video_file.close()

                    try:
                        # Load the video
                        video = cv2.VideoCapture(temp_video_filename)
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(video.get(cv2.CAP_PROP_FPS))

                        # Slider to control video playback
                        frame_index = st.slider("Frame", 0, frame_count - 1, 0, help="Use this slider to navigate through the video.")

                        # Set the video to the selected frame
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = video.read()

                        if ret:
                            # Display the original frame
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Original Video Frame', use_container_width=True)

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

                            # Add kernel size slider
                            kernel_size = st.sidebar.slider("Kernel Size", 1, 30, 5, step=2, help="Set the kernel size (odd values only).")
                            kernel_size = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)  # Ensure odd kernel size

                            # Apply mask
                            lower_bound = np.array([lower_h, lower_s, lower_v])
                            upper_bound = np.array([upper_h, upper_s, upper_v])
                            mask, result = apply_mask(frame, lower_bound, upper_bound)

                            # Apply morphological operations
                            kernel = np.ones((kernel_size, kernel_size), np.uint8)
                            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

                            # Display synchronized frames
                            st.image(mask, caption='Mask Frame', use_container_width=True)
                            st.image(result, caption='Segmented Video Frame', use_container_width=True)

                    except Exception as e:
                        st.error(f"An error occurred while processing the video: {e}")
                    finally:
                        # Release the video capture object and clean up the temporary file
                        if video is not None:
                            video.release()
                        if os.path.exists(temp_video_filename):
                            os.remove(temp_video_filename)  # Ensure temporary file is deleted

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
