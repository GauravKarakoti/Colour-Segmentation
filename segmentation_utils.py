import cv2
import numpy as np
import os

def nothing(x):
    pass

def get_valid_kernel_size(value):
    return max(1, value | 1)

def check_file_access(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"Error: '{file_path}' is not a valid file.")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Error: '{file_path}' is not readable or access is denied.")

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), int(height))
    else:
        r = width / float(w)
        dim = (int(width), int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def load_image(uploaded_file):
    # Read the image from the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Error: Unable to load image. It may be corrupted or unsupported.")
        
    return resize_with_aspect_ratio(img, width=512)

def load_video(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    check_file_access(temp_file_path)  # Check if the video file exists and is accessible

    print(f"Loading video from path: {temp_file_path}")  # Debug logging for video path

    valid_video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')  # Ensure the uploaded file has a valid extension

    if not temp_file_path.lower().endswith(valid_video_extensions):
        raise ValueError(f"Error: '{temp_file_path}' is not a valid video file. Please provide a supported video format.")

    try:
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video file '{temp_file_path}'. It may be corrupted or unsupported.")

        return cap
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading video '{temp_file_path}': {str(e)}")

def release_video(cap):
    if cap is not None and cap.isOpened():
        cap.release()

def create_named_window(window_name, topmost=True):
    try:
        cv2.namedWindow(window_name)
        if topmost:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error while creating window '{window_name}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while creating window '{window_name}': {str(e)}")

def create_trackbar(name, window_name, min_val, max_val, default_val, callback=nothing):
    cv2.createTrackbar(name, window_name, default_val, max_val, callback)

def create_display_windows(input_type):
    if input_type not in ["img", "video"]:
        raise ValueError("Invalid input_type. Use 'img' for image or 'video' for real-time processing.")

    window_names = ["Mask", "Result", "Original"]

    for window in window_names:
        create_named_window(window)

def create_trackbars(window_name="Tracking"):
    create_named_window(window_name)
    create_trackbar("LH", window_name, 0, 179, 0)
    create_trackbar("LS", window_name, 0, 255, 50)
    create_trackbar("LV", window_name, 0, 255, 50)
    create_trackbar("UH", window_name, 0, 179, 179)
    create_trackbar("US", window_name, 0, 255, 255)
    create_trackbar("UV", window_name, 0, 255, 255)

def get_trackbar_values(window_name="Tracking"):
    l_h = cv2.getTrackbarPos("LH", window_name)
    l_s = cv2.getTrackbarPos("LS", window_name)
    l_v = cv2.getTrackbarPos("LV", window_name)
    u_h = cv2.getTrackbarPos("UH", window_name)
    u_s = cv2.getTrackbarPos("US", window_name)
    u_v = cv2.getTrackbarPos("UV", window_name)

    # Ensure lower bound is never greater than upper bound
    lower_bound = np.array([min(l_h, u_h), min(l_s, u_s), min(l_v, u_v)])
    upper_bound = np.array([max(l_h, u_h), max(l_s, u_s), max(l_v, u_v)])

    return lower_bound, upper_bound

def apply_mask(image, lower, upper, hsv_converted=False, kernel_size=1):
    """
    Apply a mask to the image based on the given lower and upper HSV bounds.
    :param image: Input image (BGR or HSV depending on hsv_converted flag)
    :param lower: Lower HSV bound
    :param upper: Upper HSV bound
    :param hsv_converted: Boolean indicating if the image is already in HSV
    :param kernel_size: Size of the morphological kernel for post-processing
    :return: Tuple of (mask, result)
    """
    try:
        # Convert to HSV only if not already converted
        hsv_image = image if hsv_converted else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Apply morphological operations if kernel_size > 1
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            result = cv2.bitwise_and(image, image, mask=mask)

        return mask, result
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error during mask application: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during mask application: {str(e)}")

def display_results(original=None, mask=None, result=None, frame=None):
    """
    Display the results in OpenCV windows.

    Args:
    - original: The original image (None for video mode).
    - mask: The binary mask of the detected region.
    - result: The final segmented output.
    - frame: The current video frame (None for image mode).
    """
    try:
        if frame is not None:
            cv2.imshow("Original", frame)
        elif original is not None:
            cv2.imshow("Original", original)

        if mask is not None:
            cv2.imshow("Mask", mask)

        if result is not None:
            cv2.imshow("Result", result)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error during display: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during display: {str(e)}")
