import cv2
import numpy as np
import os

def nothing(x):
    """Callback function for trackbars (does nothing)."""
    pass

def check_file_access(file_path):
    """Check if the file exists and is accessible."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"Error: '{file_path}' is not a valid file.")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Error: '{file_path}' is not readable or access is denied.")

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio."""
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

def load_image(image_path):
    """Load and resize image, return image or None if error occurs."""
    image_path = os.path.join('images', image_path)
    check_file_access(image_path)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    if not image_path.lower().endswith(valid_extensions):
        raise ValueError(f"Error: '{image_path}' is not an image file. Please provide a valid image format.")

    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            raise ValueError(f"Error: Unable to load image '{image_path}'. It may be corrupted or unsupported.")
            
        return resize_with_aspect_ratio(img, width=512)
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading image '{image_path}': {str(e)}")

def load_video(video_path):
    """Load a video file and return a VideoCapture object. Raises an error if the file is missing."""
    video_path = os.path.join('videos', video_path)
    check_file_access(video_path)

    valid_video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    if not video_path.lower().endswith(valid_video_extensions):
        raise ValueError(f"Error: '{video_path}' is not a valid video file. Please provide a supported video format.")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video file '{video_path}'. It may be corrupted or unsupported.")

        return cap
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading video '{video_path}': {str(e)}")

def release_video(cap):
    """Safely release the VideoCapture object."""
    if cap is not None and cap.isOpened():
        cap.release()

def create_trackbars(window_name="Tracking"):
    """Create HSV trackbars for segmentation."""
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    cv2.createTrackbar("LH", window_name, 0, 179, nothing)
    cv2.createTrackbar("LS", window_name, 50, 255, nothing)
    cv2.createTrackbar("LV", window_name, 50, 255, nothing)
    cv2.createTrackbar("UH", window_name, 179, 179, nothing)
    cv2.createTrackbar("US", window_name, 255, 255, nothing)
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)

def get_trackbar_values(window_name="Tracking"):
    """Retrieve HSV values from trackbars."""
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

def apply_mask(image, lower_bound, upper_bound,  kernel_size=5):
    """Apply a mask to segment colors in the HSV range with noise reduction."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 

    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result

def create_display_windows(input_type):
    """
    Create windows for displaying images and ensure they stay on top.
    
    Args:
    - input_type (str): "img" for image processing, "video" for real-time video.

    Raises:
    - ValueError: If the input_type is not "img" or "video".
    """
    if input_type not in ["img", "video"]:
        raise ValueError("Invalid input_type. Use 'img' for image or 'video' for real-time processing.")

    window_names = ["Mask", "Result", "Original"]

    for window in window_names:
        cv2.namedWindow(window)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

def display_results(original=None, mask=None, result=None, frame=None):
    """
    Display the results in OpenCV windows.

    Args:
    - original: The original image (None for video mode).
    - mask: The binary mask of the detected region.
    - result: The final segmented output.
    - frame: The current video frame (None for image mode).
    """
    if frame is not None:
        cv2.imshow("Original", frame)
    elif original is not None:
        cv2.imshow("Original", original)

    if mask is not None:
        cv2.imshow("Mask", mask)

    if result is not None:
        cv2.imshow("Result", result)
