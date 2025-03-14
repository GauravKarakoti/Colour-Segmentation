import cv2
import numpy as np
import os

def nothing(x):
    """Callback function for trackbars (does nothing)."""
    pass

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio."""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def load_image(image_path):
    """Load and resize image, return image or None if error occurs."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: File '{image_path}' not found.")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not open image file '{image_path}'.")

    return resize_with_aspect_ratio(img, width=512)

def load_video(video_path):
    """Load a video file and return a VideoCapture object. Raises an error if the file is missing."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: File '{video_path}' not found.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file '{video_path}'.")

    return cap

def release_video(cap):
    """Safely release the VideoCapture object."""
    if cap is not None and cap.isOpened():
        cap.release()

def create_trackbars(window_name="Tracking"):
    """Create HSV trackbars for segmentation."""
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    cv2.createTrackbar("LH", window_name, 30, 179, nothing)
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

def apply_mask(image, lower_bound, upper_bound):
    """Apply a mask to segment colors in the HSV range with noise reduction."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
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

    window_names = ["Mask", "Result"]
    if input_type == "img":
        window_names.append("Original")
    else:
        window_names.append("Frame")

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
        cv2.imshow("Frame", frame)
    elif original is not None:
        cv2.imshow("Original", original)

    if mask is not None:
        cv2.imshow("Mask", mask)

    if result is not None:
        cv2.imshow("Result", result)
