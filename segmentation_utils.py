import cv2
import numpy as np
import os

def nothing(x):
    pass

def load_image(image_path):
    """Load and resize image, return image or None if error occurs."""
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open image file '{image_path}'.")
        return None

    return cv2.resize(img, (512, 512))

def load_video(video_path):
    """Load video file and return video capture object."""
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None

    return cap

def create_trackbars(window_name="Tracking"):
    """Create HSV trackbars for segmentation."""
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    cv2.createTrackbar("LH", window_name, 0, 179, nothing)
    cv2.createTrackbar("LS", window_name, 0, 255, nothing)
    cv2.createTrackbar("LV", window_name, 0, 255, nothing)
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

    original_values = (l_h, l_s, l_v, u_h, u_s, u_v)

    l_h = min(l_h, u_h)
    l_s = min(l_s, u_s)
    l_v = min(l_v, u_v)
    u_h = max(original_values[0], u_h)
    u_s = max(original_values[1], u_s)
    u_v = max(original_values[2], u_v)

    if (l_h, l_s, l_v, u_h, u_s, u_v) != original_values:
        cv2.setTrackbarPos("LH", window_name, l_h)
        cv2.setTrackbarPos("LS", window_name, l_s)
        cv2.setTrackbarPos("LV", window_name, l_v)
        cv2.setTrackbarPos("UH", window_name, u_h)
        cv2.setTrackbarPos("US", window_name, u_s)
        cv2.setTrackbarPos("UV", window_name, u_v)
        
    return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])

def apply_mask(image, lower_bound, upper_bound):
    """Apply mask and return masked result."""
    mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result

def create_display_windows(type):
    """Create windows for displaying images and keep them always on top."""
    window_names = ["Original", "Mask", "Result"]
    if type == "img":
        pass
    else:
        window_names.remove("Original")
        window_names += ["Frame"]
    for window in window_names:
        cv2.namedWindow(window)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)  

def display_results(original, mask, result,frame):
    """Display original, mask, and result images."""
    if frame is not None:
        cv2.imshow("Frame", frame)
    else:
        cv2.imshow("Original", original)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
