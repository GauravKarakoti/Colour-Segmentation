import cv2
import numpy as np
import os

def nothing(x):
    """
    Callback function for trackbars. This function does nothing.
    
    Args:
        x (int): The trackbar position (unused).
    """
    pass

def get_valid_kernel_size(value):
    """
    Ensures the kernel size is always an odd number and at least 1.
    
    Args:
        value (int): The input kernel size.
    
    Returns:
        int: The nearest odd kernel size (minimum value is 1).
    """
    return max(1, value | 1)

def check_file_access(file_path):
    """
    Checks if the given file exists and is accessible.
    
    Args:
        file_path (str): The path to the file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a valid file.
        PermissionError: If the file is not readable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"Error: '{file_path}' is not a valid file.")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Error: '{file_path}' is not readable or access is denied.")

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image while maintaining its aspect ratio.
    
    Args:
        image (numpy.ndarray): The input image.
        width (int, optional): The desired width. Default is None.
        height (int, optional): The desired height. Default is None.
        inter (cv2.InterpolationFlags, optional): The interpolation method. Default is cv2.INTER_AREA.
    
    Returns:
        numpy.ndarray: The resized image.
    """
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
    """
    Loads and resizes an image from the specified path.
    
    Args:
        image_path (str): The name of the image file (assumed to be in the 'images' directory).
    
    Returns:
        numpy.ndarray: The loaded and resized image.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or corrupted.
        RuntimeError: If an unexpected error occurs.
    """
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
    """
    Loads a video file and returns a VideoCapture object.
    
    Args:
        video_path (str): The name of the video file (assumed to be in the 'videos' directory).
    
    Returns:
        cv2.VideoCapture: The video capture object.
    
    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file format is unsupported.
        RuntimeError: If an unexpected error occurs.
    """
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
    """
    Safely releases a VideoCapture object.
    
    Args:
        cap (cv2.VideoCapture): The video capture object.
    """
    if cap is not None and cap.isOpened():
        cap.release()

def create_named_window(window_name, topmost=True):
    """
    Creates an OpenCV window with an optional topmost property.
    
    Args:
        window_name (str): The name of the window.
        topmost (bool, optional): Whether the window should stay on top. Default is True.
    """
    try:
        cv2.namedWindow(window_name)
        if topmost:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error while creating window '{window_name}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while creating window '{window_name}': {str(e)}")

def create_trackbar(name, window_name, min_val, max_val, default_val, callback=nothing):
    """
    Creates a single trackbar in the specified window.
    
    Args:
        name (str): The name of the trackbar.
        window_name (str): The name of the window.
        min_val (int): Minimum value of the trackbar.
        max_val (int): Maximum value of the trackbar.
        default_val (int): Default value of the trackbar.
        callback (function, optional): Callback function for the trackbar. Default is `nothing`.
    """
    cv2.createTrackbar(name, window_name, default_val, max_val, callback)

def create_display_windows(input_type):
    """
    Create windows for displaying images and ensure they stay on top.
    
    Args:
        input_type (str): "img" for image processing, "video" for real-time video.

    Raises:
        ValueError: If the input_type is not "img" or "video".
    """
    if input_type not in ["img", "video"]:
        raise ValueError("Invalid input_type. Use 'img' for image or 'video' for real-time processing.")

    window_names = ["Mask", "Result", "Original"]

    for window in window_names:
        create_named_window(window)

def create_trackbars(window_name="Tracking"):
    """
    Creates HSV trackbars for interactive color segmentation.
    
    Args:
        window_name (str, optional): The name of the OpenCV window. Default is "Tracking".
    """
    create_named_window(window_name)
    create_trackbar("LH", window_name, 0, 179, 0)
    create_trackbar("LS", window_name, 0, 255, 50)
    create_trackbar("LV", window_name, 0, 255, 50)
    create_trackbar("UH", window_name, 0, 179, 179)
    create_trackbar("US", window_name, 0, 255, 255)
    create_trackbar("UV", window_name, 0, 255, 255)

def get_trackbar_values(window_name="Tracking"):
    """
    Retrieves HSV values from trackbars.
    
    Args:
        window_name (str, optional): The name of the OpenCV window. Default is "Tracking".
    
    Returns:
        tuple: A pair of numpy arrays representing lower and upper HSV bounds.
    """
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
