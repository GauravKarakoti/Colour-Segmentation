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
    if isinstance(uploaded_file, str):  # If it's a file path
        uploaded_file = os.path.join("images", uploaded_file)
        img = cv2.imread(uploaded_file, cv2.IMREAD_COLOR)
    else:  # If it's a file-like object
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error: Unable to load image. It may be corrupted or unsupported.")

    return resize_with_aspect_ratio(img, width=512)  # Adjust as needed

def load_video(input_source):
    """
    Loads video from a file path or file-like object.

    Args:
        input_source (str or file-like object): File path or uploaded file.

    Returns:
        cv2.VideoCapture: OpenCV video capture object.
    """
    # If input is a file-like object (e.g., from upload)
    if hasattr(input_source, "read"):
        temp_file_path = "temp_video.mp4"
        
        # Write the uploaded video to a temporary file
        with open(temp_file_path, "wb") as f:
            f.write(input_source.read())

        print(f"Loading video from temporary file: {temp_file_path}")
        cap = cv2.VideoCapture(temp_file_path)

        # Clean up temp file after use
        if not cap.isOpened():
            os.remove(temp_file_path)
            raise ValueError("Error: Unable to open video file.")
        return cap

    # If input is a file path string
    elif isinstance(input_source, str):
        input_source = os.path.join("videos", input_source)
        print(f"Loading video from path: {input_source}")
        cap = cv2.VideoCapture(input_source)

        if not cap.isOpened():
            raise ValueError(f"Error: Unable to open video file '{input_source}'.")
        return cap

    else:
        raise TypeError("Invalid input source. Must be a file path or file-like object.")

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

def apply_mask(image, lower, upper, hsv_converted=False, kernel_size=1, apply_morph=True):
    """
    Apply a mask to the image based on the given lower and upper HSV bounds.
    :param image: Input image (BGR or HSV depending on hsv_converted flag)
    :param lower: Lower HSV bound
    :param upper: Upper HSV bound
    :param hsv_converted: Boolean indicating if the image is already in HSV
    :param kernel_size: Size of the morphological kernel for post-processing
    :param apply_morph: Boolean indicating whether to apply morphological operations
    :return: Tuple of (mask, result)
    """
    try:
        # Convert to HSV only if not already converted
        hsv_image = image if hsv_converted else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Apply morphological operations if apply_morph is True and kernel_size > 1
        if apply_morph and kernel_size > 1:
            kernel_size = get_valid_kernel_size(kernel_size)  # Ensure kernel size is odd and valid
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

def validate_numeric_input(value, min_val, max_val, default_val):
    """
    Validate numeric input to ensure it is within the specified range.
    :param value: Input value to validate.
    :param min_val: Minimum allowed value.
    :param max_val: Maximum allowed value.
    :param default_val: Default value to return if validation fails.
    :return: Validated value.
    """
    try:
        value = int(float(value))  # Handle float inputs by converting to int
        if min_val <= value <= max_val:
            return value
    except (ValueError, TypeError):
        pass
    return default_val

def validate_filename(filename, default_name):
    """
    Validate a filename to ensure it is safe and valid.
    :param filename: Input filename.
    :param default_name: Default name to use if validation fails.
    :return: Validated filename.
    """
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip() or default_name

def validate_filename_with_extension(filename, default_name, valid_extensions):
    """
    Validate a filename to ensure it is safe, valid, and has a proper extension.
    :param filename: Input filename.
    :param default_name: Default name to use if validation fails.
    :param valid_extensions: List of valid file extensions (e.g., ['png', 'jpg']).
    :return: Validated filename with a valid extension.
    """
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    filename = filename.strip() or default_name

    # Ensure the filename has a valid extension
    if not any(filename.lower().endswith(f".{ext}") for ext in valid_extensions):
        filename += f".{valid_extensions[0]}"  # Default to the first valid extension

    return filename
