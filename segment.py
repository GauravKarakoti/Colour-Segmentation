import cv2
import argparse
import numpy as np
import logging
from segmentation_utils import validate_numeric_input,load_video,create_display_windows,get_trackbar_values,apply_mask,get_valid_kernel_size,display_results
import tkinter as tk
from tkinter import simpledialog

# Initialize logging
logging.basicConfig(filename='segmentation_errors.log', level=logging.ERROR)

# Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation of a video file.")
parser.add_argument("--video", type=str, help="Path to the video file.")
parser.add_argument("--output", type=str, default="output.avi", help="Path to save the output video.")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
args = parser.parse_args()

# Get the video path from command line argument or ask the user for input
video_path = args.video if args.video else input("Enter the video file path: ").strip()
output_path = args.output
fps = args.fps

# Validate FPS input
fps = validate_numeric_input(fps, 1, 120, 30)

# Load the video
try:
    video_data = load_video(video_path)
    if isinstance(video_data, tuple):
        cap = video_data[0]  # Unpack the first element as the VideoCapture object
    else:
        cap = video_data

    if not cap.isOpened():
        raise ValueError("Error opening video file.")
except FileNotFoundError:
    logging.error(f"File not found: {video_path}")
    print("Error: Video file not found. Please check the path and try again.")
    exit(1)
except cv2.error as e:
    logging.error(f"OpenCV error: {str(e)}")
    print("Error: Unable to open video file. Ensure the file is not corrupted and is in a supported format.")
    exit(1)
except Exception as e:
    logging.error(f"Unexpected error: {str(e)}")
    print("Error: An unexpected error occurred while loading the video.")
    exit(1)

# Create display windows
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

# Create trackbars
def create_trackbars(window_name):
    cv2.resizeWindow("Tracking", 500, 300)
    cv2.createTrackbar("LH", window_name, 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar("LS", window_name, 0, 255, nothing)  # Lower Saturation
    cv2.createTrackbar("LV", window_name, 0, 255, nothing)  # Lower Value
    cv2.createTrackbar("UH", window_name, 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar("US", window_name, 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)  # Upper Value
    cv2.createTrackbar("Kernel Size", window_name, 1, 30, nothing)

def nothing(x):
    pass

def prompt_filename(title, default_name, filetypes):
    root = tk.Tk()
    root.withdraw()  

    cv2.setWindowProperty("Tracking",cv2.WND_PROP_TOPMOST,0)
    cv2.setWindowProperty("Original",cv2.WND_PROP_TOPMOST,0)
    cv2.setWindowProperty("Mask",cv2.WND_PROP_TOPMOST,0)
    cv2.setWindowProperty("Result",cv2.WND_PROP_TOPMOST,0)

    filename = simpledialog.askstring(
        title, f"Enter filename for {title} (without extension):", initialvalue=default_name
    )

    if filename is None:
        print(f"Operation canceled. Using default filename: {default_name}")
        filename = default_name

    cv2.setWindowProperty("Tracking",cv2.WND_PROP_TOPMOST,1)
    cv2.setWindowProperty("Original",cv2.WND_PROP_TOPMOST,1)
    cv2.setWindowProperty("Mask",cv2.WND_PROP_TOPMOST,1)
    cv2.setWindowProperty("Result",cv2.WND_PROP_TOPMOST,1)
    root.destroy()  
    return filename

create_display_windows(input_type='video')

# Create the trackbars
create_trackbars("Tracking")

cv2.setWindowProperty("Tracking", cv2.WND_PROP_TOPMOST, 1)

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit(1)

height, width = frame.shape[:2]
frame_height, frame_width = 512, 512

# --- Dynamic Codec Selection with Better Handling ---
output_ext = output_path.split('.')[-1].lower()

# Define codec mapping dictionary for better flexibility
codec_map = {
    'avi': ('XVID', '.avi'),
    'mp4': ('mp4v', '.mp4'),
    'mov': ('avc1', '.mov'),
    'mkv': ('X264', '.mkv'),
    'webm': ('VP80', '.webm'),
}

# Check if the extension is supported
if output_ext in codec_map:
    fourcc, valid_ext = codec_map[output_ext]
    if output_ext != valid_ext[1:]:  # Ensure consistent extension
        print(f"Correcting extension to {valid_ext}")
        output_path = output_path.rsplit('.', 1)[0] + valid_ext
else:
    print(f"Unsupported format: .{output_ext}. Defaulting to AVI.")
    fourcc, output_path = 'XVID', output_path.rsplit('.', 1)[0] + '.avi'

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (frame_width * 3, frame_height))
frame_delay = 1 / fps  # Control FPS

# Function to resize an image while maintaining aspect ratio
def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Check if the image is single-channel or multi-channel
    if len(image.shape) == 2:  # Single-channel (e.g., grayscale)
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    else:  # Multi-channel (e.g., color)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image on the canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas

# Initialize the paused flag
paused = False
prev_tick = cv2.getTickCount()

# Start the video processing loop
try:
    while True:
        current_tick = cv2.getTickCount()

        try:
            if not paused:
                ret, frame = cap.read()  # Read a frame from the video
                if not ret:
                    print("End of video or unable to read frame. Exiting...")
                    break  # Exit when video ends or an error occurs

            # Get the current values of the trackbars (lower and upper bounds for segmentation)
            lower, upper = get_trackbar_values("Tracking")

            kernel_size = cv2.getTrackbarPos("Kernel Size", "Tracking")
            kernel_size = validate_numeric_input(kernel_size, 1, 30, 1)  # Ensure valid kernel size
            kernel_size = get_valid_kernel_size(kernel_size)

            # Apply mask directly on the BGR frame
            mask, result = apply_mask(frame, lower, upper, kernel_size=kernel_size)

            # Resize video frame, mask, and result while maintaining aspect ratio
            frame = resize_with_aspect_ratio(frame, frame_width, frame_height)
            mask = resize_with_aspect_ratio(mask, frame_width, frame_height)
            result = resize_with_aspect_ratio(result, frame_width, frame_height)

            # Adjustable Morphology Parameters: Dynamically adjust kernel size using trackbars
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a kernel of specified size
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Apply opening (dilation + erosion)

            # Display the original frame in the "Original" window
            cv2.imshow("Original", frame)

            # Display the mask and result side by side in the "Tracking" window
            display_results(frame=frame, mask=mask, result=result)

            # Convert mask to 3-channel image for saving (remove color mapping)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert grayscale mask to 3-channel without color mapping
            
            # Combine results into a single frame
            combined_output = np.hstack((frame, mask_3ch, result))
            
            # Write the combined frame to the output video
            try:
                if combined_output.shape[1] == frame_width * 3 and combined_output.shape[0] == frame_height:
                    out.write(combined_output)
                else:
                    raise ValueError("Combined output frame has invalid dimensions.")
            except Exception as e:
                logging.error(f"Error writing frame: {str(e)}")

            elapsed_time = (current_tick - prev_tick) / cv2.getTickFrequency()
            wait_time = max(1, int((1 / fps - elapsed_time) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF

            # Exit on ESC key
            if key == 27:
                break
            elif key == 32:  # Toggle pause on spacebar press
                paused = not paused
                if paused:
                    print("Video paused. Press SPACE to resume.")
                else:
                    print("Video resumed.")
            elif key == ord('s'):  # Save current frame, mask, and result
                # Define filetypes for each save operation
                frame_filetypes = ["png", "jpg"]
                mask_filetypes = ["png", "jpg"]
                result_filetypes = ["png", "jpg"]

                # Get filenames and append extensions
                frame_filename = prompt_filename("Frame Image", "segment_frame", frame_filetypes)
                frame_filename = f"{frame_filename}.{frame_filetypes[0]}"  # Append .png

                mask_filename = prompt_filename("Mask Image", "segment_mask", mask_filetypes)
                mask_filename = f"{mask_filename}.{mask_filetypes[0]}"  # Append .png

                result_filename = prompt_filename("Result Image", "segment_result", result_filetypes)
                result_filename = f"{result_filename}.{result_filetypes[0]}"  # Append .png

                # Convert result to BGR before saving
                result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

                # Save images with correct extensions
                cv2.imwrite(frame_filename, frame)
                cv2.imwrite(mask_filename, mask_3ch)
                cv2.imwrite(result_filename, result_bgr)

                print(f"Frame saved as {frame_filename}")
                print(f"Mask saved as {mask_filename}")
                print(f"Result saved as {result_filename}")

            prev_tick = current_tick

        except cv2.error as e:
            logging.error(f"OpenCV error during processing: {str(e)}")
            print("Error: An error occurred during video processing. Check the log file for details.")
            break
        except Exception as e:
            logging.error(f"Unexpected error during processing: {str(e)}")
            print("Error: An unexpected error occurred during video processing. Check the log file for details.")
            break
except Exception as e:
    logging.error(f"Critical error in video processing loop: {str(e)}")
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
