import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

# Create a black image of size 300x512 with 3 color channels (RGB)
img = np.zeros((300, 512, 3), np.uint8)

# Function to be called when trackbar values change (not used but required)
def nothing(x):
    return

# Create a window named "Colour Palette"
cv2.namedWindow("Colour Palette")

# Create trackbars for Red, Green, and Blue color adjustments
cv2.createTrackbar("R", "Colour Palette", 0, 255, nothing)
cv2.createTrackbar("G", "Colour Palette", 0, 255, nothing)
cv2.createTrackbar("B", "Colour Palette", 0, 255, nothing)

# Infinite loop to update the window with new colors
while True:
    # Exit the loop if the window is closed
    if cv2.getWindowProperty("Colour Palette", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Get the current positions of the trackbars
    r = cv2.getTrackbarPos("R", "Colour Palette")
    g = cv2.getTrackbarPos("G", "Colour Palette")
    b = cv2.getTrackbarPos("B", "Colour Palette")
    
    # Update the image with the selected color
    img[:] = [b, g, r]
    
    # Create a separate image filled with the selected color (for saving later)
    color_img = np.full((300, 512, 3), (b, g, r), dtype=np.uint8)
    
    # Calculate the brightness (luminance) using standard formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Choose text color based on brightness to ensure visibility
    text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
    
    # Create a text label showing the RGB values
    text = f"R={r}, G={g}, B={b}"
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Display the updated image
    cv2.imshow("Colour Palette", img)
    
    # Capture key press events
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break
    elif key == ord('s'):  # Press 's' to save images
        cv2.imwrite("selected_color_with_text.png", img)  # Save with text overlay
        cv2.imwrite("selected_color.png", color_img)  # Save plain color image
        print("Saved: 'selected_color.png' (without text) & 'selected_color_with_text.png' (with text)")

# Close all OpenCV windows after exiting the loop
cv2.destroyAllWindows()