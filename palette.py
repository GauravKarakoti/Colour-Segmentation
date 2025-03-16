import cv2
import numpy as np

def nothing(x):
    """Callback function for trackbars (unused but required)."""
    pass

def create_color_palette_window():
    """Creates the OpenCV window and trackbars for color selection."""
    cv2.namedWindow("Colour Palette")
    # Create HSV trackbars
    cv2.createTrackbar("LH", "Colour Palette", 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar("LS", "Colour Palette", 50, 255, nothing)  # Lower Saturation
    cv2.createTrackbar("LV", "Colour Palette", 50, 255, nothing)  # Lower Value
    cv2.createTrackbar("UH", "Colour Palette", 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar("US", "Colour Palette", 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("UV", "Colour Palette", 255, 255, nothing)  # Upper Value

def get_trackbar_values():
    """Retrieves the current values from the HSV trackbars."""
    l_h = cv2.getTrackbarPos("LH", "Colour Palette")
    l_s = cv2.getTrackbarPos("LS", "Colour Palette")
    l_v = cv2.getTrackbarPos("LV", "Colour Palette")
    u_h = cv2.getTrackbarPos("UH", "Colour Palette")
    u_s = cv2.getTrackbarPos("US", "Colour Palette")
    u_v = cv2.getTrackbarPos("UV", "Colour Palette")
    
    return (l_h, l_s, l_v), (u_h, u_s, u_v)

def calculate_luminance(r, g, b):
    """Calculates brightness (luminance) using a standard formula."""
    return 0.299 * r + 0.587 * g + 0.114 * b

def update_display(img, l_h, l_s, l_v, u_h, u_s, u_v):
    """Updates the display image with selected color and text information."""
    # Handle hue wrapping for red colors
    if l_h > u_h:
        lower_bound = np.array([l_h, min(l_s, u_s), min(l_v, u_v)])
        upper_bound = np.array([179, max(l_s, u_s), max(l_v, u_v)])
        lower_bound2 = np.array([0, min(l_s, u_s), min(l_v, u_v)])
        upper_bound2 = np.array([u_h, max(l_s, u_s), max(l_v, u_v)])
    else:
        lower_bound = np.array([min(l_h, u_h), min(l_s, u_s), min(l_v, u_v)])
        upper_bound = np.array([max(l_h, u_h), max(l_s, u_s), max(l_v, u_v)])

    # Update the display color and text information
    img[:] = [lower_bound[0], lower_bound[1], lower_bound[2]]  # Using hue as RGB for display
    luminance = calculate_luminance(lower_bound[0], lower_bound[1], lower_bound[2])
    text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
    text = f"LH={l_h}, LS={l_s}, LV={l_v} | UH={u_h}, US={u_s}, UV={u_v}"
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    instructions = "Press 'S' to save | Press 'ESC' to exit"
    cv2.putText(img, instructions, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

def save_images(img, color_img):
    """Saves the selected color images with and without text overlay."""
    try:
        cv2.imwrite("selected_color_with_text.png", img)
        cv2.imwrite("selected_color.png", color_img)
        print("Saved: 'selected_color.png' (without text) & 'selected_color_with_text.png' (with text)")
    except Exception as e:
        print(f"Error saving images: {e}")

def main():
    """Main function to run the color palette application."""
    img = np.zeros((300, 512, 3), np.uint8)
    create_color_palette_window()
    
    while True:
        if cv2.getWindowProperty("Colour Palette", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        (l_h, l_s, l_v), (u_h, u_s, u_v) = get_trackbar_values()
        update_display(img, l_h, l_s, l_v, u_h, u_s, u_v)
        
        # Create a display color image based on the selected HSV range
        color_img = np.full((300, 512, 3), (l_h, l_s, l_v), dtype=np.uint8)
        
        # Display the color palette window
        cv2.imshow("Colour Palette", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):  # 's' key to save
            save_images(img, color_img)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
