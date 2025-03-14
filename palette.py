import cv2
import numpy as np

def nothing(x):
    """Callback function for trackbars (unused but required)."""
    pass

def create_color_palette_window():
    """Creates the OpenCV window and trackbars for color selection."""
    cv2.namedWindow("Colour Palette")
    cv2.createTrackbar("R", "Colour Palette", 0, 255, nothing)
    cv2.createTrackbar("G", "Colour Palette", 0, 255, nothing)
    cv2.createTrackbar("B", "Colour Palette", 0, 255, nothing)

def get_trackbar_values():
    """Retrieves the current values from the RGB trackbars."""
    r = cv2.getTrackbarPos("R", "Colour Palette")
    g = cv2.getTrackbarPos("G", "Colour Palette")
    b = cv2.getTrackbarPos("B", "Colour Palette")
    return r, g, b

def calculate_luminance(r, g, b):
    """Calculates brightness (luminance) using a standard formula."""
    return 0.299 * r + 0.587 * g + 0.114 * b

def update_display(img, r, g, b):
    """Updates the display image with selected color and text information."""
    img[:] = [b, g, r]
    luminance = calculate_luminance(r, g, b)
    text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
    text = f"R={r}, G={g}, B={b}"
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
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
        
        r, g, b = get_trackbar_values()
        update_display(img, r, g, b)
        color_img = np.full((300, 512, 3), (b, g, r), dtype=np.uint8)
        cv2.imshow("Colour Palette", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):  # 's' key to save
            save_images(img, color_img)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()