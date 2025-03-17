import cv2
import numpy as np
from tkinter import Tk, simpledialog

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

    overlay = img.copy()

    text = f"R={r}, G={g}, B={b}"
    instructions = "Press 'S' to save | 'I' to input RGB | 'R' to reset | 'ESC' to exit"

    # Get the sizes of text
    text_size_rgb = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_size_inst = cv2.getTextSize(instructions,cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]

    # Background rectangles behind text
    x_offset_rgb = 20
    y_offset_rgb = 40
    cv2.rectangle(overlay, (x_offset_rgb - 5, y_offset_rgb - 30), 
                  (x_offset_rgb + text_size_rgb[0] + 5, y_offset_rgb + 10), (0, 0, 0), -1)

    x_offset_inst = 20
    y_offset_inst = 280
    cv2.rectangle(overlay, (x_offset_inst - 5, y_offset_inst - 20), 
                  (x_offset_inst + text_size_inst[0] + 5, y_offset_inst + 10), (0, 0, 0), -1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x_offset_rgb, y_offset_rgb), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(img, instructions, (x_offset_inst, y_offset_inst), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

def ask_filename():
    """Opens a GUI popup to ask for the filename."""
    root = Tk()
    root.withdraw()
    filename = simpledialog.askstring("Save Image", "Enter filename:")
    return filename.strip() if filename else "selected_color"

def save_images(img, color_img):
    """Saves the selected color images with user-defined filenames."""
    filename = ask_filename()
    img_with_text_path = f"{filename}_with_text.png"
    img_without_text_path = f"{filename}.png"
    
    try:
        cv2.imwrite(img_with_text_path, img)
        cv2.imwrite(img_without_text_path, color_img)
        print(f"Saved: '{img_without_text_path}' & '{img_with_text_path}'")
    except (cv2.error, IOError) as e:
        print(f"Error saving images: {e}")

def reset_trackbar_values():
    """Resets trackbars to 0."""
    cv2.setTrackbarPos("R", "Colour Palette", 0)
    cv2.setTrackbarPos("G", "Colour Palette", 0)
    cv2.setTrackbarPos("B", "Colour Palette", 0)

def get_rgb_input():
    """Opens a GUI popup to input RGB values manually."""
    root = Tk()
    root.withdraw()  

    current_r = cv2.getTrackbarPos("R", "Colour Palette")
    current_g = cv2.getTrackbarPos("G", "Colour Palette")
    current_b = cv2.getTrackbarPos("B", "Colour Palette")

    def get_value(color_name, current_value):
        value = simpledialog.askstring("Input RGB", f"Enter {color_name} value (0-255):")
        return int(value) if value and value.isdigit() else current_value
    try:
        r = get_value("Red", current_r)
        g = get_value("Green", current_g)
        b = get_value("Blue", current_b)

        # Ensure the value in the range
        r, g, b = max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))
        
        # Update the trackbar
        cv2.setTrackbarPos("R", "Colour Palette", r)
        cv2.setTrackbarPos("G", "Colour Palette", g)
        cv2.setTrackbarPos("B", "Colour Palette", b)

    except (ValueError, TypeError):
        print("[ERROR] Invalid input. Please enter numbers between 0 and 255.")

def main():
    """Main function to run the color palette application."""
    img = np.zeros((300, 750, 3), np.uint8)  
    create_color_palette_window()

    while True:
        if cv2.getWindowProperty("Colour Palette", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        r, g, b = get_trackbar_values()
        update_display(img, r, g, b)
        color_img = np.full((300, 750, 3), (b, g, r), dtype=np.uint8)
        cv2.imshow("Colour Palette", img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break
        elif key == ord('s') or key == ord('S'):  
            save_images(img, color_img)
        elif key == ord('r') or key == ord('R'):  
            reset_trackbar_values()
        elif key == ord('i') or key == ord('I'):  
            get_rgb_input()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try :
        main()
    except KeyboardInterrupt:
        print("[INFO] Program terminated by user.")
        cv2.destroyAllWindows()