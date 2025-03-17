import cv2
import numpy as np
from tkinter import Tk, simpledialog

def nothing(x):
    pass

def create_hsv_palette_window():
    cv2.namedWindow("HSV Palette")
    cv2.createTrackbar("LH", "HSV Palette", 0, 179, nothing)
    cv2.createTrackbar("LS", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("LV", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("UH", "HSV Palette", 179, 179, nothing)
    cv2.createTrackbar("US", "HSV Palette", 255, 255, nothing)
    cv2.createTrackbar("UV", "HSV Palette", 255, 255, nothing)

def create_rgb_palette_window():
    cv2.namedWindow("RGB Palette")
    cv2.createTrackbar("R", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("G", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("B", "RGB Palette", 0, 255, nothing)

def get_hsv_values():
    l_h = cv2.getTrackbarPos("LH", "HSV Palette")
    l_s = cv2.getTrackbarPos("LS", "HSV Palette")
    l_v = cv2.getTrackbarPos("LV", "HSV Palette")
    u_h = cv2.getTrackbarPos("UH", "HSV Palette")
    u_s = cv2.getTrackbarPos("US", "HSV Palette")
    u_v = cv2.getTrackbarPos("UV", "HSV Palette")
    return (l_h, l_s, l_v), (u_h, u_s, u_v)

def get_rgb_values():
    r = cv2.getTrackbarPos("R", "RGB Palette")
    g = cv2.getTrackbarPos("G", "RGB Palette")
    b = cv2.getTrackbarPos("B", "RGB Palette")
    return (r, g, b)

def display_hsv_palette(img, l_h, l_s, l_v, u_h, u_s, u_v):
    img[:] = [l_h, l_s, l_v]
    text_color = (255, 255, 255)
    cv2.putText(img, f"LH={l_h}, LS={l_s}, LV={l_v}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, f"UH={u_h}, US={u_s}, UV={u_v}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, "Press 'S' to save | Press 'ESC' to exit", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

def display_rgb_palette(img, r, g, b):
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

    text_color = (255, 255, 255)
    cv2.putText(img, f"R={r}, G={g}, B={b}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, "Press 'S' to save | Press 'ESC' to exit", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

def save_images(img_hsv, img_rgb):
    cv2.imwrite("hsv_palette_with_text.png", img_hsv)
    cv2.imwrite("rgb_palette_with_text.png", img_rgb)
    
    img_hsv_plain = np.zeros_like(img_hsv)
    img_rgb_plain = np.zeros_like(img_rgb)
    img_hsv_plain[:] = img_hsv[0, 0]
    img_rgb_plain[:] = img_rgb[0, 0]
    
    cv2.imwrite("hsv_palette.png", img_hsv_plain)
    cv2.imwrite("rgb_palette.png", img_rgb_plain)
    print("Saved: 'hsv_palette_with_text.png', 'rgb_palette_with_text.png', 'hsv_palette.png', and 'rgb_palette.png'")

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

    img_hsv = np.zeros((300, 512, 3), np.uint8)
    img_rgb = np.zeros((300, 512, 3), np.uint8)
    create_hsv_palette_window()
    create_rgb_palette_window()
    
    while True:
        (l_h, l_s, l_v), (u_h, u_s, u_v) = get_hsv_values()
        r, g, b = get_rgb_values()
        
        display_hsv_palette(img_hsv, l_h, l_s, l_v, u_h, u_s, u_v)
        display_rgb_palette(img_rgb, r, g, b)
        
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

        cv2.imshow("HSV Palette", img_hsv)
        cv2.imshow("RGB Palette", img_rgb)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_images(img_hsv, img_rgb)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try :
        main()
    except KeyboardInterrupt:
        print("[INFO] Program terminated by user.")
        cv2.destroyAllWindows()