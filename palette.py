import cv2
import numpy as np

def nothing(x):
    pass

def create_hsv_palette_window():
    cv2.namedWindow("HSV Palette")
    if cv2.getWindowProperty("HSV Palette", cv2.WND_PROP_VISIBLE) < 1:
        print("Error: HSV Palette window not created.")
        return False
    cv2.createTrackbar("LH", "HSV Palette", 0, 179, nothing)
    cv2.createTrackbar("LS", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("LV", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("UH", "HSV Palette", 179, 179, nothing)
    cv2.createTrackbar("US", "HSV Palette", 255, 255, nothing)
    cv2.createTrackbar("UV", "HSV Palette", 255, 255, nothing)
    return True

def create_rgb_palette_window():
    cv2.namedWindow("RGB Palette")
    if cv2.getWindowProperty("RGB Palette", cv2.WND_PROP_VISIBLE) < 1:
        print("Error: RGB Palette window not created.")
        return False
    cv2.createTrackbar("R", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("G", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("B", "RGB Palette", 0, 255, nothing)
    return True

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
    # Blend lower and upper HSV values
    blended_h = (l_h + u_h) // 2
    blended_s = (l_s + u_s) // 2
    blended_v = (l_v + u_v) // 2
    img[:] = [blended_h, blended_s, blended_v]
    text_color = (255, 255, 255)
    cv2.putText(img, f"LH={l_h}, LS={l_s}, LV={l_v}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, f"UH={u_h}, US={u_s}, UV={u_v}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, "Press 'S' to save | Press 'ESC' to exit", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

def display_rgb_palette(img, r, g, b):
    img[:] = [b, g, r]
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

def main():
    img_hsv = np.zeros((300, 512, 3), np.uint8)
    img_rgb = np.zeros((300, 512, 3), np.uint8)
    if not create_hsv_palette_window() or not create_rgb_palette_window():
        print("Error: Could not create one or more windows.")
        return
    
    while True:
        # Check if windows are still open
        if cv2.getWindowProperty("HSV Palette", cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty("RGB Palette", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        (l_h, l_s, l_v), (u_h, u_s, u_v) = get_hsv_values()
        r, g, b = get_rgb_values()
        
        display_hsv_palette(img_hsv, l_h, l_s, l_v, u_h, u_s, u_v)
        display_rgb_palette(img_rgb, r, g, b)
        
        cv2.imshow("HSV Palette", img_hsv)
        cv2.imshow("RGB Palette", img_rgb)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_images(img_hsv, img_rgb)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()