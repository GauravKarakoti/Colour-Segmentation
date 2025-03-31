import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import queue
import threading

def nothing(x):
    pass

def create_hsv_palette_window():
    cv2.namedWindow("HSV Palette", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("LH", "HSV Palette", 0, 179, nothing)
    cv2.createTrackbar("LS", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("LV", "HSV Palette", 50, 255, nothing)
    cv2.createTrackbar("UH", "HSV Palette", 179, 179, nothing)
    cv2.createTrackbar("US", "HSV Palette", 255, 255, nothing)
    cv2.createTrackbar("UV", "HSV Palette", 255, 255, nothing)
    return True

def create_rgb_palette_window():
    cv2.namedWindow("RGB Palette", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("R", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("G", "RGB Palette", 0, 255, nothing)
    cv2.createTrackbar("B", "RGB Palette", 0, 255, nothing)
    return True

def get_hsv_values():
    try:
        l_h = cv2.getTrackbarPos("LH", "HSV Palette")
        l_s = cv2.getTrackbarPos("LS", "HSV Palette")
        l_v = cv2.getTrackbarPos("LV", "HSV Palette")
        u_h = cv2.getTrackbarPos("UH", "HSV Palette")
        u_s = cv2.getTrackbarPos("US", "HSV Palette")
        u_v = cv2.getTrackbarPos("UV", "HSV Palette")
        return (l_h, l_s, l_v), (u_h, u_s, u_v)
    except cv2.error:
        return (0, 0, 0), (0, 0, 0)

def get_rgb_values():
    try:
        r = cv2.getTrackbarPos("R", "RGB Palette")
        g = cv2.getTrackbarPos("G", "RGB Palette")
        b = cv2.getTrackbarPos("B", "RGB Palette")
        return r, g, b
    except cv2.error:
        return 0, 0, 0

def draw_text_with_semi_transparent_bg(img, text, position, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5, thickness=2):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def display_hsv_palette(img, l_h, l_s, l_v, u_h, u_s, u_v):
    if l_h > u_h:  # Handle Hue wrap-around
        hue_range = list(range(l_h, 180)) + list(range(0, u_h + 1))
        mid_hue = hue_range[len(hue_range) // 2]
    else:
        mid_hue = (l_h + u_h) // 2

    # Blend lower and upper HSV values
    blended_s = (l_s + u_s) // 2
    blended_v = (l_v + u_v) // 2
    
    # Create an HSV image and convert it to BGR
    hsv_color = np.full((300, 900, 3), (mid_hue, blended_s, blended_v), dtype=np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

    # Update the image with the corrected BGR color
    img[:] = bgr_color
    text_color = (255, 255, 255) if blended_v < 128 else (0, 0, 0)
    draw_text_with_semi_transparent_bg(img, f"LH={l_h}, LS={l_s}, LV={l_v}", (20, 50), color=text_color)
    draw_text_with_semi_transparent_bg(img, f"UH={u_h}, US={u_s}, UV={u_v}", (20, 90), color=text_color)
    draw_text_with_semi_transparent_bg(img, "Press 'S' to save | Press 'I' for RGB Input | Press 'H' for HSV Input | Press 'ESC' to exit", (20, 280), font_scale=0.6, color=text_color)

def display_rgb_palette(img, r, g, b):
    img[:] = [b, g, r]
    text_color = (255, 255, 255) if (r + g + b) / 3 < 128 else (0, 0, 0)
    draw_text_with_semi_transparent_bg(img, f"R={r}, G={g}, B={b}", (20, 50), color=text_color)
    draw_text_with_semi_transparent_bg(img, "Press 'S' to save | Press 'I' for RGB Input | Press 'H' for HSV Input | Press 'ESC' to exit", (20, 280), font_scale=0.6, color=text_color)

def save_images(img_hsv, img_rgb):
    root = tk.Tk()
    root.withdraw()

    hsv_filename = simpledialog.askstring("Save Image", '''Enter filename for HSV palette (without extension):''',initialvalue="default_hsv")
    rgb_filename = simpledialog.askstring("Save Image", '''Enter filename for RGB palette (without extension):''',initialvalue="default_rgb")

    if hsv_filename and rgb_filename:
        cv2.imwrite(f"{hsv_filename}_with_text.png", img_hsv)
        cv2.imwrite(f"{rgb_filename}_with_text.png", img_rgb)

        img_hsv_plain = np.zeros_like(img_hsv)
        img_rgb_plain = np.zeros_like(img_rgb)
        img_hsv_plain[:] = img_hsv[0, 0]
        img_rgb_plain[:] = img_rgb[0, 0]

        cv2.imwrite(f"{hsv_filename}.png", img_hsv_plain)
        cv2.imwrite(f"{rgb_filename}.png", img_rgb_plain)

        messagebox.showinfo("Success", f"Saved:\n{hsv_filename}_with_text.png\n{rgb_filename}_with_text.png\n{hsv_filename}.png\n{rgb_filename}.png")

rgb_queue = queue.Queue()
def rgb_input_window():
    def apply_values():
        try:
            r = int(entry_r.get())
            g = int(entry_g.get())
            b = int(entry_b.get())
            r, g, b = max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))
            rgb_queue.put((r, g, b))
            dialog_root.after(0, dialog_root.destroy)  # Non-blocking destroy
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid RGB values (0-255).")

    dialog_root = tk.Tk()
    dialog_root.title("RGB Input")

    tk.Label(dialog_root, text="R:").grid(row=0, column=0)
    tk.Label(dialog_root, text="G:").grid(row=1, column=0)
    tk.Label(dialog_root, text="B:").grid(row=2, column=0)

    entry_r = tk.Entry(dialog_root)
    entry_g = tk.Entry(dialog_root)
    entry_b = tk.Entry(dialog_root)

    entry_r.grid(row=0, column=1)
    entry_g.grid(row=1, column=1)
    entry_b.grid(row=2, column=1)

    tk.Button(dialog_root, text="Apply", command=apply_values).grid(row=3, column=0, columnspan=2)

    dialog_root.mainloop()
def start_rgb_window():
    threading.Thread(target=rgb_input_window, daemon=True).start()

hsv_queue = queue.Queue()
def hsv_input_window():
    dialog_root = tk.Tk()
    dialog_root.title("HSV Input")
    def apply_values():
        try:
            lh = int(entry_lh.get())
            ls = int(entry_ls.get())
            lv = int(entry_lv.get())
            uh = int(entry_uh.get())
            us = int(entry_us.get())
            uv = int(entry_uv.get())

            lh, ls, lv = max(0, min(lh, 179)), max(0, min(ls, 255)), max(0, min(lv, 255))
            uh, us, uv = max(0, min(uh, 179)), max(0, min(us, 255)), max(0, min(uv, 255))

            hsv_queue.put(((lh, ls, lv), (uh, us, uv)))
            dialog_root.after(0, dialog_root.destroy)  # Non-blocking destroy
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid HSV values.")

    dialog_root.title("HSV Input")

    tk.Label(dialog_root, text="LH:").grid(row=0, column=0)
    tk.Label(dialog_root, text="LS:").grid(row=1, column=0)
    tk.Label(dialog_root, text="LV:").grid(row=2, column=0)
    tk.Label(dialog_root, text="UH:").grid(row=3, column=0)
    tk.Label(dialog_root, text="US:").grid(row=4, column=0)
    tk.Label(dialog_root, text="UV:").grid(row=5, column=0)

    entry_lh = tk.Entry(dialog_root)
    entry_ls = tk.Entry(dialog_root)
    entry_lv = tk.Entry(dialog_root)
    entry_uh = tk.Entry(dialog_root)
    entry_us = tk.Entry(dialog_root)
    entry_uv = tk.Entry(dialog_root)

    entry_lh.grid(row=0, column=1)
    entry_ls.grid(row=1, column=1)
    entry_lv.grid(row=2, column=1)
    entry_uh.grid(row=3, column=1)
    entry_us.grid(row=4, column=1)
    entry_uv.grid(row=5, column=1)

    tk.Button(dialog_root, text="Apply", command=apply_values).grid(row=6, column=0, columnspan=2)
    dialog_root.mainloop()

def start_hsv_window():
    threading.Thread(target=hsv_input_window, daemon=True).start()

def default_hsv():
    return (0, 50, 50), (179, 255, 255)

def default_rgb():
    return (0, 0, 0), (255, 255, 255)

def main():
    img_hsv = np.zeros((300, 900, 3), np.uint8)
    img_rgb = np.zeros((300, 900, 3), np.uint8)

    create_hsv_palette_window()
    create_rgb_palette_window()

    frame_delay = 1 / 30  # 30 FPS
    while True:
        start_time = time.time()
        if cv2.getWindowProperty("HSV Palette", cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty("RGB Palette", cv2.WND_PROP_VISIBLE) < 1:
            break

        (l_h, l_s, l_v), (u_h, u_s, u_v) = get_hsv_values()
        r, g, b = get_rgb_values()

        display_hsv_palette(img_hsv, l_h, l_s, l_v, u_h, u_s, u_v)
        display_rgb_palette(img_rgb, r, g, b)

        cv2.imshow("HSV Palette", img_hsv)
        cv2.imshow("RGB Palette", img_rgb)

        while not rgb_queue.empty():
            r, g, b = rgb_queue.get()
            cv2.setTrackbarPos("R", "RGB Palette", r)
            cv2.setTrackbarPos("G", "RGB Palette", g)
            cv2.setTrackbarPos("B", "RGB Palette", b)
        
        while not hsv_queue.empty():
            (lh, ls, lv), (uh, us, uv) = hsv_queue.get()
            cv2.setTrackbarPos("LH", "HSV Palette", lh)
            cv2.setTrackbarPos("LS", "HSV Palette", ls)
            cv2.setTrackbarPos("LV", "HSV Palette", lv)
            cv2.setTrackbarPos("UH", "HSV Palette", uh)
            cv2.setTrackbarPos("US", "HSV Palette", us)
            cv2.setTrackbarPos("UV", "HSV Palette", uv)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_images(img_hsv, img_rgb)
        elif key == ord('i'):
            start_rgb_window()
        elif key == ord('h'):
            start_hsv_window()

        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed_time))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[INFO] Program terminated by user.")
