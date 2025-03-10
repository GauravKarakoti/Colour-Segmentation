import cv2
import numpy as np
img=np.zeros((300,512,3),np.uint8)
# frame=cv2.imread("image1.webp",-1)
def nothing(x):
    return
cv2.namedWindow("Colour Segmentation")
cv2.createTrackbar("R","Colour Segmentation",0,255,nothing)
cv2.createTrackbar("G","Colour Segmentation",0,255,nothing)
cv2.createTrackbar("B","Colour Segmentation",0,255,nothing)
while(True):
    if cv2.getWindowProperty("Colour Segmentation", cv2.WND_PROP_VISIBLE) < 1:
        break

    r=cv2.getTrackbarPos("R","Colour Segmentation")
    g=cv2.getTrackbarPos("G","Colour Segmentation")
    b=cv2.getTrackbarPos("B","Colour Segmentation")
    img[:]=[b,g,r]

    color_img = np.full((300, 512, 3), (b, g, r), dtype=np.uint8)
    # Calculate the brightness
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)

    text = f"R={r}, G={g}, B={b}"
    cv2.putText(img,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,2)

    cv2.imshow("Colour Segmentation", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite("selected_color_with_text.png",img)
        cv2.imwrite("selected_color.png",color_img)
        print("Saved: 'selected_color.png' (without text) & 'selected_color_with_text.png' (with text)")

cv2.destroyAllWindows()