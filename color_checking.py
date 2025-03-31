import cv2
import numpy as np

# Load the image
image_path = "images\\image7.webp" 
img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load the image. Check the file path.")
    exit(1)

# Convert to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Convert back to BGR (corrected display)
bgr_corrected = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# Stack images side by side for comparison
comparison = np.hstack((img, hsv_img, bgr_corrected))

# Show all images
cv2.imshow("Original | Incorrect HSV Display | Corrected HSV to BGR", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save images for further inspection
cv2.imwrite("original_image.png", img)
cv2.imwrite("incorrect_hsv_display.png", hsv_img)  # Incorrect display
cv2.imwrite("corrected_hsv_to_bgr.png", bgr_corrected)  # Corrected version

# Pick a pixel coordinate (modify as needed)
x, y = img.shape[1] // 2, img.shape[0] // 2

print("Original BGR:", img[y, x])
print("HSV:", hsv_img[y, x])
print("Converted BGR:", bgr_corrected[y, x])

