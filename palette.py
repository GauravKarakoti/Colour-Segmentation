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
    cv2.imshow('Colour Segmentation',img)
    r=cv2.getTrackbarPos("R","Colour Segmentation")
    g=cv2.getTrackbarPos("G","Colour Segmentation")
    b=cv2.getTrackbarPos("B","Colour Segmentation")
    img[:]=[b,g,r]
    cv2.waitKey(1)
cv2.destroyAllWindows()