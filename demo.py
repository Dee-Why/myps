import cv2 as cv
import sys
from utils import read_image, show_image, save_image

img = cv.imread("pictures/001/zzy_dark.jpg", cv.IMREAD_UNCHANGED)
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("pictures/001/zzy_dark.png", img)
