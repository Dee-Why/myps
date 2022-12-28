import cv2 as cv
import sys


def read_image(filename):
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)  # IMREAD_COLOR, IMREAD_GREYSCALE
    if img is None:
        sys.exit("Could not read the image.")
    return img


def show_image(img):
    cv.imshow("Display window", img)
    k = cv.waitKey(0)


def save_image(img, filename):
    cv.imwrite(filename, img)
