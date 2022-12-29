import cv2 as cv
import sys
import numpy as np
from PIL import Image


def read_image_cv(filename):
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)  # IMREAD_COLOR, IMREAD_GREYSCALE
    if img is None:
        sys.exit("Could not read the image.")
    return img

# def show_image_cv(img):
#     cv.imshow("Display window", img)
#     k = cv.waitKey(0)


def save_image_cv(img, filename):
    cv.imwrite(filename, img)


def cvarray_to_pilImage(opencv_array):
    color_coverted = cv.cvtColor(opencv_array, cv.COLOR_BGR2RGB)
    im_converted = Image.fromarray(color_coverted, mode="RGB")
    return im_converted


def pilImage_to_cvarray(pilImage):
    arr = np.asarray(pilImage)
    converted_arr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
    return converted_arr

