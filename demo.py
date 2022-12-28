import cv2 as cv
import sys
# from utils import read_image, show_image, save_image
from basic_pil import *

img = read_image("pictures/001/zzy_dark.jpg")

img = modify_contrast(img, 20)
img = modify_sharpness(img, 25)
img = modify_brightness(img, 30)

save_image(img, "pictures/output/c20s25b30.png")

