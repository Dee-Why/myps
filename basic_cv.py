import os
import sys
import cv2 as cv
import numpy as np

class PSShadowHighlight:
    """
    色阶调整
    """
    def __init__(self, image):
        self.shadows_light = 50

        img = image.astype(float)/255.0

        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB

        # 阴影选取
        luminance = (1-srcGray) * (1-srcGray)

        self.maskThreshold = np.mean(luminance)
        mask = luminance > self.maskThreshold
        imgRow = np.size(img, 0)
        imgCol = np.size(img, 1)

        self.rgbMask = np.zeros([imgRow, imgCol, 3], dtype=bool)
        self.rgbMask[:, :, 0] = self.rgbMask[:, :, 1] = self.rgbMask[:, :, 2] = mask

        self.rgbLuminance = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.rgbLuminance[:, :, 0] = self.rgbLuminance[:, :, 1] = self.rgbLuminance[:, :, 2] = luminance

        self.midtonesRate = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.brightnessRate = np.zeros([imgRow, imgCol, 3], dtype=float)

    def adjust_image(self, img):
        maxRate = 4
        brightness = (self.shadows_light / 100.0 - 0.0001) / maxRate
        midtones = 1 + maxRate * brightness

        self.midtonesRate[self.rgbMask] = midtones
        self.midtonesRate[~self.rgbMask] = (midtones-1.0) / self.maskThreshold * self.rgbLuminance[~self.rgbMask] + 1.0

        self.brightnessRate[self.rgbMask] = brightness
        self.brightnessRate[~self.rgbMask] = (1 / self.maskThreshold * self.rgbLuminance[~self.rgbMask]) * brightness

        outImg = 255 * np.power(img / 255.0, 1.0 / self.midtonesRate) * (1.0 / (1 - self.brightnessRate))

        img = outImg
        img[img < 0] = 0
        img[img > 255] = 255

        img = img.astype(np.uint8)
        return img


def ps_shadow_highlight_adjust_and_save_img(psSH, origin_image):
    psSH.shadows_light = 50
    image = psSH.adjust_image(origin_image)
    cv.imwrite('py_sh_out_01.png', image)


def ps_shadow_highlight_adjust(path):
    """
    阴影提亮调整
    """
    origin_image = cv.imread(path)
    psSH = PSShadowHighlight(origin_image)
    image = psSH.adjust_image(origin_image)
    cv.imwrite('test_output.png', image)


if __name__ == '__main__':
    """
    usage:
    python basic_cv.py test.jpg
    """
    if len(sys.argv) == 1:
        print("参数错误，没有检测到图片路径")
        sys.exit(-1)
    img_path = sys.argv[1]
    print("img_path Params:", img_path)
    ps_shadow_highlight_adjust(img_path)
