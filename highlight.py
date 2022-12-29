class PSShadowHighlight:
    """
    色阶调整 默认输入图片为opencv风格的np.array
    """
    def __init__(self, image, parameter=50, threshold_percentile=90):
        self.parameter = parameter
        self.threshold_percentile = threshold_percentile

        # 单位化图片
        img = image.astype(float)/255.0

        # 取不同通道
        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB

        # 计算明度矩阵
        luminance = (1-srcGray) * (1-srcGray)

        self.maskThreshold = np.percentile(luminance, self.threshold_percentile)
        # 将明度小于均值的部分视为阴影区，这里mask里1为明亮区，0为阴影区
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
        maxRate = 0.25
        midTones = 1 + self.parameter / 100.0
        brightness = (self.parameter / 100.0) * maxRate

        self.midtonesRate[self.rgbMask] = midTones
        self.midtonesRate[~self.rgbMask] = (midTones-1.0) / self.maskThreshold * self.rgbLuminance[~self.rgbMask] + 1.0

        self.brightnessRate[self.rgbMask] = brightness
        self.brightnessRate[~self.rgbMask] = (1 / self.maskThreshold * self.rgbLuminance[~self.rgbMask]) * brightness

        outImg = 255 * np.power(img / 255.0, 1.0 / self.midtonesRate) * (1.0 / (1 - self.brightnessRate))

        img = outImg
        img[img < 0] = 0
        img[img > 255] = 255

        img = img.astype(np.uint8)
        return img


def ps_shadow_highlight_adjust_and_save_img(psSH, origin_image):
    psSH.parameter = 50
    image = psSH.adjust_image(origin_image)
    cv.imwrite('py_sh_out_01.png', image)


def ps_shadow_highlight_adjust(path, outPath):
    """
    阴影提亮调整
    """
    origin_image = cv.imread(path)
    psSH = PSShadowHighlight(origin_image)
    image = psSH.adjust_image(origin_image)
    cv.imwrite(outPath, image)


def modify_shadow(cvimage, factor, shadow_percentile):
    psSH = PSShadowHighlight(cvimage, factor, shadow_percentile)
    image = psSH.adjust_image(cvimage)
    return image