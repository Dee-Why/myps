import os
import sys
import cv2 as cv
import numpy as np


class PSShadow:
    """
    色阶调整 默认输入图片为opencv风格的np.array
    """

    def __init__(self, image, parameter=50, threshold_percentile=50):
        self.parameter = parameter
        self.threshold_percentile = threshold_percentile

        # 单位化图片
        img = image.astype(float) / 255.0

        # 取不同通道
        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]

        # 计算明度矩阵
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB
        shade_score = (1 - srcGray) * (1 - srcGray)

        self.maskThreshold = np.percentile(shade_score, self.threshold_percentile)
        # 将明度小于均值的部分视为阴影区，这里mask里1为阴影区，0为明亮区
        mask = shade_score > self.maskThreshold
        imgRow = np.size(img, 0)
        imgCol = np.size(img, 1)

        self.rgbMask = np.zeros([imgRow, imgCol, 3], dtype=bool)
        self.rgbMask[:, :, 0] = self.rgbMask[:, :, 1] = self.rgbMask[:, :, 2] = mask

        self.rgb_shade_score = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.rgb_shade_score[:, :, 0] = self.rgb_shade_score[:, :, 1] = self.rgb_shade_score[:, :, 2] = shade_score

        self.midtonesRate = np.zeros([imgRow, imgCol, 3], dtype=float)
        self.brightnessRate = np.zeros([imgRow, imgCol, 3], dtype=float)

    def adjust_image(self, img):
        maxRate = 0.25
        midTones = 1 + self.parameter / 100.0
        brightness = (self.parameter / 100.0 - 0.001) * maxRate

        # 阴影区固定增强， 明亮区
        self.midtonesRate[self.rgbMask] = midTones  # 这个是阴影区
        self.midtonesRate[~self.rgbMask] = (midTones - 1.0) * self.rgb_shade_score[
            ~self.rgbMask] / self.maskThreshold + 1.0

        self.brightnessRate[self.rgbMask] = brightness
        self.brightnessRate[~self.rgbMask] = (1 * self.rgb_shade_score[~self.rgbMask] / self.maskThreshold) * brightness

        outImg = 255 * np.power(img / 255.0, 1.0 / self.midtonesRate) * (1.0 / (1 - self.brightnessRate))

        img = outImg
        img[img < 0] = 0
        img[img > 255] = 255

        img = img.astype(np.uint8)
        return img


def lightup_shadow(cvimage, factor, threshold):
    psS = PSShadow(cvimage, factor, threshold)
    image = psS.adjust_image(cvimage)
    return image


class PSHighlight:
    """
    色阶调整 默认输入图片为opencv风格的np.array
    """

    def __init__(self, image, parameter=50, threshold_percentile=50):
        self.parameter = parameter
        self.threshold_percentile = threshold_percentile

        # 单位化图片
        img = image.astype(float) / 255.0

        # 取不同通道
        srcR = img[:, :, 2]
        srcG = img[:, :, 1]
        srcB = img[:, :, 0]

        # 计算明度矩阵
        srcGray = 0.299 * srcR + 0.587 * srcG + 0.114 * srcB
        shade_score = (1 - srcGray) * (1 - srcGray)

        self.maskThreshold = np.percentile(shade_score, self.threshold_percentile)
        # 将暗度小于均值的部分视为明亮区，这里mask里1为明亮区，0为阴影区
        mask = shade_score < self.maskThreshold
        self.int_mask = np.zeros(img.shape, dtype=np.uint8)
        self.int_mask[mask] = 255

    def adjust_image(self, img):
        alpha = self.parameter / 100.0
        beta = 2 * alpha
        # alpha，beta 共同決定高光消除後的模糊程度
        # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
        # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
        outImg = cv.illuminationChange(img, mask=self.int_mask, alpha=alpha, beta=beta)

        img = outImg.clip(0, 255)
        img = img.astype(np.uint8)
        return img


def lower_highlight(cvimage, factor, threshold):
    psH = PSHighlight(cvimage, factor, threshold)
    image = psH.adjust_image(cvimage)
    return image


def reduce_highlights(img, threshold_percentile=50):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 先轉成灰階處理
    maskThreshold = np.percentile(img_gray, threshold_percentile)
    ret, thresh = cv.threshold(img_gray, maskThreshold, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        img_zero[y:y + h, x:x + w] = 255
        mask = img_zero

    # alpha，beta 共同決定高光消除後的模糊程度
    # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
    # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
    result = cv.illuminationChange(img, mask, alpha=0.2, beta=0.4)

    return result


def modify_color_temperature(img, factor):
    imgB = img[:, :, 0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]

    # 1. 色温(b分量 与 yellow(rg)分量之比)
    bAve = cv.mean(imgB)[0] + factor
    gAve = cv.mean(imgG)[0]
    rAve = cv.mean(imgR)[0]

    aveGray = int((bAve + gAve + rAve) / 3)

    # 2. 計算各通道增益係數，並使用此係數計算結果,该组系数保持图片曝光度不变
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve

    imgb = np.floor((imgB * bCoef))  # 向下取整
    imgg = np.floor((imgG * gCoef))
    imgr = np.floor((imgR * rCoef))

    imgb[imgb > 255] = 255
    imgg[imgg > 255] = 255
    imgr[imgr > 255] = 255

    tuned_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8)

    return tuned_rgb


def modify_color_tone(img, factor):
    imgB = img[:, :, 0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]

    # 1. 色温(g分量 与 purple(rb)分量之比)
    bAve = cv.mean(imgB)[0]
    gAve = cv.mean(imgG)[0] + factor
    rAve = cv.mean(imgR)[0]

    aveGray = int((bAve + gAve + rAve) / 3)

    # 2. 計算各通道增益係數，並使用此係數計算結果,该组系数保持图片曝光度不变
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve

    imgb = np.floor((imgB * bCoef))  # 向下取整
    imgg = np.floor((imgG * gCoef))
    imgr = np.floor((imgR * rCoef))

    imgb[imgb > 255] = 255
    imgg[imgg > 255] = 255
    imgr[imgr > 255] = 255

    tuned_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8)

    return tuned_rgb

# if __name__ == '__main__':
#     """
#     usage:
#     python basic_cv.py test.jpg
#     """
#     if len(sys.argv) == 1:
#         print("参数错误，没有检测到图片路径")
#         sys.exit(-1)
#     img_path = sys.argv[1]
#     out_path = sys.argv[2]
#     print("img_path Params:", img_path, "out_path:", out_path)
#     ps_shadow_highlight_adjust(img_path, out_path)
