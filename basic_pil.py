from PIL import Image, ImageEnhance
import numpy as np

def modify_saturation(image, factor):
    # 这里认为用户输入的是 -100 到 100 之间的数
    assert -100 < factor <= 100
    factor = 1 + factor / 100
    enhancer = ImageEnhance.Color(image)
    im_output = enhancer.enhance(factor)
    return im_output


def modify_brightness(image, factor):
    # 这里认为用户输入的是 -100 到 100 之间的数
    assert -100 < factor <= 100
    factor = 1 + factor / 100
    enhancer = ImageEnhance.Brightness(image)
    im_output = enhancer.enhance(factor)
    return im_output


def modify_sharpness(image, factor):
    assert -100 < factor <= 100
    factor = 1 + factor / 100
    enhancer = ImageEnhance.Sharpness(image)
    im_output = enhancer.enhance(factor)
    return im_output


def modify_contrast(image, factor):
    assert -100 < factor <= 100
    factor = 1 + factor / 100
    enhancer = ImageEnhance.Contrast(image)
    im_output = enhancer.enhance(factor)
    return im_output


def modify_hue(img, factor):
    hsv_img = img.convert('HSV')
    hsv = np.array(hsv_img)
    hsv[..., 0] = (hsv[..., 0]+factor) % 360
    new_img = Image.fromarray(hsv, 'HSV')
    return new_img.convert('RGB')