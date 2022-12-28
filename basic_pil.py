from PIL import Image, ImageEnhance


def read_image(filename):
    im = Image.open(filename)
    return im


def save_image(image,filename):
    image.save(filename)


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
