from utils import read_image, save_image, pilImage_to_cvarray, cvarray_to_pilImage
from basic_pil import modify_brightness, modify_contrast, modify_sharpness, modify_saturation
from basic_pil import modify_hue, modify_exposure
from basic_cv import lightup_shadow, lower_highlight, modify_color_temperature, modify_color_tone

filter = {'exposure': 0,
           'contrast': 0,
           'sharpness': 0,
           'shadow': 0,
           'brightness': 0,
           'highlight': 0,
           'temperature': 0,
           'tone': 0,
           'saturation': 0,
           'hue': 0,
           }

filter1 = {'exposure': 10,
           'contrast': 20,
           'sharpness': 25,
           'shadow': 50,
           'brightness': -40,
           'highlight': 20,
           'temperature': -50,
           'tone': 50,
           'saturation': 10,
           'hue': 0,
           }

# 读入图片
img = read_image("pictures/001/PKU.jpg")


# 调整曝光度
img = modify_exposure(img, factor=10)
# 调整对比度
img = modify_contrast(img, factor=20)
# 调整锐度
img = modify_sharpness(img, factor=25)
# 阴影调整（factor正数增量，负数降暗）（luminance_threshold越小则将更大的部分划归为阴影，更小的部分划分为高光）
cvimg = pilImage_to_cvarray(img)
cvimg = lightup_shadow(cvimg, factor=50, threshold=50)
img = cvarray_to_pilImage(cvimg)
# 调整整体亮度
img = modify_brightness(img, factor=-40)
# 高光调整（factor正数增量，负数降暗）（luminance_threshold越高则将更大的部分划归为阴影，更小的部分划分为高光）
cvimg = pilImage_to_cvarray(img)
cvimg = lower_highlight(cvimg, factor=20, threshold=50)
img = cvarray_to_pilImage(cvimg)
# 调整色温（负蓝正黄）
cvimg = pilImage_to_cvarray(img)
cvimg = modify_color_temperature(cvimg, factor=-50)
img = cvarray_to_pilImage(cvimg)
# 调整色调（负绿正紫）
cvimg = pilImage_to_cvarray(img)
cvimg = modify_color_tone(cvimg, factor=50)
img = cvarray_to_pilImage(cvimg)
# 调整饱和度
img = modify_saturation(img, factor=10)
#
# """
# 注意上述调整的factor均应该属于 [-100,100]范围内
# 色相的factor调整则可以为任意数值，原因见HSV色彩空间中Hue（色相）的定义
# """
#
# 调整色相
img = modify_hue(img, factor=10)


# 保存处理后的图片
save_image(img, "pictures/output/PKU_0.png")

