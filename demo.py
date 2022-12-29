from utils import read_image, save_image, pilImage_to_cvarray, cvarray_to_pilImage
from basic_pil import modify_brightness, modify_contrast, modify_sharpness, modify_saturation, modify_hue
from basic_cv import modify_shadow

# 读入图片
img = read_image("pictures/001/zzy_dark.jpg")
# # 调整对比度
# img = modify_contrast(img, 20)
# # 调整锐度
# img = modify_sharpness(img, 25)
# # 阴影提亮
# cvimg = pilImage_to_cvarray(img)
# cvimg = modify_shadow(cvimg, factor=90, shadow_percentile=90)
# img = cvarray_to_pilImage(cvimg)
# # 调整整体亮度
# img = modify_brightness(img, -10)
# # 调整饱和度
# img = modify_saturation(img, 10)
# 调整色调
img = modify_hue(img, 10)
# 保存处理后的图片
save_image(img, "pictures/output/hue0.png")

