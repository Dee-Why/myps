# myps：photo style modifier

use main.py to understand how this system works.

## 图片风格调整
按照短视频网站中 林生手机摄影 进行风格模仿
包括的滤镜有：
* 新海诚动漫风：曝光-25，对比度35，饱和度20，自然饱和30，锐化30，阴影22，色温-20
* 酷感氛围：光感-30，饱和度-30，锐化40，高光100，阴影-50，色温-30，色调100，亮度-50
* 高级情绪蓝：光感-20，对比度20，锐化40，高光20，色温-100

需要预制滤镜的
* 净白ins风：滤镜净白70，对比度20，饱和度20，锐化35，高光20，色温-15，色调-10
* 动漫奶油风： 滤镜奶杏，对比度25，饱和度20，锐化30，高光-30，色温30
* ins发光： ABG缤果70，光感-30，对比度30，饱和度20，高光30，色温-30


# 项目总结
PIL可以解决的部分：
* 对比度-contrast: 使用内置方法
* 锐化-sharpness: 使用内置方法
* 亮度-brightness: 使用内置方法
* 饱和度-color: 使用内置方法
* 曝光度（光感）-exposure: 将RGB图片映射到HSV色彩空间，调节V值
* 色相-hue: 将RGB图片映射到HSV色彩空间，调节H值

OpenCV可以解决的部分:
* 阴影-shadow: 通过百分比参数计算阈值，对阴影区进行统一增量，对明亮区进行梯度增亮（越亮的地方越倾向于不变化）
* 高光-highlight: 通过百分比参数计算阈值，利用内置方法梯度调低亮度
* 色温-color_temperature: 在保持整体图片曝光度不变的情况下，改变蓝色通道和（红，绿）色通道的比例
* 色调-color_tone:在保持整体图片曝光度不变的情况下，改变绿色通道和（红，蓝）色通道的比例

至此所有基本操作全部实现。具体参数与ps有所不同，需要人为进行调整。