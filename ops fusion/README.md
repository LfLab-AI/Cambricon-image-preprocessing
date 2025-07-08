## 文件说明

fusion.mlu: BangC 代码实现细节

pymlu_fusion.py: python 接口

## 使用方法

cncc 版本: 4.8.2

直接 make，生成 mlu.so，与 pymlu_fusion.py 放在同一目录下

目前提供了 gray_hflip、resize_hflip 两种融合算子

使用示例：

```
import cv2
from pymlu_fusion import gray_hflip, resize_hflip

img = cv2.imread(img_path)

# new_img = gray_hflip(img)
new_img = resize_hflip(img, 512, 512)
```