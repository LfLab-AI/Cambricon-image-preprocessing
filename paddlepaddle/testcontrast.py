import os
import cv2
import numpy as np
from PIL import Image
from paddle.vision.transforms import functional as F
import timeit
import paddle
input_folder = '/work/mlu/dataset/ship/'
output_folder = '/work/mlu/lf/output/'
x = paddle.randn([730, 730])
y = paddle.randn([730, 730])
# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

start_time = timeit.default_timer()
time = 0
count = 0
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
   # image = Image.open(filepath)
    time1 = timeit.default_timer()
    img_resize = F.adjust_contrast(image, 2)
    res = paddle.matmul(x, y)

    time2 = timeit.default_timer()
    time += (time2-time1)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path,img_resize)
   # img_resize.save(output_path)
    count = count + 1

end_time = timeit.default_timer()
print("The number of images: ",count)
##print("The average time to process an image without I/O: {} ms".format(time * 1000/count))
print("The average time to process an image: {}ms".format((end_time-start_time)*1000/count))
