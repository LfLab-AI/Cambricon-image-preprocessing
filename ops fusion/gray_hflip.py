from PIL import Image
import cv2
import datetime
from paddle.vision.transforms import functional as F

############################################ single image ############################################

img_path = "../image/src.jpg"
save_path = "../image/dst.jpg"
img = cv2.imread(img_path)
start = datetime.datetime.now()
new_img = F.hflip(F.to_grayscale(img))
end = datetime.datetime.now()
cv2.imwrite(save_path, new_img)
print(f"single image: {end - start}")

############################################ 100 image ############################################

# img_path = "/work/mlu/dataset/ship/"
# img_nums = 100
# src_img = []
# for i in range(img_nums):
#     src_img.append(cv2.imread(f"{img_path}{i}.jpg"))
# start = datetime.datetime.now()
# for i in range(img_nums):
#     dst_img = F.hflip(F.to_grayscale(src_img[i]))
# end = datetime.datetime.now()
# print(f"100 images: {(end - start) / img_nums}")

############################################ 100 image with IO #####################################

# img_path = "/work/mlu/dataset/ship/"
# save_path = "../image/dst.jpg"
# img_nums = 100
# start = datetime.datetime.now()
# for i in range(img_nums):
#     src_img = cv2.imread(f"{img_path}{i}.jpg")
#     dst_img = F.hflip(F.to_grayscale(src_img))
#     cv2.imwrite(save_path, dst_img)
# end = datetime.datetime.now()
# print(f"average images: {(end - start) / img_nums}")

############################################ pillow: 100 image with IO #############################

# img_path = "/work/mlu/dataset/ship/"
# save_path = "../image/dst.jpg"
# img_nums = 100
# start = datetime.datetime.now()
# for i in range(img_nums):
#     src_img = Image.open(f"{img_path}{i}.jpg")
#     dst_img = F.hflip(F.to_grayscale(src_img))
#     dst_img.save(save_path)
# end = datetime.datetime.now()
# print(f"100 images(include read and write): {(end - start) / img_nums}")
