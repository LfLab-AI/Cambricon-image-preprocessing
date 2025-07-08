import cv2
import datetime
import timeit
import os
from pymlu import to_grayscale, pad, perspective, resize, flip, adjustContrast

input_folder = "/home/sancog/mlu/dataset/ship"
output_folder = "/home/sancog/mlu/lf/output"

time = 0
count = 0
start_time = timeit.default_timer()
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
    time1 = timeit.default_timer()
    
    new_img = adjustContrast(image,2.0)

    time2 = timeit.default_timer()
    time += (time2-time1)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path,new_img)
    count = count + 1

end_time = timeit.default_timer()
print("The number of images: ",count)
print("The average time to process an image: {}ms".format((end_time-start_time)*1000/(count)))
# img_path = "/home/sancog/mlu/dataset/ship/"
# img_nums = 100
# src_img = []
# for i in range(img_nums):
#     src_img.append(cv2.imread(f"{img_path}{i}.jpg"))
# start = datetime.datetime.now()
# for i in range(img_nums):
#     dst_img = perspective(src_img[i], startpoints, endpoints)
# end = datetime.datetime.now()
# print(f"100 images: {(end - start) / img_nums}")

# img_path = "/home/sancog/mlu/dataset/ship/"
# save_path = "/home/sancog/mlu/gsj/image/dst.jpg"
# img_nums = 100
# start = datetime.datetime.now()
# for i in range(img_nums):
#     src_img = cv2.imread(f"{img_path}{i}.jpg")
#     # dst_img = perspective(src_img, startpoints, endpoints)
#     # dst_img = to_grayscale(src_img)
#     dst_img = pad(src_img, padding=[100, 100, 100, 100], fill=[255, 255, 0])
#     cv2.imwrite(save_path, dst_img)
# end = datetime.datetime.now()
# print(f"100 images: {(end - start) / img_nums}")
