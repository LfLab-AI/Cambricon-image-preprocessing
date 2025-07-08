import os
import datetime
import numpy as np
import cv2
from PIL import Image

from pymlu import to_grayscale, pad, perspective, resize, flip

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        # img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

timeStamp0 = datetime.datetime.now()

ship_dir = '/home/sancog/mlu/dataset/ship'
sailboat_dir = '/home/sancog/mlu/dataset/sailboat'

# 加载图像
ship_images = load_images_from_folder(ship_dir)
sailboat_images = load_images_from_folder(sailboat_dir)

timeStamp1 = datetime.datetime.now()

# 创建标签
ship_labels = [0] * len(ship_images)
sailboat_labels = [1] * len(sailboat_images)

# 合并数据和标签
images = ship_images + sailboat_images
labels = ship_labels + sailboat_labels

timeStamp2 = datetime.datetime.now()

# Flip
tmp_imgs = []
for image in images:
    tmp_imgs.append(flip(image))

timeStamp3 = datetime.datetime.now()

# # Gray
# images = []
# for image in tmp_imgs:
#     images.append(to_grayscale(image, output_channels=3))

# Pad
images = []
for image in tmp_imgs:
    images.append(pad(image, padding=[100, 100, 100, 100], fill=[255, 255, 0]))

timeStamp4 = datetime.datetime.now()

# Resize
tmp_imgs = []
for image in images:
    tmp_imgs.append(resize(image, 512, 512))

timeStamp5 = datetime.datetime.now()

# 将数据转换为numpy数组并展平
images = np.array(tmp_imgs)
labels = np.array(labels)
n_samples, width, height, channels = images.shape
images = images.reshape(n_samples, -1)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

timeStamp6 = datetime.datetime.now()

# 训练
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, n_jobs=4)
rf_clf.fit(X_train,y_train)

timeStamp7 = datetime.datetime.now()

# 推理
score = rf_clf.score(X_test, y_test)

timeStamp8 = datetime.datetime.now()

# 输出
print("random forest")
print(f"score: {score}")
print(f"read images: {timeStamp1 - timeStamp0}")
print(f"flip: {timeStamp3 - timeStamp2}")
# print(f"gray: {timeStamp4 - timeStamp3}")
print(f"pad: {timeStamp4 - timeStamp3}")
print(f"resize: {timeStamp5 - timeStamp4}")
print(f"train: {timeStamp7 - timeStamp6}")
print(f"predict: {timeStamp8 - timeStamp7}")
print(f"total: {timeStamp8 - timeStamp0}")
