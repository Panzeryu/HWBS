from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input
from keras.layers.core import Activation, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras
import numpy as np
import os
import random

train_file_path = 'F:/HWBS/data/kaggle/train/'
val_file_path = 'F:/HWBS/data/kaggle/test/'

val_x = []
train_x = []
count = 0
batch_size = 32
# for dir, file, images in os.walk(train_file_path):
#     for image in images:
#         # print(image)
#         count += 1
#         fullname = os.path.join(dir, image)
#         if count % 5 == 0:
#             val_x.append(fullname)
#             train_x.append(fullname)


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


train_x = get_img_file(train_file_path)
print(len(train_x))
val_x = get_img_file(val_file_path)
print(len(val_x))

# print(len(train_x))
#
# for i, val in enumerate(train_x):
#     print(val)
#     if i == 10:
#         break


def get_image_label(image_paths):
    # print(image_paths)
    image_labels = []
    for image_path in image_paths:
        image_name = image_path.split('/')[-1]
        # print(image_name)
        if 'cat' in image_name:
            image_labels.append([1, 0])
            # print(0)
        else:
            image_labels.append([0, 1])
            # print(1)

        # if '1-' in image_name:
        #     image_labels.append(0)
        # elif '2-' in image_name:
        #     image_labels.append(1)
        # elif '3-' in image_name:
        #     image_labels.append(2)
        # elif '4-' in image_name:
        #     image_labels.append(3)
        # elif '5-' in image_name:
        #     image_labels.append(4)
        # elif '6-' in image_name:
        #     image_labels.append(5)
        # elif '7-' in image_name:
        #     image_labels.append(6)
        # elif '8-' in image_name:
        #     image_labels.append(7)
        # break
    return image_labels


# image_labels = get_image_label(train_x)
# for i, image_label in enumerate(image_labels):
#     print(image_label)
#     if i == 10:
#         break


# 读取图片
def load_batch_image(img_path, train_set=True, target_size=(150, 150)):
    im = load_img(img_path, target_size=target_size)
    if train_set:
        return img_to_array(im)  # converts image to numpy array
    else:
        return img_to_array(im) / 255.0


# 建立一个数据迭代器
def GET_DATASET_SHUFFLE(X_samples, batch_size, train_set=True):
    random.shuffle(X_samples)
    # for i, image_label in enumerate(X_samples):
    #     print(image_label)
    #     if i == 10:
    #         break

    batch_num = int(len(X_samples) / batch_size)
    max_len = batch_num * batch_size
    X_samples = np.array(X_samples[:max_len])
    y_samples = np.array(get_image_label(X_samples))

    print('X_samples.shape:', X_samples.shape)

    X_batches = np.split(X_samples, batch_num)
    # print(X_batches)
    # for x_batch in X_batches:
    #     print(x_batch)
    #     break
    y_batches = np.split(y_samples, batch_num)

    # for i, y_batch in y_batches:
    #     print('y_batch:', y_batch)
    #     if i == 10:
    #         break
    # print('y_batches:', y_batches)

    for i in range(len(X_batches)):
        if train_set:
            x = np.array(list(map(load_batch_image, X_batches[i], [True for _ in range(batch_size)])))
        else:
            x = np.array(list(map(load_batch_image, X_batches[i], [False for _ in range(batch_size)])))
        # print(x.shape)
        y = np.array(y_batches[i])
        yield x, y


# 搭建模型
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu',
#                  input_shape=(150, 150, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# print(model.summary())
#
# model.compile(loss='binary_crossentropy',
#               optimizer=RMSprop(lr=1e-5),
#               metrics=['acc'])
#
# model.fit_generator(
#     GET_DATASET_SHUFFLE(train_x, batch_size, True),
#     epochs=1,
#     steps_per_epoch=20 // batch_size,)
#
# predictions = model.predict_generator(GET_DATASET_SHUFFLE(train_x, batch_size, True), max_queue_size=10, steps=10)
# predictions = [np.argmax(p) for p in predictions]
# print(predictions)


model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, classes=2, input_shape=(150, 150, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit_generator(
    GET_DATASET_SHUFFLE(train_x, batch_size, True),
    epochs=1,
    steps_per_epoch=len(train_x) // batch_size,)

predictions = model.predict_generator(GET_DATASET_SHUFFLE(val_x, batch_size, True), max_queue_size=10, steps=10)
predictions = [np.argmax(p) for p in predictions]
print(predictions)
