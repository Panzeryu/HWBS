import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.utils.data as Data
import os
import sys
from shutil import copyfile
from PIL import Image
import h5py
import scipy
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
# import keras
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, LSTM, Input, GlobalAveragePooling2D, BatchNormalization
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.models import load_model
# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from tqdm import tqdm
from glob import glob, iglob
import cv2
import email.utils
from scipy.fftpack import fft,ifft
import linecache
import pandas


# EPOCH = 1
# BATCH_SIZE = 5
# LR = 0.001
# DOWNLOAD_MNIST = False   #是否下载数据集

# train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(),
#                                         download=False)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,)


# train_datafft = torchvision.datasets.ImageFolder(root='F:/HWBS/picture', transform=torchvision.transforms.ToTensor(),)   #注意修改地址
# train_loaderfft = Data.DataLoader(dataset=train_datafft, batch_size=BATCH_SIZE, shuffle=True,)
#
# #测试数导入代码尚未测试
# test_datafft = torchvision.datasets.ImageFolder(root='F:/HWBS/picture', transform=torchvision.transforms.ToTensor(),)   #注意修改地址
# test_loaderfft = Data.DataLoader(dataset=train_datafft, batch_size=BATCH_SIZE, shuffle=True,)
#
# for step, (batch_x, batch_y) in enumerate(test_loaderfft):
#     print(step)
#     print(batch_y)
#     print(batch_x)


# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# train_generator = train_datagen.flow_from_directory(
#         # 1
#       #  '/Users/morningstarwang/Documents/Projects/SHL_Challenge_2019_LTB/data/torso/train',  # this is the target directory
#         '/Users/morningstarwang/Desktop/torso/torso/train',
#         target_size=(48, 48),  # all images will be resized to 150x150
#         batch_size=BATCH_SIZE,
#         class_mode='categorical')


# con1 = nn.Conv2d(3, 6, 9, 1, 0)
# con2 = nn.Conv2d(6, 12, 9, 1, 0)
# con3 = nn.Conv2d(12, 24, 9, 1, 0)
# out0 = nn.AdaptiveAvgPool2d(1)
# out1 = nn.Linear(8 * 1 * 1, 10)
# out2 = nn.Softmax(dim=-1)
#
# inp = torch.randn(50, 3, 90, 120)
# x = con1(inp)
# print(x.size())
# x = con2(x)
# print(x.size())
# x = con3(x)
# print(x.size())
# x = out0(x)
# x = x.view(x.size(0), -1)
# print(x.size())
# x = out1(x)
# print(x.size())


# if torch.cuda.is_available():
#     ten1 = torch.randn(50, 3, 90, 120).cuda()
#     print('gpu')
# else:
#     ten1 = torch.randn(50, 3, 90, 120)
#     print('cpu')


# label_dict = {
#     "1": "1",
#     "2": "2",
#     "3": "3",
#     "4": "4",
#     "5": "5",
#     "6": "6",
#     "7": "7",
#     "8": "8",
# }
#
# # os.makedirs('F:\\HWBS\\picture\\1')
# # os.makedirs('F:\\HWBS\\picture\\2')
# # os.makedirs('F:\\HWBS\\picture\\3')
# # os.makedirs('F:\\HWBS\\picture\\4')
# # os.makedirs('F:\\HWBS\\picture\\5')
# # os.makedirs('F:\\HWBS\\picture\\6')
# # os.makedirs('F:\\HWBS\\picture\\7')
# # os.makedirs('F:\\HWBS\\picture\\8')
# os.makedirs('F:/HWBS/picture/9')
#
#
# files = os.listdir("F:/HWBS/picture")
# for file in files:
#     if '.jpg' in file:
#         marks = file.split("-")
#         print(marks)
#         # target_path = sys.path[0] + "/" + label_dict[marks[0]] + "/"
#         target_path = "F:/HWBS/picture" + "/" + label_dict[marks[0]] + "/"
#         print("target_path=%s" % target_path)
#         # copyfile(sys.path[0] + "/" + file, target_path + label_dict[marks[0]]+marks[1])
#         copyfile("F:/HWBS/picture" + "/" + file, target_path + label_dict[marks[0]] + marks[1])


# a = np.array((1, 2, 3))
# b = np.array([2, 3, 4])
# print(a)
# print(b)


# inp = Input(shape=(((28, 28, 1))))
# out_1 = Conv2D(batch_input_shape=(None, 1, 28, 28), filters=192, kernel_size=5, strides=1, padding='valid',
#                  data_format='channels_last', kernel_regularizer=keras.regularizers.l2(), kernel_initializer="he_normal")(inp)
# out_1 = BatchNormalization()(out_1)
# out_1 = Activation('relu')(out_1)
# print(out_1.shape)
# out_2 = Conv2D(160, (1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(), kernel_initializer="he_normal")(out_1)
# out_2 = BatchNormalization()(out_2)
# out_2 = Activation('relu')(out_2)
# print(out_2.shape)
# out_3 = Conv2D(96, (1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(), kernel_initializer="he_normal")(out_2)
# out_3 = BatchNormalization()(out_3)
# out_3 = Activation('relu')(out_3)
# print(out_3.shape)
# out_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(out_3)
# # out_4 = GlobalAveragePooling2D()(out_3)
# print(out_4.shape)


# img_sizey = 735  # 根据实际情况可修改669
# img_sizex = 848  # 根据实际情况可修改704
# train_path = 'F:/HWBS/picture/'  # 根据自己的目录修改
# nub_train = len(glob(train_path + '*/*.jpg'))
# # 先生成空array，然后往里填每张图片的array
# X_train = np.zeros((nub_train, img_sizey, img_sizex, 3), dtype=np.uint8)
# y_train = np.zeros((nub_train,), dtype=np.uint8)
#
# i = 0
# for img_path in tqdm(glob(train_path + '*/*.jpg')):
#     print(img_path)
#     marks = img_path.split("\\")
#     print(marks)
#     img = Image.open(img_path)
#     img = img.crop((142, 66, 990, 801))
#     # img = img.resize((img_size, img_size))  # 图片resize
#     arr = np.asarray(img)  # 图片转array
#     X_train[i, :, :, :] = arr  # 赋值
#
#     if img_path.split('\\')[-2] == '4':
#         y_train[i] = 4  # 4
#     elif img_path.split('\\')[-2] == '5':
#         y_train[i] = 5  # 5
#     elif img_path.split('\\')[-2] == '6':
#         y_train[i] = 6  # 6
#
#     i += 1
#
# print(i)
# print(y_train)
# fig, axes = plt.subplots(4, 4, figsize=(20, 20))
#
# for i, img in enumerate(X_train[:16]):
#     axes[i//4, i % 4].imshow(img)
#
# plt.show()


# # 参数：paths：要读取的图片路径列表img_rows:图片行img_cols:图片列color_type:图片颜色通道返回:imgs: 图片数组
# def get_im_cv2(paths, img_rows, img_cols, color_type=1, normalize=True):
#     # Load as grayscale
#     imgs = []
#     for path in paths:
#         if color_type == 1:
#             img = cv2.imread(path, 0)
#         elif color_type == 3:
#             img = cv2.imread(path)
#         # Reduce size
#         resized = cv2.resize(img, (img_cols, img_rows))
#         if normalize:
#             resized = resized.astype('float32')
#             resized /= 127.5
#             resized -= 1.
#             imgs.append(resized)
#     return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)
#
#
# # '''    参数：        X_train：所有图片路径列表        y_train: 所有图片对应的标签列表        batch_size:批次
# # img_w:图片宽        img_h:图片高        color_type:图片类型        is_argumentation:是否需要数据增强
# # 返回:     一个generator， x: 获取的批次图片  y: 获取的图片对应的标签    '''
# def get_train_batch(X_train, y_train, batch_size, img_w, img_h, color_type, is_argumentation):
#     while 1:
#         for i in range(0, len(X_train), batch_size):
#             x = get_im_cv2(X_train[i:i+batch_size], img_w, img_h, color_type)
#             y = y_train[i:i+batch_size]
#             # if is_argumentation:                 # 数据增强
#             #     x, y = img_augmentation(x, y)
#             # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
#             yield({'input': x}, {'output': y})


# model = load_model(model_name)
#     predictions = model.predict(
#         {'gyrx_input': gyr_v_x, 'gyry_input': gyr_v_x, 'gyrz_input': gyr_v_z,
#          'laccx_input': acc_v_x, 'laccy_input': acc_v_y, 'laccz_input': acc_v_z,
#          })
#
#     predictions = [np.argmax(p) for p in predictions]
#     lv = [np.argmax(t) for t in label_v]
#     # print("pred="+str(predictions))
#     # print("lv="+str(lv))菜单
#     accuracy = 0
#     cnf = [[0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0]]
#     for (p, t) in zip(predictions, lv):
#         cnf[t][p] += 1
#         if p == t:
#             accuracy += 1
#     accuracy /= float(len(predictions))
#     print('acc', accuracy)
#     print(np.array(cnf))
#     print('1: %f\n2: %f\n3: %f\n4: %f\n5: %f\n6: %f\n7: %f\n8: %f' % (cnf[0][0] / float(np.sum(cnf[0])),
#                                                                       cnf[1][1] / float(np.sum(cnf[1])),
#                                                                       cnf[2][2] / float(np.sum(cnf[2])),
#                                                                       cnf[3][3] / float(np.sum(cnf[3])),
#                                                                       cnf[4][4] / float(np.sum(cnf[4])),
#                                                                       cnf[5][5] / float(np.sum(cnf[5])),
#                                                                       cnf[6][6] / float(np.sum(cnf[6])),
#                                                                       cnf[7][7] / float(np.sum(cnf[7]))))
#
#
# if __name__ == '__main__':
#     main()
#
#
# mymodel = load_model('.h')
# predictions = model.predict_generator(data, max_queue_size=10, steps=10)
# predictions = [np.argmax(p) for p in predictions]
# print(predictions)


# def get_name_list(filepath):  # 获取各个类别的名字
#     pathDir = os.listdir(filepath)
#     out = []
#     for allDir in pathDir:
#         if os.path.isdir(os.path.join(filepath, allDir)):
#             print(allDir)
#             # child = allDir.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
#             out.append(allDir)
#     return out
#
#
# out = get_name_list('F:/HWBS/picture')
# print(out)


image_labels = np.array([1, 2, 1, 4, 5])
image_labels1 = np.array([1, 2, 3, 4, 5])
print(image_labels1)
print(sum(image_labels == image_labels1))

help(pandas.read_csv)