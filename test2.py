import numpy as np
np.random.seed(1337)  # for reproducibility
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import datetime
import os
import time


# 时间差计算函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


# LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        path = r'F:\HWBS\xinxi.txt'  # 文件路径
        self.savetotxt(self.losses, path)

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # self.loss_plot('epoch')

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("F:/HWBS/result.png")
        plt.show()

    def savetotxt(self, item, p):
        path = p
        f = open(path, 'w', encoding='utf-8')  # 以'w'方式打开文件
        for k, v in item:  # 遍历字典中的键值
            s2 = str(v)  # 把字典的值转换成字符型
            f.write(k + '\n')  # 键和值分行放，键在单数行，值在双数行
            f.write(s2 + '\n')
        f.close()  # 关闭文件


# class LossHistory(keras.callbacks.Callback):
#     # 函数开始时创建盛放loss与acc的容器
#     def on_train_begin(self, logs={}):
#         self.losses = {'batch': [], 'epoch': []}
#         self.accuracy = {'batch': [], 'epoch': []}
#         self.val_loss = {'batch': [], 'epoch': []}
#         self.val_acc = {'batch': [], 'epoch': []}
#
#     # 按照batch来进行追加数据
#     def on_batch_end(self, batch, logs={}):
#         # 每一个batch完成后向容器里面追加loss，acc
#         self.losses['batch'].append(logs.get('loss'))
#         self.accuracy['batch'].append(logs.get('acc'))
#         self.val_loss['batch'].append(logs.get('val_loss'))
#         self.val_acc['batch'].append(logs.get('val_acc'))
#         # 每五秒按照当前容器里的值来绘图
#         if int(time.time()) % 5 == 0:
#             self.draw_p(self.losses['batch'], 'loss', 'train_batch')
#             self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
#             self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
#             self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
#
#     def on_epoch_end(self, batch, logs={}):
#         # 每一个epoch完成后向容器里面追加loss，acc
#         self.losses['epoch'].append(logs.get('loss'))
#         self.accuracy['epoch'].append(logs.get('acc'))
#         self.val_loss['epoch'].append(logs.get('val_loss'))
#         self.val_acc['epoch'].append(logs.get('val_acc'))
#         # 每五秒按照当前容器里的值来绘图
#         if int(time.time()) % 5 == 0:
#             self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
#             self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
#             self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
#             self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
#
#     # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
#     def draw_p(self, lists, label, type):
#         plt.figure()
#         plt.plot(range(len(lists)), lists, 'r', label=label)
#         plt.ylabel(label)
#         plt.xlabel(type)
#         plt.legend(loc="upper right")
#         plt.savefig(type + '_' + label + '.jpg')
#         plt.show()
#
#     # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
#     # 所以这里的方法会在整个训练结束以后调用
#     def end_draw(self):
#         self.draw_p(self.losses['batch'], 'loss', 'train_batch')
#         self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
#         self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
#         self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
#         self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
#         self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
#         self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
#         self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')


# batch_size = 16
# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         # shear_range=0.2,
#         # zoom_range=0.2,
#         # horizontal_flip=True
#         )
#
# # this is the augmentation configuration we will use for testing:
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # this is a generator that will read pictures found in
# train_generator = train_datagen.flow_from_directory(
#       #  '/Users/morningstarwang/Documents/Projects/SHL_Challenge_2019_LTB/data/torso/train',  # this is the target directory
#       #   '/public/lhy/data/Challenge2019/challenge-2019-train_torso/train/Torso/fftnew',
#       #   'F:/HWBS/picture',
#         'F:/HWBS/data/kaggle/train',
#         target_size=(100, 100),  # all images will be resized to 150x150
#         batch_size=batch_size,
#         save_to_dir=r'F:\HWBS\pic_train_result',
#         class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
#
# # this is a similar generator, for validation data
# validation_generator = test_datagen.flow_from_directory(
#         #'/Users/morningstarwang/Documents/Projects/SHL_Challenge_2019_LTB/data/torso/validate',
#         # '/public/lhy/data/Challenge2019/challenge-2019-validate_all/validate/Torso/fftnew',
#         # 'F:/HWBS/picture',
#         'F:/HWBS/data/kaggle/validate1',
#         target_size=(100, 100),
#         batch_size=batch_size,
#         class_mode='categorical')


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28, 1)/255.
X_test = X_test.reshape(-1, 28, 28, 1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(batch_input_shape=(None, 28, 28, 1), filters=25, kernel_size=5, strides=1, padding='same',
                        ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(16, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(9, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(Convolution2D(25, 5, strides=1, padding='same',))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(25, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(25, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(Convolution2D(25, 5, strides=1, padding='same',))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(25, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(10, (1, 1), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
adam = Adam(lr=1e-5)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = LossHistory()
checkpoint = ModelCheckpoint("20190619.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
startdate = datetime.datetime.now()  # 获取当前时间
startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式

print('Training ------------')
# Another way to train the model
# hlist = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint, history])

input_tensor = Input(shape=(28, 28, 1))
model = keras.applications.resnet50.ResNet50(include_top=False, weights=None)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint, history])

# hlist = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=3, validation_data=validation_generator, validation_steps=250, callbacks=[history])
# print(hlist.history)
# with open('cnn_nin_mnist_history.txt', 'w') as f:
#     f.write(str(hlist.history))

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

enddate = datetime.datetime.now()  # 获取当前时间
enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式

print('start date ', startdate)
print('end date ', enddate)
print('Time ', subtime(startdate, enddate)) # enddate > startdate

history.loss_plot('batch')
# history.end_draw()
print(history.accuracy)
print(history.losses)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


# model.summary()
#
#
#
#
# checkpoint = ModelCheckpoint("20190619.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
#
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000 // batch_size,
#         epochs=10000,
#         validation_data=validation_generator,
#         callbacks=callbacks_list,
#         validation_steps=800 // batch_size)
