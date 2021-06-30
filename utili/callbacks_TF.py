import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import \
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
import tensorflow as tf
import numpy as np
import math, random


log_dir = './'
#-------------------------------------------------------------------------------#
#   训练参数的设置
#   logging表示tensorboard的保存地址
#   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
#   reduce_lr用于设置学习率下降的方式
#   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
#-------------------------------------------------------------------------------#
checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
tensorboard = TensorBoard(log_dir=log_dir)


class DataSet(tf.keras.utils.Sequence):
    def __init__(self, img_path, lab_path, batch_size):     # 必要函数，初始化参数
        super(DataSet, self).__init__()
        self.img_path = img_path
        self.lab_path = lab_path
        self.batch_size = batch_size
        self.img_nums = len(os.listdir(self.img_path))
        self.soft_list = np.arange(0, self.img_nums)

        self.load_data()

    def __len__(self):      # 必要函数，确定每一个EPOCH需要多少步
        return math.ceil(self.img_nums / self.batch_size)

    def __getitem__(self, item):    # 必要函数，在每个EPOCH内，依次返回一个batch_size的数据，item指当前是第几个batch
        # 封装数据
        return self.img_arr[self.batch_size * item : self.batch_size * (item + 1)], \
               self.lab_arr[self.batch_size * item : self.batch_size * (item + 1)]

    def on_epoch_end(self):         # 可选函数，在每个EPOCH结束时要做的事情，比如重新打乱数据
        random.shuffle(self.soft_list)

    def load_data(self):            # 自定义函数，读取数据、数据增强等 可在这里进行。
        self.img_arr = []
        self.lab_arr = []
        for i in np.array(os.listdir(self.img_path))[self.soft_list]:
            self.img_arr.append(os.path.join(self.img_path, i))
            self.lab_arr.append(os.path.join(self.lab_path, i))


if __name__ == '__main__':
    dataset = DataSet(img_path=r'D:\dataset\my_test\image',
                      lab_path=r'D:\dataset\my_test\label',
                      batch_size=2)

    for idx, i in enumerate(dataset):
        print('第{}个batch，数据为：{}'.format(idx, i))