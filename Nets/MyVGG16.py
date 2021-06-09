import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Subtract, \
    Conv2DTranspose, Input, MaxPooling2D, AveragePooling2D, Add
from tensorflow.keras.models import Model

from utili.multi_pool_module import Finally_Pool



class VGG16():
    def __init__(self, input_shape=(256,256,3)):
        self.inputs = Input(input_shape)
        self.pool_block()
        self.pool_func = Finally_Pool()

    def build(self):
        filters_num = 64
        self.conv1 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv1_1')(self.inputs)
        self.conv1 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv1_2')(self.conv1)
        self.pool1 = MaxPooling2D(name='pool1')(self.conv1)
        self.pool1 = tf.concat([self.maxPool1, self.avePool1, self.pool1], axis=-1)
        self.pool1 = self.pool_func(self.pool1)
        self.print_shape(self.conv1, self.pool1)

        filters_num = filters_num * 2
        self.conv2 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv2_1')(self.pool1)
        self.conv2 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv2_2')(self.conv2)
        self.pool2 = MaxPooling2D(name='pool2')(self.conv2)
        self.pool2 = tf.concat([self.maxPool2, self.avePool2, self.pool2], axis=-1)
        self.pool2 = self.pool_func(self.pool2)
        self.print_shape(self.conv2, self.pool2)

        filters_num = filters_num * 2
        self.conv3 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv3_1')(self.pool2)
        self.conv3 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv3_2')(self.conv3)
        self.conv3 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv3_3')(self.conv3)
        self.pool3 = MaxPooling2D(name='pool3')(self.conv3)
        self.pool3 = tf.concat([self.maxPool3, self.avePool3, self.pool3], axis=-1)
        self.pool3 = self.pool_func(self.pool3)
        self.print_shape(self.conv3, self.pool3)

        filters_num = filters_num * 2
        self.conv4 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv4_1')(self.pool3)
        self.conv4 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv4_2')(self.conv4)
        self.conv4 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv4_3')(self.conv4)
        self.pool4 = MaxPooling2D(name='pool4')(self.conv4)
        self.pool4 = tf.concat([self.maxPool4, self.avePool4, self.pool4], axis=-1)
        self.pool4 = self.pool_func(self.pool4)
        self.print_shape(self.conv4, self.pool4)

        self.conv5 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv5_1')(self.pool4)
        self.conv5 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv5_2')(self.conv5)
        self.conv5 = Conv2D(filters=filters_num,
                            kernel_size=3,
                            padding='same',
                            name='conv5_3')(self.conv5)
        self.print_shape(self.conv5, self.conv5)
        # model = Model(self.inputs, [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])
        return self.conv1, self.conv2, self.conv3, self.conv4, self.conv5
        # return model

    def pool_block(self):
        self.maxPool1 = MaxPooling2D(name='maxPool1')(self.inputs)
        self.maxPool2 = MaxPooling2D(name='maxPool2')(self.maxPool1)
        self.maxPool3 = MaxPooling2D(name='maxPool3')(self.maxPool2)
        self.maxPool4 = MaxPooling2D(name='maxPool4')(self.maxPool3)

        self.avePool1 = AveragePooling2D(name='avePool1')(self.inputs)
        self.avePool2 = AveragePooling2D(name='avePool2')(self.avePool1)
        self.avePool3 = AveragePooling2D(name='avePool3')(self.avePool2)
        self.avePool4 = AveragePooling2D(name='avePool4')(self.avePool3)

    def print_shape(self, in1, in2):
        print('in1',in1.shape)
        print('in2',in2.shape)