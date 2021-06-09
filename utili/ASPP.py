# -*- coding:UTF-8 -*-


"""
文件说明：
    
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import*


class Aspp(tf.keras.layers.Layer):
    def __init__(self, depth=3):
        super(Aspp, self).__init__()
        self.mean = tf.keras.layers.MaxPool2D()
        self.conv = tf.keras.layers.Conv2D(depth, 1, padding='same',
                                                    kernel_initializer='he_normal')
        self.atrous_block1 = tf.keras.layers.Conv2D(depth, 1, padding='same',
                                                    kernel_initializer='he_normal')
        self.atrous_block6 = tf.keras.layers.Conv2D(depth, 1, dilation_rate=6, padding='same',
                                                    kernel_initializer='he_normal')
        self.atrous_block12 = tf.keras.layers.Conv2D(depth, 1, dilation_rate=12, padding='same',
                                                    kernel_initializer='he_normal')
        self.atrous_block18 = tf.keras.layers.Conv2D(depth, 1, dilation_rate=18, padding='same',
                                                    kernel_initializer='he_normal')
        self.conv_1x1_output = tf.keras.layers.Conv2D(depth*5, 1, padding='same',
                                           kernel_initializer='he_normal')

    def call(self, inputs, training=None, mask=None):
        x1 = self.mean(inputs)
        x1 = self.conv(x1)
        x1 = tf.keras.layers.UpSampling2D(2,2)(x1)
        print('x1 = ', x1)
        x2 = self.atrous_block1(inputs)
        x3 = self.atrous_block6(inputs)
        x4 = self.atrous_block12(inputs)
        x5 = self.atrous_block18(inputs)
        print('x5 = ', x5)
        x = tf.keras.layers.Concatenate([x1,x2,x3,x4,x5], axis=-1)
        x = self.conv_1x1_output(x)
        print('x = ', x)
        return x



class AdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = np.array(output_size)
        # def build(self, input_shape):
        #     super(AdaptiveAvgPool2D, self).build(input_shape)
        def call(self, x):
            input_size = [x.shape[1], x.shape[2]]
            stride = np.floor((input_size / self.output_size))
            kernel_size = x.shape[1:3] - (self.output_size - 1) * stride
            kernel_size = tuple(kernel_size)
            out = tf.nn.avg_pool2d(x, ksize=kernel_size, strides=stride, padding='VALID')
            return out


def psp_modele(input_x, filters_nums=512, k_size=1):
    # 金字塔池化 Pyramid Scene Parsing Network
    _, _, c, _ = input_x.shape

    poolsize = [8, 4, 2, 1]
    print(input_x.shape)
    # 6
    x_c1 = AveragePooling2D(pool_size=c // poolsize[0], name='ave_c1')(input_x)
    x_c1 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c1')(x_c1)
    x_c1 = BatchNormalization(momentum=0.95, axis=-1)(x_c1)
    x_c1 = Activation(activation='relu')(x_c1)
    x_c1 = UpSampling2D(size=(c // poolsize[0], c // poolsize[0]), name='up_c1')(x_c1)
    print(x_c1.shape)

    # 3
    x_c2 = AveragePooling2D(pool_size=c // poolsize[1], name='ave_c2')(input_x)
    x_c2 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c2')(x_c2)
    x_c2 = BatchNormalization(momentum=0.95, axis=-1)(x_c2)
    x_c2 = Activation(activation='relu')(x_c2)
    x_c2 = UpSampling2D(size=(c // poolsize[1], c // poolsize[1]), name='up_c2')(x_c2)
    print(x_c2.shape)

    # 2
    x_c3 = AveragePooling2D(pool_size=c // poolsize[2], name='ave_c3')(input_x)
    x_c3 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c3')(x_c3)
    x_c3 = BatchNormalization(momentum=0.95, axis=-1)(x_c3)
    x_c3 = Activation(activation='relu')(x_c3)
    x_c3 = UpSampling2D(size=(c // poolsize[2], c // poolsize[2]), name='up_c3')(x_c3)
    print(x_c3.shape)

    # 1
    x_c4 = GlobalAveragePooling2D(name='glob1')(input_x)
    x_c4 = tf.reshape(x_c4, (-1, 1, 1, filters_nums))
    x_c4 = Conv2D(filters=filters_nums, kernel_size=k_size, strides=1, padding='same', name='conv_c4')(x_c4)
    x_c4 = BatchNormalization(momentum=0.95, axis=-1)(x_c4)
    x_c4 = Activation(activation='relu')(x_c4)
    x_c4 = UpSampling2D(size=(c, c), name='up_c4')(x_c4)
    print(x_c4.shape)

    x = Concatenate(axis=-1, name='concat')([input_x, x_c1, x_c2, x_c3, x_c4])
    x = Conv2D(filters=filters_nums, kernel_size=3, name='conv_c6', padding='same')(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='relu')(x)
    return x



def conv_bn_relu(input_tensor, f_size, k_size, dilation_rate=1, is_relu=False):
    conv = Conv2D(filters=f_size, kernel_size=k_size, dilation_rate=dilation_rate,kernel_initializer='ones', padding='SAME')(input_tensor)
    out = BatchNormalization()(conv)
    if is_relu:
        out = Activation('relu')(out)
    return out


def RFB(input_tensor):
    _, h, w, c = input_tensor.shape

    x1 = conv_bn_relu(input_tensor, c / 8, 1, is_relu=True)
    x1 = conv_bn_relu(x1, c/4, 3)

    x2 = conv_bn_relu(input_tensor, c / 8, 1, is_relu=True)
    x2 = conv_bn_relu(x2, c / 8, 3)
    x2 = conv_bn_relu(x2, c/4, 3, dilation_rate=3)

    x3 = conv_bn_relu(input_tensor, c / 8, 1, is_relu=True)
    x3 = conv_bn_relu(x3, c/8, 5)
    x3 = conv_bn_relu(x3, c/4, 3, dilation_rate=5)

    out = Concatenate()([x1, x2, x3])
    out = conv_bn_relu(out, c, 1)

    x4 = conv_bn_relu(input_tensor, c, 1)

    out = out*0.1 + x4
    out = Activation('relu')(out)
    return out







if __name__ == '__main__':
    arr = np.arange(0,108).reshape(1,6, 6, 3).astype(np.float32)
    # print(arr)
    poolx = tf.keras.layers.AveragePooling2D(pool_size=2)(arr)
    print(poolx.shape)
    up1 = UpSampling2D(size=(2, 2))(poolx)
    print(up1.shape)
#     # aspp = Aspp()
#     # adpool = AdaptiveAvgPool2D(3)
#     # print('*** adpool')
#     # print(adpool.call(arr))
#     #
#     # print('*** aspp')
#     # aspp_arr = aspp.call(arr)
#     #
#     # print(aspp_arr)
#     # print(aspp_arr.shape)
#     #
#     # psp = psp_modele(arr, 3)
#     # print('psp', psp)

    # one = np.ones((1,64,64,64)).astype(np.float32)
    # print(RFB(one))
