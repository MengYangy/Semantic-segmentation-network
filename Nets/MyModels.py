import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Subtract, \
    Conv2DTranspose, Input, MaxPooling2D, AveragePooling2D, Add
from tensorflow.keras.models import Model
from utili.select_path_module import Finally_Path
from Nets.MyVGG16 import VGG16
from backBone.resNet import ResNetFamily
from backBone.resNext import resNext


class Feature_Extract_Net():
    def __init__(self, input_shape=(512,512,3)):
        self.input1_tensor = Input(input_shape, name='input1')
        self.input_tensor = Input(shape=input_shape, name='input2')
        self.finally_path = Finally_Path()

    def feature_Extract(self, name=None):
        # 编码网络
        if name == 'vgg16' or name == None:
            net = VGG16(self.input_tensor)
            Feature_Extract_Model = Model(inputs=self.input_tensor,
                                          outputs=net.build())
            # Feature_Extract_Model.summary()

            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                Feature_Extract_Model(self.input1_tensor)
        elif name == 'res18':
            res = ResNetFamily()
            # 1、使用res50第四层
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                res.res18(self.input1_tensor)

        elif name == 'res34':
            res = ResNetFamily()
            # 1、使用res50第四层
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                res.res34(self.input1_tensor)
        elif name == 'res50':
            res = ResNetFamily()
            # 1、使用res50第四层
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                res.res50(self.input1_tensor)
        elif name == 'res101':
            res = ResNetFamily()
            # 1、使用res50第四层
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                res.res101(self.input1_tensor)
        elif name == 'res152':
            res = ResNetFamily()
            # 1、使用res50第四层
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                res.res152(self.input1_tensor)
        elif name == 'resnext50':
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                resNext(self.input1_tensor)

        elif name == 'resnext101':
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                resNext(self.input1_tensor, layer_num=[3, 4, 23, 3])

        else:
            feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X = \
                resNext(self.input1_tensor, layer_num=[3, 8, 36, 3])

        out = self.encode(feature_1_X, feature_2_X, feature_3_X, feature_4_X, feature_5_X)

        model = Model(inputs=self.input1_tensor,
                      outputs = out)
        return model

    def encode(self, l1, l2, l3, l4, l5):
        # 解码网络
        l1 = self.s_c_attention(l1)
        l2 = self.s_c_attention(l2)
        l3 = self.s_c_attention(l3)
        l4 = self.s_c_attention(l4)
        l5 = self.s_c_attention(l5)
        filters_num = 512
        l5 = Conv2DTranspose(filters=filters_num,
                             kernel_size=3,
                             strides=2,
                             padding='same',
                             activation='relu')(l5)

        l4_5 = tf.concat([l4, l5], axis=-1)
        l4 = Conv2D(filters=filters_num,
                    kernel_size=3,
                    padding='same',
                    activation='relu')(l4_5)
        l4 = Conv2DTranspose(filters=filters_num // 2,
                             kernel_size=3,
                             strides=2,
                             padding='same',
                             activation='relu')(l4)

        l3_4 = tf.concat([l3, l4], axis=-1)
        l3 = Conv2D(filters=filters_num // 2,
                    kernel_size=3,
                    padding='same',
                    activation='relu')(l3_4)
        l3 = Conv2DTranspose(filters=filters_num // 4,
                             kernel_size=3,
                             strides=2,
                             padding='same',
                             activation='relu')(l3)

        l2_3 = tf.concat([l2, l3], axis=-1)
        l2 = Conv2D(filters=filters_num // 4,
                    kernel_size=3,
                    padding='same',
                    activation='relu')(l2_3)
        l2 = Conv2DTranspose(filters=filters_num // 8,
                             kernel_size=3,
                             strides=2,
                             padding='same',
                             activation='relu')(l2)

        l1_2 = tf.concat([l1, l2], axis=-1)
        l1 = Conv2D(filters=filters_num // 8,
                    kernel_size=3,
                    padding='same',
                    activation='relu')(l1_2)
        l1 = Conv2D(2, (3, 3), activation='softmax', padding='same')(l1)
        return l1


    def s_c_attention(self, inputs):
        # 调用注意力机制
        return self.SpatialAttention(inputs)
        # return inputs

    def SpatialAttention(self, inputs):
        # 空间注意力机制
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        out = tf.concat([avg_out, max_out], axis=3)
        out = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same')(out)
        out = tf.nn.sigmoid(out)
        out = out + inputs
        return out