import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Model
from backBone.resNet import ResNetFamily
from utili.ASPP import psp_modele


def psp(input_shape=(512, 512, 3)):
    input_tensor = Input(input_shape)
    res = ResNetFamily()
    # 1、使用res50第四层
    net = res.res50(input_tensor, Flag=True)[-1]

    # 2、使用res50第五层
    # net = res.res50(input_tensor)
    # net = Conv2D(filters=1024, kernel_size=1, name='conv_c7')(net)
    # net = BatchNormalization(momentum=0.95, axis=-1)(net)
    # net = Activation(activation='relu')(net)

    x = psp_modele(net)
    x = Conv2D(filters=2, kernel_size=3, name='conv_c8', padding='same')(x)
    x = BatchNormalization(momentum=0.95, axis=-1)(x)
    x = Activation(activation='softmax')(x)
    x = UpSampling2D(size=(8, 8))(x)
    return Model(input_tensor, x)


if __name__ == '__main__':
    model = psp()
    model.summary()
