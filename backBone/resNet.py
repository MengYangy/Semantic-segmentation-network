import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, \
    Input, Activation, Add
from tensorflow.keras.models import Model


class ResNetFamily():
    def __init__(self, input_shape=(512,512,3)):
        self.inputs = Input(input_shape)
        self.f_size = 64

    def bn_conv_a(self, input_tensor, f_size, k_size, name, dila_rate=1):
        x = Conv2D(filters=f_size, kernel_size=k_size, padding='same',
                   name=name,dilation_rate=dila_rate,
                   kernel_initializer='he_normal')(input_tensor)
        x = BatchNormalization(name='{}_BN'.format(name))(x)
        x = Activation('relu', name='{}_AC'.format(name))(x)
        return x

    def res_block1(self, input_tensor, f_size, name):
        convx = self.bn_conv_a(input_tensor, f_size, 3, name='{}_1'.format(name))
        convx = self.bn_conv_a(convx, f_size, 3, name='{}_2'.format(name))
        out_tensor = Add(name='{}_add'.format(name))([input_tensor, convx])
        out_tensor = Activation('relu', name='{}_AC'.format(name))(out_tensor)
        return out_tensor

    def res_block2(self, input_tensor, f_size, name):
        conv1x1 = self.bn_conv_a(input_tensor, f_size//4, k_size=1, name='{}_1x1_1'.format(name))
        conv3x3 = self.bn_conv_a(conv1x1, f_size//4, k_size=3, name='{}_3x3_2'.format(name))
        conv1x1 = self.bn_conv_a(conv3x3, f_size, k_size=1, name='{}_1x1_3'.format(name))
        out_tensor = Add(name='{}_add'.format(name))([input_tensor, conv1x1])
        out_tensor = Activation('relu', name='{}_AC'.format(name))(out_tensor)
        return out_tensor

    def res18(self, input_tensor, Flag=False):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(2):
            conv2 = self.res_block1(conv2, f_size=f_size, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*2, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(2):
            conv3 = self.res_block1(conv3, f_size=2 * f_size, name='conv3_{}'.format(i))

        conv4 = Conv2D(f_size * 4, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(2):
            conv4 = self.res_block1(conv4, f_size=4 * f_size, name='conv4_{}'.format(i))

        if Flag:
            conv5 = Conv2D(f_size * 8, 1, strides=1, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i))

        else:
            conv5 = Conv2D(f_size * 8, 1, strides=2, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i))
        return conv1, conv2, conv3, conv4, conv5

    def res34(self, input_tensor, Flag=False):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(3):
            conv2 = self.res_block1(conv2, f_size=f_size, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*2, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(4):
            conv3 = self.res_block1(conv3, f_size=2 * f_size, name='conv3_{}'.format(i))

        conv4 = Conv2D(f_size * 4, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(6):
            conv4 = self.res_block1(conv4, f_size=4 * f_size, name='conv4_{}'.format(i))

        if Flag:
            conv5 = Conv2D(f_size * 8, 1, strides=1, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i))

        else:
            conv5 = Conv2D(f_size * 8, 1, strides=2, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i))
        return conv1, conv2, conv3, conv4, conv5

    def res50(self, input_tensor, Flag=False):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size*4, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(3):
            conv2 = self.res_block2(conv2, f_size=f_size*4, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*8, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(4):
            conv3 = self.res_block2(conv3, f_size=8 * f_size, name='conv3_{}'.format(i))

        conv4 = Conv2D(f_size * 16, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(6):
            conv4 = self.res_block2(conv4, f_size=16 * f_size, name='conv4_{}'.format(i))


        if Flag:
            conv5 = Conv2D(f_size * 32, 1, strides=1, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))

        else:
            conv5 = Conv2D(f_size * 32, 1, strides=2, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))
        return conv1, conv2, conv3, conv4, conv5

    def res101(self, input_tensor, Flag=False):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size*4, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(3):
            conv2 = self.res_block2(conv2, f_size=f_size*4, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*8, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(4):
            conv3 = self.res_block2(conv3, f_size=8 * f_size, name='conv3_{}'.format(i))

        conv4 = Conv2D(f_size * 16, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(23):
            conv4 = self.res_block2(conv4, f_size=16 * f_size, name='conv4_{}'.format(i))

        if Flag:
            conv5 = Conv2D(f_size * 32, 1, strides=1, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))

        else:
            conv5 = Conv2D(f_size * 32, 1, strides=2, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))
        return conv5, conv4, conv3, conv2, conv1

    def res152(self, input_tensor, Flag=False):
        f_size = self.f_size

        conv1 = self.bn_conv_a(input_tensor, f_size=f_size, k_size=3, name='conv1_1')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_2')
        conv1 = self.bn_conv_a(conv1, f_size=f_size, k_size=3, name='conv1_3')
        # pool1 = MaxPool2D()(conv1)
        conv2 = Conv2D(f_size*4, 1, strides=2, padding='same', name='pool1')(conv1)
        for i in range(3):
            conv2 = self.res_block2(conv2, f_size=f_size*4, name='conv2_{}'.format(i))

        conv3 = Conv2D(f_size*8, 1, strides=2, padding='same', name='pool2')(conv2)
        for i in range(8):
            conv3 = self.res_block2(conv3, f_size=8 * f_size, name='conv3_{}'.format(i))

        conv4 = Conv2D(f_size * 16, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(36):
            conv4 = self.res_block2(conv4, f_size=16 * f_size, name='conv4_{}'.format(i))

        if Flag:
            conv5 = Conv2D(f_size * 32, 1, strides=1, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))

        else:
            conv5 = Conv2D(f_size * 32, 1, strides=2, padding='same', name='pool4')(conv4)
            for i in range(3):
                conv5 = self.res_block2(conv5, f_size=32 * f_size, name='conv5_{}'.format(i))
        return conv5, conv4, conv3, conv2, conv1

    def run_model(self, name):
        if name == 'res18':
            net = self.res18(self.inputs)
        elif name == 'res34':

            net = self.res34(self.inputs)
        elif name == 'res50':
            net = self.res50(self.inputs)
        elif name == 'res101':
            net = self.res101(self.inputs)
        elif name == 'res152':
            net = self.res152(self.inputs)
        else:
            raise ValueError('This network does not exist.')

        model = Model(self.inputs, net[-1])
        return model


if __name__ == '__main__':
    resnet = ResNetFamily()
    model = resnet.run_model('res50')
    model.summary()
