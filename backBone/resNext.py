import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, \
    Input, Activation, Add
from tensorflow.keras.models import Model


'''
使用类写
'''
class ResNextFamily():
    def __init__(self, input_shape=(512,512,3)):
        self.inputs = Input(input_shape)
        self.f_size = 64
        self.split_size = 32

    def identity_block(self, input_tensor, filter_size):
        input_tensor_list = tf.split(input_tensor, num_or_size_splits=self.split_size, axis=3)
        filter_x = filter_size // self.split_size

        def conv_func(input_tensor):
            convx = Conv2D(filter_x, 1, padding='same', kernel_initializer='he_normal')(input_tensor)
            convx = BatchNormalization()(convx)
            convx = Activation('relu')(convx)
            convx = Conv2D(filter_x, 3, padding='same', kernel_initializer='he_normal')(convx)
            convx = BatchNormalization()(convx)
            convx = Activation('relu')(convx)
            return convx
        out_list = list(map(conv_func, input_tensor_list))
        out = tf.concat(out_list, axis=-1)
        out = Conv2D(filter_size, 1, padding='same', kernel_initializer='he_normal')(out)
        out = BatchNormalization()(out)
        out = Add()([out, input_tensor])
        out = Activation('relu')(out)
        return out

    def conv_func(self, input_tensor, filter_size):
        convx = Conv2D(filter_size, 1, padding='same', kernel_initializer='he_normal')(input_tensor)
        convx = BatchNormalization()(convx)
        convx = Activation('relu')(convx)
        convx = Conv2D(filter_size, 3, padding='same', kernel_initializer='he_normal')(convx)
        convx = BatchNormalization()(convx)
        convx = Activation('relu')(convx)
        return convx

    def conv_bn(self, filter_size, x):
        convx = Conv2D(filter_size, 3, padding='same')(x)
        bnx = BatchNormalization()(convx)
        acx = Activation('relu')(bnx)
        return acx

    def resNeXt_50(self, input_tensor):
        filters = self.f_size
        '''
        第一层
        '''
        conv1 = self.conv_bn(filters, input_tensor)
        conv1 = self.conv_bn(filters, conv1)
        conv1 = self.conv_bn(filters, conv1)
        pool1 = MaxPool2D()(conv1)

        '''
        第二层
        '''
        conv2 = self.conv_bn(filters * 4, pool1)
        for _ in range(3):
            conv2 = self.identity_block(conv2, filters * 4)
        pool2 = MaxPool2D()(conv2)

        '''
        第三层
        '''
        conv3 = self.conv_bn(filters * 8, pool2)
        for _ in range(4):
            conv3 = self.identity_block(conv3, filters * 8)
        pool3 = MaxPool2D()(conv3)

        '''
        第四层
        '''
        conv4 = self.conv_bn(filters * 16, pool3)
        for _ in range(6):
            conv4 = self.identity_block(conv4, filters * 16)
        pool4 = MaxPool2D()(conv4)

        '''
        第五层
        '''
        conv5 = self.conv_bn(filters * 32, pool4)
        for _ in range(3):
            conv5 = self.identity_block(conv5, filters * 32)
        return conv5

    def run_model(self, name):
        if name == 'res18':
            net = self.resNeXt_50(self.inputs)
        elif name == 'res34':

            net = self.resNeXt_50(self.inputs)
        elif name == 'res50':
            net = self.resNeXt_50(self.inputs)
        elif name == 'res101':
            net = self.resNeXt_50(self.inputs)
        elif name == 'res152':
            net = self.resNeXt_50(self.inputs)
        else:
            raise ValueError('This network does not exist.')

        model = Model(self.inputs, net)
        return model


'''
    用函数定义写
'''
# resNext家族，resNext50，resNext101，resNext152
def resNext(input_tensor, layer_num=[3,4,6,3]):
    conv1 = input_tensor
    f_size = 64

    # --------->>> 第一层3个3*3卷积代替1个7*7卷积；512，512，3 -> 512,512,64
    for i in range(3):
        conv1 = Conv2D(f_size, 3, padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;512,512,64 -> 256,256,256
    pool1 = MaxPool2D()(conv1)
    conv2 = Conv2D(f_size*4, 1,  padding='same')(pool1)

    # --------->>> 第一组分组卷积；
    # 256,256,256 -> 256,256,128 -> 256,256,128/32 -> 256,256,128 -> 256,256,256
    for i in range(layer_num[0]):
        conv2 = split_conv_block(conv2)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;256,256,256 -> 128,128,512
    pool2 = MaxPool2D()(conv2)
    conv3 = Conv2D(f_size * 8, 3,  padding='same')(pool2)

    # --------->>> 第二组分组卷积；
    # 128,128,512 -> 128,128,256 -> 128,128,256/32 -> 128,128,256 -> 128,128,512
    for i in range(layer_num[1]):
        conv3 = split_conv_block(conv3)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;128,128,512 -> 64,64,1024
    pool3 = MaxPool2D()(conv3)
    conv4 = Conv2D(f_size * 16, 1,  padding='same')(pool3)

    # --------->>> 第三组分组卷积；
    # 64,64,1024 -> 64,64,512 -> 64,64,512/32 -> 64,64,512 -> 64,64,1024
    for i in range(layer_num[2]):
        conv4 = split_conv_block(conv4)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;64,64,1024 -> 32,32,2048
    pool4 = MaxPool2D()(conv4)
    conv5 = Conv2D(f_size * 32, 1,  padding='same')(pool4)

    # --------->>> 第四组分组卷积；
    # 32,32,2048 -> 32,32,1024 -> 32,32,1024/32 -> 32,32,1024 -> 32,32,2048
    for i in range(layer_num[3]):
        conv5 = split_conv_block(conv5)
    return conv1, conv2, conv3, conv4, conv5


# Conv2D + BatchNormalization + Activation
def conv_block(input_tensor, f_size, k_size):
    if k_size == 3 or k_size == 5:
        convx = Conv2D(f_size, k_size, padding='same')(input_tensor)
    elif k_size == 1:
        convx = Conv2D(f_size, k_size)(input_tensor)
    else:
        raise ValueError('k_size 输入有误')
    convx = BatchNormalization()(convx)
    convx = Activation('relu')(convx)
    return convx


# 分组卷积模块
# 输入经过1*1卷积后，通道数减少一半
# 在通道维度分成32组，每一组分别进行3*3卷积，把32组结果concat
# 使用1*1卷积恢复原始通道数
def split_conv_block(input_tensor):
    _,_,_,c = input_tensor.shape
    convx = conv_block(input_tensor, c//2, 1)
    tensor_list = tf.split(convx, 32, axis=-1)
    def conv_func(input_tensor):
        convx = conv_block(input_tensor, c//2//32, 3)
        return convx

    out_list = list(map(conv_func, tensor_list))
    out = tf.concat(out_list, axis=-1)
    out = Conv2D(c, 1, padding='same', kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    out = Add()([out, input_tensor])
    out = Activation('relu')(out)
    return out


# 运行模型
def run_model(name):
    input_tensor = Input((512,512,3))
    if name == 'resnext50':
        net = resNext(input_tensor)
        model = Model(input_tensor, net[-1])
    elif name== 'resnext101':
        net = resNext(input_tensor, layer_num=[3, 4, 23, 3])
        model = Model(input_tensor, net[-1])
    elif name== 'resnext152':
        net = resNext(input_tensor, layer_num=[3, 8, 36, 3])
        model = Model(input_tensor, net[-1])
    else:
        raise ModuleNotFoundError('没有发现该模型')
    return model



if __name__ == '__main__':
    # resnext = ResNextFamily()
    # model = resnext.run_model('res152')

    model = run_model('resnext101')
    model.summary()

