import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, MaxPool2D,\
    BatchNormalization,Activation,concatenate,add


def Conv_bn_relu(num_filters,
                 kernel_size,
                 batchnorm=True,
                 strides=(1, 1),
                 padding='same'):

    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer


def SEblock():
    def layer(input_tensor):
        x = input_tensor
        return x
    return layer

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list


def res2net_block(num_filters, slice_num):
    def layer(input_tensor):
        short_cut = input_tensor
        x = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(input_tensor)
        slice_list = slice_layer(x, slice_num, x.shape[-1])
        side = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
        z = concatenate([slice_list[0], side])   # for one and second stage
        for i in range(2, len(slice_list)):
            y = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
            side = y
            z = concatenate([z, y])
        z = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(z)
        out = concatenate([z, short_cut])
        return out
    return layer


def res2netFamily(input_tensor, layer_num=[3,4,6,3]):
    conv1 = input_tensor
    f_size_list = [64,128,256,512,1024]
    # --------->>> 第一层3个3*3卷积代替1个7*7卷积；512，512，3 -> 512,512,64
    for i in range(3):
        conv1 = Conv2D(f_size_list[0], 3, padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;512,512,64 -> 256,256,256
    pool1 = MaxPool2D()(conv1)
    conv2 = Conv2D(f_size_list[1], 1, padding='same')(pool1)

    # --------->>> 第一组分组卷积；
    # 256,256,256 -> 256,256,128 -> 256,256,128/32 -> 256,256,128 -> 256,256,256
    block = res2net_block(num_filters=f_size_list[1], slice_num=4)
    for i in range(layer_num[0]):
        conv2 = block(conv2)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;256,256,256 -> 128,128,512
    pool2 = MaxPool2D()(conv2)
    conv3 = Conv2D(f_size_list[2], 3, padding='same')(pool2)

    # --------->>> 第二组分组卷积；
    # 128,128,512 -> 128,128,256 -> 128,128,256/32 -> 128,128,256 -> 128,128,512
    block = res2net_block(num_filters=f_size_list[2], slice_num=8)
    for i in range(layer_num[1]):
        conv3 = block(conv3)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;128,128,512 -> 64,64,1024
    pool3 = MaxPool2D()(conv3)
    conv4 = Conv2D(f_size_list[3], 1, padding='same')(pool3)

    # --------->>> 第三组分组卷积；
    # 64,64,1024 -> 64,64,512 -> 64,64,512/32 -> 64,64,512 -> 64,64,1024
    block = res2net_block(num_filters=f_size_list[3], slice_num=8)
    for i in range(layer_num[2]):
        conv4 = block(conv4)

    # --------->>> 最大池化下采样，结合1*1卷积层改变通道数;64,64,1024 -> 32,32,2048
    pool4 = MaxPool2D()(conv4)
    conv5 = Conv2D(f_size_list[4], 1, padding='same')(pool4)

    # --------->>> 第四组分组卷积；
    # 32,32,2048 -> 32,32,1024 -> 32,32,1024/32 -> 32,32,1024 -> 32,32,2048
    block = res2net_block(num_filters=f_size_list[4], slice_num=8)
    for i in range(layer_num[3]):
        conv5 = block(conv5)
    return conv5


# 运行模型
def run_model(name):
    input_tensor = Input((512,512,3))
    if name == 'res2net50':
        net = res2netFamily(input_tensor)
        model = Model(input_tensor, net)
    elif name== 'res2net101':
        net = res2netFamily(input_tensor, layer_num=[3, 4, 23, 3])
        model = Model(input_tensor, net)
    elif name== 'res2net152':
        net = res2netFamily(input_tensor, layer_num=[3, 8, 36, 3])
        model = Model(input_tensor, net)
    else:
        raise ModuleNotFoundError('没有发现该模型')
    return model



if __name__ == '__main__':
    model = run_model('res2net50')
    model.summary()


# x = Input((256, 256, 256))
# print(x.shape)
# x_conv_nor = Conv_bn_relu(512, (3, 3))(x)
# print(x_conv_nor.shape)
# out = slice_layer(x_conv_nor, 8, 512)
# print(out)
# print(len(out))
# x = res2net_block(512, 8)(x_conv_nor)
# print(x.shape)
