import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import glob, time, os
import cv2 as cv
import numpy as np


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

    def res_block1(self, input_tensor, f_size, name, dila_rate=1):
        convx = self.bn_conv_a(input_tensor, f_size, 3, name='{}_1'.format(name), dila_rate=dila_rate)
        convx = self.bn_conv_a(convx, f_size, 3, name='{}_2'.format(name), dila_rate=dila_rate)
        out_tensor = Add(name='{}_add'.format(name))([input_tensor, convx])
        out_tensor = Activation('relu', name='{}_AC'.format(name))(out_tensor)
        return out_tensor

    def res34(self, input_tensor):
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
            conv3 = self.res_block1(conv3, f_size=2 * f_size, name='conv3_{}'.format(i), dila_rate=1)

        conv4 = Conv2D(f_size * 4, 1, strides=2, padding='same', name='pool3')(conv3)
        for i in range(6):
            conv4 = self.res_block1(conv4, f_size=4 * f_size, name='conv4_{}'.format(i), dila_rate=1)

        conv5 = Conv2D(f_size * 8, 1, strides=2, padding='same', name='pool4')(conv4)
        for i in range(3):
            conv5 = self.res_block1(conv5, f_size=8 * f_size, name='conv5_{}'.format(i), dila_rate=1)
        return [conv1, conv2, conv3, conv4, conv5]

    def feature_fusion(self, net):
        conv1, conv2, conv3, conv4, conv5 = net

        conv2, conv3 = self.low_to_high_feature(conv1, conv2, conv3)
        conv3, conv4 = self.low_to_high_feature(conv2, conv3, conv4)
        conv1 = self.attention_demo(conv1)
        conv2 = self.attention_demo(conv2)
        conv3 = self.attention_demo(conv3)
        conv4 = self.attention_demo(conv4)
        conv5 = self.attention_demo(conv5)

        up4 = self.upsame_feature(conv4, conv5, name='4', dila_rate=1)
        up3 = self.upsame_feature(conv3, up4, name='3', dila_rate=1)
        up2 = self.upsame_feature(conv2, up3, name='2', dila_rate=1)
        up1 = self.upsame_feature(conv1, up2, name='1', dila_rate=1)
        out_tensor = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(up1)
        out_tensor = Conv2D(2, 3, padding='same', activation='softmax', kernel_initializer='he_normal')(out_tensor)
        return out_tensor

    def attention_demo(self, inputs):
        gap=tf.keras.layers.GlobalAveragePooling2D()(inputs)
        in_filters=gap.shape[-1]

        fc1=tf.keras.layers.Dense(in_filters//2)(gap)
        fc1=tf.keras.layers.BatchNormalization()(fc1)
        fc1=tf.keras.layers.ReLU()(fc1)

        fc2=tf.keras.layers.Dense(in_filters)(fc1)
        fc2=tf.keras.layers.BatchNormalization()(fc2)
        fc2=tf.keras.layers.Activation('sigmoid')(fc2)

        out=tf.reshape(fc2,[-1,1,1,in_filters])
    #     in_filters是通道数目,shape用list表示
        out=tf.multiply(inputs,out)
        return  out


    def channel_gate(self, inputs, rate=16):
        channels = inputs.shape[-1]
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)

        fc1 = tf.keras.layers.Dense(channels // rate)(avg_pool)
        fc1 = tf.keras.layers.BatchNormalization()(fc1)
        fc1 = tf.keras.layers.Activation('relu')(fc1)

        fc2 = tf.keras.layers.Dense(channels // rate)(fc1)
        fc2 = tf.keras.layers.BatchNormalization()(fc2)
        fc2 = tf.keras.layers.Activation('relu')(fc2)

        fc3 = tf.keras.layers.Dense(channels)(fc2)

        return fc3

    def spatial_gate(self, inputs, rate=16, d=4):
        channels = inputs.shape[-1]

        conv = tf.keras.layers.Conv2D(channels // rate, 1)(inputs)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)

        conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)

        conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)

        conv = tf.keras.layers.Conv2D(1, 1)(conv)

        return conv

    def upsame_feature(self,low_f, high_f, name, dila_rate=1):
        high_f_up = Conv2DTranspose(low_f.shape[-1], 2, strides=(2, 2),
                                    activation = 'relu', padding = 'same')(high_f)
        out_f = tf.concat([low_f, high_f_up], axis=-1)
        out_f = Conv2D(low_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(out_f)
        out_f = self.res_block1(out_f, out_f.shape[-1], name='upsame_{}'.format(name), dila_rate=dila_rate)
        return out_f

    def low_to_high_feature(self, low_f, mid_f, high_f):
        low_f_1 = MaxPool2D()(low_f)
        low_f_2 = MaxPool2D(strides=4)(low_f)
        mid_f_1 = MaxPool2D()(mid_f)
        high_f = tf.concat([high_f, mid_f_1, low_f_2], axis=-1)
        high_out_f = Conv2D(high_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(high_f)
        mid_f = tf.concat([mid_f, low_f_1], axis=-1)
        mid_out_f = Conv2D(mid_f.shape[-1], 1, activation='relu', kernel_initializer='he_normal')(mid_f)
        return mid_out_f, high_out_f


    def run_model(self, name):
        if name == 'res34':
            net = self.res34(self.inputs)   # Trainable params: 22,910,272
            net = self.feature_fusion(net)
        else:
            raise ValueError('This network does not exist.')

        model = Model(self.inputs, net)
        return model




def myPredict():
    images = glob.glob('D:/dataset/val/image/*')
    stride = 512
    image_size = 512
    e = time.time()
    for im in range(len(images)):
        # im = -2
        name = (os.listdir('D:/dataset/val/image/')[im].split('.')[0])
        #     image=tf.io.read_file(images[im])
        #     image=tf.image.decode_png(image,channels=3)
        image = cv.imread(images[im])
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        h, w, c = image.shape
        #     print(h,w)
        if (h % stride != 0):
            padding_h = (h // stride + 1) * stride
        else:
            padding_h = (h // stride) * stride
        if (w % stride != 0):
            padding_w = (w // stride + 1) * stride
        else:
            padding_w = (w // stride) * stride
            #     print(padding_h,padding_w)

        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        #     print(padding_img.shape)

        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)

        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :3]
                pred_result = np.ones((image_size, image_size), np.int8)
                crop = tf.cast(crop, tf.float32)
                test_part = crop
                test_part = test_part / 127.5 - 1
                test_part = tf.expand_dims(test_part, axis=0)
                pred_part = model.predict(test_part)
                pred_part = tf.argmax(pred_part, axis=-1)
                pred_part = pred_part[..., tf.newaxis]
                pred_part = tf.squeeze(pred_part)
                pred_result = pred_part.numpy()
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred_result[:, :]
        cv.imwrite('D:/dataset/pred1/{}.png'.format(name), mask_whole[0:h, 0:w] * 255)
    #     print('完成第{}张图像预测'.format(im + 1))
    d = time.time()
    print('预测{}张图像共消耗时间为: {}秒，平均每张所需: {}秒'.format(len(images), d - e, (d - e) / len(images)))


if __name__ == '__main__':
    resnet = ResNetFamily()
    model = resnet.run_model('res34')
    # model.summary()
    model.load_weights(r'C:\Users\MYY\Downloads\net_weight3.h5')
    myPredict()
