import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import*
from tensorflow.keras.models import*
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(500).reshape(500,1).astype(np.float32)
# x
y = np.arange(500).reshape(500,1).astype(np.float32)+2.5
# y


class En_train_para(Layer):
    def __init__(self, parameter_initializer=tf.zeros_initializer(),
                 parameter_regularizer=None,
                 parameter_constraint=None,
                 **kwargs):
        super(En_train_para, self).__init__(**kwargs)
        self.gamma_initializer = parameter_initializer
        self.gamma_regularizer = parameter_regularizer
        self.gamma_constraint = parameter_constraint

        self.alpha_initializer = parameter_initializer
        self.alpha_regularizer = parameter_regularizer
        self.alpha_constraint = parameter_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.alpha = self.add_weight(shape=(1,),
                                     initializer=self.alpha_initializer,
                                     name='alpha',
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        out = self.gamma * inputs + self.alpha
        return out



den = En_train_para()
def net(in_shape=(1,)):
    input_tensor = Input(in_shape)
    outs = den(input_tensor)
    model = Model(input_tensor, outs)
    return model

model = net()
model.compile(optimizer='adam',loss='MSE')
model.fit(x=x, y=y, epochs=1000)
print(den.gamma.numpy()[0])
print(den.alpha.numpy()[0])



'''
# 自定义训练方法
https://cloud.tencent.com/developer/article/1788396
'''
# p_a = []
# p_b = []
# for i in range(10):
#
#     p_a.append(den.gamma.numpy()[0])
#     p_b.append(den.alpha.numpy()[0])
#
# # np.arange(100)
# plt.plot(np.arange(10),p_a)
# plt.show()