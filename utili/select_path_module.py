# -*- coding:UTF-8 -*-
from tensorflow.keras.layers import*
import tensorflow as tf
import numpy as np


class Finally_Path(Layer):
    def __init__(self, parameter_initializer=tf.zeros_initializer(),
                 parameter_regularizer=None,
                 parameter_constraint=None,
                 **kwargs):
        super(Finally_Path, self).__init__(**kwargs)
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

    def exp_func(self, val1, val2):
        return tf.math.exp(val1) / (1 + tf.math.exp(val1) + tf.math.exp(val2))

    def call(self, *args):
        print('args', args)
        input_1 = args[0]
        input_2 = args[0]
        input_3 = args[0]
        print('input_1的维度', input_1)
        print('input_2的维度', input_2)
        print('input_3的维度', input_3)
        re_gamma = self.exp_func(self.gamma, self.alpha)
        re_alpha = self.exp_func(self.alpha, self.gamma)
        out = tf.keras.layers.Add()([input_1 * re_gamma,
                                     input_2 * re_alpha,
                                     input_3 * (1 - re_gamma - re_alpha)])
        out = Conv2D(input_3.shape[3], 1, activation='relu')(out)
        return out


if __name__ == '__main__':
    one = np.ones((1,10,10,9))
    print(one)
    finally_pool = Finally_Path()
    print(finally_pool(one, one, one))
    print(finally_pool(one, one, one))
    # print(finally_pool.__doc__)


