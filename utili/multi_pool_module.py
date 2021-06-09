# -*- coding:UTF-8 -*-

# author:MYY
# contact: test@test.com
# datetime:2021/3/11 10:55
# software: PyCharm


from tensorflow.keras.layers import*
import tensorflow as tf


class Finally_Pool(Layer):
    """
    功能说明：
        定义具有多支路的池化结构，
        1、直接对原始图像进行多次平均池化  input_1 : avepool
        2、直接对原始图像进行多次最大池化  input_2 : maxpool
        3、原始的卷积层后池化             input_3 : convpool
        三个分支总的比重为1
        finally_out = gamma * avepool + alpha * maxpool + (1-gamma-alpha) * convpool


        call函数未定义具体操作之前，该模块仅有2个参数：gamma 和 alpha
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_4 (InputLayer)         [(None, 512, 512, 3)]     0
        _________________________________________________________________
        finally__out_18 (Finally_Out (None, 512, 512, 3)       2
        =================================================================
        Total params: 2
        Trainable params: 2
        Non-trainable params: 0
        _________________________________________________________________

    输入格式：
        该模块是处理三个输入，在此之前，需要先把三个支路合并
        inputs = tf.concat([input_1,input_2,input_3], axis=-1)
            input_1 的 shape 应为4维 [-1，None, None, c1]
            input_2 的 shape 应为4维 [-1，None, None, c2]
            input_3 的 shape 应为4维 [-1，None, None, c3]
            inputs  的 shape 也为4维 [-1，None, None, c1 + c2 + c3]
        之后在 call 函数中进行拆分
            input_1 = inputs[:,:,:,:c1]
            input_2 = inputs[:,:,:,c1:c1+c2]
            input_3 = inputs[:,:,:,c1+c2:]
        最终 return input_1 * gamma + input_2 * alpha + input_3 * (1 - gamma - alpha)
    注意：
        三个分支总的比重为1，
        即：又需要满足 0 < gamma < 1, 0 < alpha < 1, 0 < gamma + alpha < 1
        定义一个公式： e^x1 / (1 + e^x1 + e^x2)
                tf.math.exp(val1) / (1 + tf.math.exp(val1) + tf.math.exp(val2))
    """
    def __init__(self, parameter_initializer=tf.zeros_initializer(),
                 parameter_regularizer=None,
                 parameter_constraint=None,
                 **kwargs):
        super(Finally_Pool, self).__init__(**kwargs)
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

    def call(self, inputs, **kwargs):
        input_1 = inputs[:,:,:,:3]
        input_2 = inputs[:,:,:,3:6]
        input_3 = inputs[:,:,:,6:]
        # print('input_1的维度', input_1.shape)
        # print('input_2的维度', input_2.shape)
        # print('input_3的维度', input_3.shape)
        re_gamma = self.exp_func(self.gamma, self.alpha)
        re_alpha = self.exp_func(self.alpha, self.gamma)
        out = tf.concat([input_1 * re_gamma,
                         input_2 * re_alpha,
                         input_3 * (1 - re_gamma - re_alpha)],
                        axis=3)
        out = Conv2D(input_3.shape[3], 1, activation='relu')(out)
        return out

if __name__ == '__main__':
    finally_pool = Finally_Pool()
    print(finally_pool.__doc__)


