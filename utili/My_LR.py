import tensorflow as tf


def lrfn(epoch, init_lr=0.05, attenuation_rate=0.5, attenuation_step=5):

    lr = init_lr
    lr = lr * attenuation_rate**(epoch//attenuation_step)
    return lr


class Mylearning_rate():
    def __init__(self):
        pass

    def fixed_step_attenuation(self,
                               epoch,
                               init_lr=0.05,
                               attenuation_rate=0.5,
                               attenuation_step=5):
        ''' Fixed step size decay
            :param epoch: 当前epoch
            :param init_lr: 初始学习率
            :param attenuation_rate: 衰减率
            :param attenuation_step: 衰减步长
            :return: init_lr * attenuation_rate**(epoch//attenuation_step)
            '''
        lr = init_lr
        lr = lr * attenuation_rate ** (epoch // attenuation_step)
        return lr

    def exponential_attenuation(self, epoch):
        # 指数衰减
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    def cos_attenuation(self, epoch, init_lr=0.05, period = 10):
        '''
        :param epoch: 当前epoch
        :param init_lr: 初始学习率
        :param period: 几步长衰减到0
        :return:
        '''
        num = 2 / init_lr
        lr = (tf.math.cos(epoch*3/period) + 1) / num
        if lr <0:
            raise ValueError('The learning rate is less than 0')
        return lr