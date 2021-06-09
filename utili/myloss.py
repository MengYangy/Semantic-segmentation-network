# -*- coding:UTF-8 -*-


"""
文件说明：
    
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


# 定义Dice损失函数
# dice = 1 - (2*y_true*y_pred)/(y_true+y_pred)
def dice_coef_loss(y_true, y_pred):
    y_truef = K.flatten(y_true)  # 将y_true拉为一维
    y_predf = K.flatten(y_pred)
    intersection = K.sum(y_truef * y_predf)
    return 1 - (2 * intersection + K.epsilon()) / (K.sum(y_truef) + K.sum(y_predf) + K.epsilon())


def ce_dice_loss(y_true, y_pred):
    # binary_crossentropy L=-[y_true * log(y_pred)+(1-y_true)*log(1 - y_pred)]
    ce_loss = binary_crossentropy(y_true, y_pred)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))  # 平方和
    dice_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))  # K.epsilon() = 1e-07 防止分母为零
    loss = ce_loss + dice_loss
    return loss

def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    loss = 1. - intersection / (union + K.epsilon())
    return loss

def ce_jaccard_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = - tf.math.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + jaccard_loss
    return loss


def tversky_loss(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return 1 - (true_pos + K.epsilon())/(true_pos + alpha * false_neg + (1-alpha) * false_pos + K.epsilon())


def focal_loss_sum(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def focal_loss_mean(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed