# -*- coding:UTF-8 -*-

"""
文件说明：
    常用的语义分割评价指标
    原定义：
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP )
        Recall = TP / (TP + FN)
        IoU = TP / (TP + FP + FN)
        MIoU  = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
        Dice = (2*TP)/(2*TP +FP +FN)
        F1 = 2*Precision*Recall/(Precision+Recall)
"""
import tensorflow as tf
import tensorflow.keras.backend as K


# -------------------->>>> 精确率（Precision）评价指标
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    return precision

def precision(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
#   (None,256,256,1)
    y_pred=tf.argmax(y_pred,axis=-1)
#   (None,256,256,2)=======>(None,256,256)
    y_pred=tf.cast(y_pred,tf.int32)
#   (None,256,256)
    y_pred=y_pred[...,tf.newaxis]
#   (None,256,256)========>(None,256,256,1)
    TP3=tf.reduce_sum(y_true*y_pred)
#     计算本来就是正样本，预测的结果也是正样本的个数
    TN3=tf.reduce_sum((1-y_true)*(1-y_pred))
#     计算本来就是负样本，预测的结果也是负样本的个数
    FP3=tf.reduce_sum((1-y_true)*(y_pred))
#     计算本来是负样本，但结果却被预测为了正样本的个数
    FN3=tf.reduce_sum(y_true*(1-(y_pred)))
#     计算本来是正样本，但结果却被预测成了负样本的个数
    TP3=tf.cast(TP3,tf.float32)
    FP3=tf.cast(FP3,tf.float32)
    precision=(TP3)/(TP3+FP3+K.epsilon())
    return precision


# ---------------->>>>  召回率（Recall）评价指标
def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall

def recall(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
#   (None,256,256,1)
    y_pred=tf.argmax(y_pred,axis=-1)
#   (None,256,256,2)=======>(None,256,256)
    y_pred=tf.cast(y_pred,tf.int32)
#   (None,256,256)
    y_pred=y_pred[...,tf.newaxis]
#   (None,256,256)========>(None,256,256,1)
    TP4=tf.reduce_sum(y_true*y_pred)
#     计算本来就是正样本，预测的结果也是正样本的个数
    TN4=tf.reduce_sum((1-y_true)*(1-y_pred))
#     计算本来就是负样本，预测的结果也是负样本的个数
    FP4=tf.reduce_sum((1-y_true)*(y_pred))
#     计算本来是负样本，但结果却被预测为了正样本的个数
    FN4=tf.reduce_sum(y_true*(1-(y_pred)))
#     计算本来是正样本，但结果却被预测成了负样本的个数
    recall=(TP4+1)/(TP4+FN4+1)
    return recall

# ------------->>>>>> F1-score评价指标
# 1
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score
# 2
def f1_score(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
#   (None,256,256,1)
    y_pred=tf.argmax(y_pred,axis=-1)
#   (None,256,256,2)=======>(None,256,256)
    y_pred=tf.cast(y_pred,tf.int32)
#   (None,256,256)
    y_pred=y_pred[...,tf.newaxis]
#   (None,256,256)========>(None,256,256,1)
    TP=tf.reduce_sum(y_true*y_pred)
#     计算本来就是正样本，预测的结果也是正样本的个数
    TN=tf.reduce_sum((1-y_true)*(1-y_pred))
#     计算本来就是负样本，预测的结果也是负样本的个数
    FP=tf.reduce_sum((1-y_true)*(y_pred))
#     计算本来是负样本，但结果却被预测为了正样本的个数
    FN=tf.reduce_sum(y_true*(1-(y_pred)))
#     计算本来是正样本，但结果却被预测成了负样本的个数
    precision=(TP+1)/(TP+FP+1)
    recall=(TP+1)/(TP+FN+1)
    F1score=2*precision*recall/(precision+recall)
    return F1score


# ------------->>>> Iou
def metric_iou(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    iou = TP / (TP + FP + FN)
    return iou

def iou(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
#   (None,256,256,1)
    y_pred=tf.argmax(y_pred,axis=-1)
#   (None,256,256,2)=======>(None,256,256)
    y_pred=tf.cast(y_pred,tf.int32)
#   (None,256,256)
    y_pred=y_pred[...,tf.newaxis]
#   (None,256,256)========>(None,256,256,1)
    TP5=tf.reduce_sum(y_true*y_pred)
#     计算本来就是正样本，预测的结果也是正样本的个数
    TN5=tf.reduce_sum((1-y_true)*(1-y_pred))
#     计算本来就是负样本，预测的结果也是负样本的个数
    FP5=tf.reduce_sum((1-y_true)*(y_pred))
#     计算本来是负样本，但结果却被预测为了正样本的个数
    FN5=tf.reduce_sum(y_true*(1-(y_pred)))
#     计算本来是正样本，但结果却被预测成了负样本的个数
    iou=(TP5+1)/(TP5+FN5+FP5+1)
    return iou


# ------------------->>> miou
def metric_miou(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    miou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    return miou

def aiou(y_true,y_pred):
    y_true=tf.cast(y_true,tf.int32)
#   (None,256,256,1)
    y_pred=tf.argmax(y_pred,axis=-1)
#   (None,256,256,2)=======>(None,256,256)
    y_pred=tf.cast(y_pred,tf.int32)
#   (None,256,256)
    y_pred=y_pred[...,tf.newaxis]
#   (None,256,256)========>(None,256,256,1)
    TP1=tf.reduce_sum(y_true*y_pred)
#     计算本来就是正样本，预测的结果也是正样本的个数
    TN1=tf.reduce_sum((1-y_true)*(1-y_pred))
#     计算本来就是负样本，预测的结果也是负样本的个数
    FP1=tf.reduce_sum((1-y_true)*(y_pred))
#     计算本来是负样本，但结果却被预测为了正样本的个数
    FN1=tf.reduce_sum(y_true*(1-(y_pred)))
#     计算本来是正样本，但结果却被预测成了负样本的个数
    iou1=(TP1+1)/(TP1+FN1+FP1+1)
    iou0=(TN1+1)/(TN1+FN1+FP1+1)
    return (iou1+iou0)/2


#Dice
def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#正确率Acc
def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    K.equal 两数相等为True,不相等为False
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))


def PA(y_true, y_pred):
    '''
    二分类正确率

    '''
    y_true = tf.cast(y_true, tf.int32)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    TP = tf.reduce_sum(tf.cast(y_true * y_pred, tf.int32))
    TN = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.int32))
    FP = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.int32))
    FN = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.int32))

    TP = tf.cast(TP, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    PA = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
    return PA

