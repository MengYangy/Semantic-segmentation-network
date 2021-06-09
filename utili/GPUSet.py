import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def setGpu():
    '''
    功能：设置GPU
    :return:
    '''
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('GPU:', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    setGpu()