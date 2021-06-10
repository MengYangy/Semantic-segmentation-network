import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def setGpu():
    '''
    功能：设置GPU
    :return:
    '''
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

    # # 设置GPU使用显存为4GB
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    # )

    # 单个物理GPU模拟多个GPUS
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

    # 指定训练GPU
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    print('GPU:', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    setGpu()