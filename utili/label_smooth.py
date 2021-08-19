import numpy as np


def label_smoothing(num_classes, class_ind):
    '''
    @num_classes    :   分类数目
    @class_ind      :   当前标签所属类别
    '''
    onehot = np.zeros(num_classes, dtype=np.float)
    onehot[class_ind] = 1.0
    uniform_distribution = np.full(num_classes, 1.0 / num_classes)
    deta = 0.01
    smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
    return smooth_onehot

if __name__ == '__main__':
    print(label_smoothing(5, 2))
    # [0.002 0.002 0.992 0.002 0.002]