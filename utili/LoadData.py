import cv2 as cv
import numpy as np
import os, glob, random


class My_LoadData():
    def __init__(self):
        pass

    def loadData(self, img_path, isLab=False):
        # 加载数据：图像+标签
        img = cv.imread(img_path)
        if isLab:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
            img = np.array(img, dtype='float') / 255
        else:
            img = np.array(img, dtype='float') / 127.5 - 1
        return img

    def hori_flip_img(self, arr):
        return np.flip(arr, axis=1)

    def vert_flip_img(self, arr):
        return np.flip(arr, axis=0)

    def get_train_val_fileName(self, img_dir, rate=0.1):
        all_name = os.listdir(img_dir)
        random.shuffle(all_name)
        split_num = int(len(all_name)*rate)
        train_name = all_name[:-split_num]
        val_name = all_name[-split_num:]
        return train_name, val_name

    def generate_train_val_data(self, img_dir, lab_dir, batch_size, data_name):
        while True:
            img_data = []
            lab_data = []
            batch = 0
            for i in data_name:
                batch += 1
                img = self.loadData(img_dir + i)
                lab = self.loadData(lab_dir + i, isLab=True)
                img_data.append(img)
                lab_data.append(lab)
                if batch % batch_size == 0:
                    yield np.array(img_data), np.array(lab_data)
                    img_data = []
                    lab_data = []


if __name__ == '__main__':
    genarate = My_LoadData()
    train_name, val_name = genarate.get_train_val_fileName(
        'C:/Users/MYY/Desktop/MyCode/data/trains/')  # C:\Users\MYY\Desktop\MyCode\data\trains
    train_generate = genarate.generate_train_val_data(img_dir='C:/Users/MYY/Desktop/MyCode/data/trains/',
                                       lab_dir='C:/Users/MYY/Desktop/MyCode/data/labels/',
                                       batch_size=2,
                                       data_name=train_name)

    val_generate = genarate.generate_train_val_data(img_dir='C:/Users/MYY/Desktop/MyCode/data/trains/',
                                     lab_dir='C:/Users/MYY/Desktop/MyCode/data/labels/',
                                     batch_size=2,
                                     data_name=val_name)
    a = next(train_generate)
    b = next(val_generate)
    print(a)
    print(a[0].shape, a[1].shape)
    print(b[0].shape, b[1].shape)