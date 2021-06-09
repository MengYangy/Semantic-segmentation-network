import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
import tensorflow as tf
import time, random, shutil
from multiprocessing import Process
from tqdm import tqdm
import skimage


'''
介绍      --->    数据增强方法
1、原始图像有4736张
2、使用随机上下翻转，设置阈值为0.3，随机产生一个0-1内的小数，大于阈值时，进行上下翻转
3、使用随机左右翻转，设置阈值为0.3，随机产生一个0-1内的小数，大于阈值时，进行左右翻转
4、随机尺度缩放+随机copy
    对标签进行分析: 若标签中建筑物的占比大于20%，不处理；小于20%记为待处理样本
    在待处理样本中: 若标签中建筑物的占比小于20%、大于7.5%记为优样本；
                 若标签中建筑物的占比小于7.5%记为劣样本；
        随机在优样本和劣样本中各取一张图像，进行随机尺度缩放，然后把处理过的优样本中的建筑物单独提取出来，copy到劣样本中。
'''

class MyProcess(Process):   # 多进程函数
    def __init__(self):
        super(MyProcess, self).__init__()

    def run(self) -> None:
        cv.imwrite('{}'.format(self.img_path), self.img)
        cv.imwrite('{}'.format(self.lab_path), self.lab)

    def getPara(self, img, lab, img_path, lab_path):
        self.img = img
        self.lab = lab
        self.img_path = img_path
        self.lab_path = lab_path


class Data_Enhance():
    def __init__(self):
        self.img_w = self.img_h = 512
        self.read_img_path = r'D:\dataset\train\image'
        self.read_lab_path = r'D:\dataset\train\label'
        self.save_img_path = r'D:\dataset\new_train1\image'
        self.save_lab_path = r'D:\dataset\new_train1\label'
        self.save_img_path_random_copy = r'D:\dataset\random_train\image'
        self.save_lab_path_random_copy = r'D:\dataset\random_train\label'
        self.test_img_path = r'D:\dataset\test\image'
        self.test_lab_path = r'D:\dataset\test\label'

        if not os.path.exists(self.read_img_path) or not os.path.exists(self.read_lab_path):
            raise FileNotFoundError('路径不存在')
        self.mkdirs_path(self.save_img_path)
        self.mkdirs_path(self.save_lab_path)
        self.mkdirs_path(self.save_img_path_random_copy)
        self.mkdirs_path(self.save_lab_path_random_copy)
        random.seed()
        self.ct = 0
        self.save_format = '.png'

    def run(self, use_process=True):
        img_names = os.listdir(self.read_img_path)
        for i in tqdm(range(len(img_names)), ncols = 80, desc='process', mininterval = 0.1):
        # for i in range(5):
            #
            img = cv.imread(os.path.join(self.read_img_path, img_names[i]))
            lab = cv.imread(os.path.join(self.read_lab_path, img_names[i]))
            img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + self.save_format)
            lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + self.save_format)
            self.start_process(img, lab, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机上下翻转
                img_1 = tf.image.flip_up_down(img).numpy()
                lab_1 = tf.image.flip_up_down(lab).numpy()
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_1' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_1' + self.save_format)
                self.start_process(img_1, lab_1, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机左右旋转
                img_2 = tf.image.flip_left_right(img).numpy()
                lab_2 = tf.image.flip_left_right(lab).numpy()
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_2' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_2' + self.save_format)
                self.start_process(img_2, lab_2, img_path, lab_path, use_process)

            if random.random() > 0.2:   # 随机0.6-2尺度缩放
                scale_factor = random.randint(6,20) / 10
                img_3, lab_3 = self.random_scale_resize(img, lab, scale_factor)
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_3' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_3' + self.save_format)
                self.start_process(img_3, lab_3, img_path, lab_path, use_process)

            if random.random() > 0.7:   # 随机 颜色变换
                img_4 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_4' + self.save_format)
                lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_4' + self.save_format)
                self.start_process(img_4, lab, img_path, lab_path, use_process)

            # if random.random() > 0.5:   # 随机增加噪声
            #     img_5 = skimage.util.random_noise(img, mode='s&p')
            #     img_path = os.path.join(self.save_img_path, img_names[i].split('.')[0] + '_5' + self.save_format)
            #     lab_path = os.path.join(self.save_lab_path, img_names[i].split('.')[0] + '_5' + self.save_format)
            #     self.start_process(img_5, lab, img_path, lab_path, use_process)

        print('处理完成！')

    def copy_func(self):    # 复制+重命名
        img_names = os.listdir(r'D:\dataset\test\label')
        new_img_p = r'D:\dataset\new_test\image'
        new_lab_p = r'D:\dataset\new_test\label'
        self.mkdirs_path(new_img_p)
        self.mkdirs_path(new_lab_p)
        for i in tqdm(range(len(img_names)), ncols = 80, desc='process', mininterval = 0.1):
            old_img_path = os.path.join(r'D:\dataset\test\image', img_names[i])
            old_lab_path = os.path.join(r'D:\dataset\test\label', img_names[i])
            if os.path.exists(old_img_path) and os.path.exists(old_lab_path):
                new_img_path = os.path.join(new_img_p,
                                            img_names[i].split('.')[0] + self.save_format)
                new_lab_path = os.path.join(new_lab_p,
                                            img_names[i].split('.')[0] + self.save_format)
                shutil.copy(old_img_path, new_img_path)
                shutil.copy(old_lab_path, new_lab_path)
        print('处理完成')

    def transform_format(self): # 转图片格式
        img_names = os.listdir(r'D:\dataset\test\label')
        new_img_p = r'D:\dataset\new_tests\image'
        new_lab_p = r'D:\dataset\new_tests\label'
        self.mkdirs_path(new_img_p)
        self.mkdirs_path(new_lab_p)
        for i in tqdm(range(len(img_names)), ncols=80, desc='process', mininterval=0.1):
            img_path = os.path.join(self.test_img_path, img_names[i])
            lab_path = os.path.join(self.test_lab_path, img_names[i])
            save_img_path = os.path.join(new_img_p, img_names[i].split('.')[0] + self.save_format)
            save_lab_path = os.path.join(new_lab_p, img_names[i].split('.')[0] + self.save_format)
            img = cv.imread(img_path)
            lab = cv.imread(lab_path)
            self.start_process(img, lab, save_img_path, save_lab_path, use_process=False)

    def random_copy(self, use_process=False):
        # 尺度缩放(0.8->1.2) + copy
        img_size = self.img_h * self.img_w
        img_names = os.listdir(self.save_img_path)
        good_lab = []   # 占比小于20%、大于7.5%
        bad_lab = []    # 占比小于7.5%
        best_lab = []   # 占比大于20%

        # for i in tqdm(range(30), ncols=80, desc='process', mininterval=0.1):
        for i in tqdm(range(len(img_names)), ncols=80, desc='process', mininterval=0.1):
            lab = cv.imread(os.path.join(self.save_lab_path, img_names[i]))
            if np.sum(lab==255) > img_size//4 and np.sum(lab==255) < img_size:
                good_lab.append(img_names[i])
            elif np.sum(lab==255) >= img_size:
                best_lab.append(img_names[i])
            else:
                bad_lab.append(img_names[i])

        times = 2   # 随机缩放与copy次数

        for t in range(times):
            # 第一次是0.7 - 1.2倍缩放， 第二次是0.5 - 1.7倍缩放
            scale_factor = 0.7**t
            random.shuffle(good_lab)
            random.shuffle(bad_lab)
            print('\ngood_lab', len(good_lab))
            print('bad_lab', len(bad_lab))
            print('best_lab', len(best_lab))

            fusion_num = min(len(bad_lab), len(good_lab))
            for j in tqdm(range(fusion_num), ncols=80, desc='process', mininterval=0.1):
                first_img = cv.imread(os.path.join(self.save_img_path, good_lab[j]))
                last_img = cv.imread(os.path.join(self.save_img_path, bad_lab[j]))
                first_lab = cv.imread(os.path.join(self.save_lab_path, good_lab[j]))
                last_lab = cv.imread(os.path.join(self.save_lab_path, bad_lab[j]))

                # 小尺度随机尺度缩放
                # ------------------------------------->>>>>>
                scale1 = random.randint(int(7*scale_factor + 0.5), int(12/scale_factor + 0.5)) / 10
                first_img, first_lab = self.random_scale_resize(first_img, first_lab, scale1)

                scale2 = random.randint(7, 12) / 10
                last_img, last_lab = self.random_scale_resize(last_img, last_lab, scale2)
                # -------------------------------------<<<<<<

                # 融合
                # ------------------------------------->>>>>>
                new_img, new_lab = self.two_img_fusion(first_img, last_img, first_lab, last_lab)
                img_path = os.path.join(self.save_img_path_random_copy, 'r_{}_'.format(t) + img_names[j].split('.')[0] + '_0' + self.save_format)
                lab_path = os.path.join(self.save_lab_path_random_copy, 'r_{}_'.format(t) + img_names[j].split('.')[0] + '_0' + self.save_format)
                self.start_process(new_img, new_lab, img_path, lab_path, use_process)

                # -------------------------------------<<<<<<
                # 用于查看融合效果
                # img_path = os.path.join(self.save_img_path_random_copy, 'r_' + img_names[j].split('.')[0] + '_1' + self.save_format)
                # lab_path = os.path.join(self.save_lab_path_random_copy, 'r_' + img_names[j].split('.')[0] + '_1' + self.save_format)
                # self.start_process(first_img, first_lab, img_path, lab_path, use_process)
                #
                # img_path = os.path.join(self.save_img_path_random_copy, 'r_' + img_names[j].split('.')[0] + '_2' + self.save_format)
                # lab_path = os.path.join(self.save_lab_path_random_copy, 'r_' + img_names[j].split('.')[0] + '_2' + self.save_format)
                # self.start_process(last_img, last_lab, img_path, lab_path, use_process)

    def random_scale_resize(self, img, lab, random_scale):  # 随机尺度缩放
        img_h, img_w, _ = img.shape
        n_h = int(img_h * random_scale)
        n_w = int(img_w * random_scale)
        x = (img_w - n_w) // 2
        y = (img_h - n_h) // 2
        image = cv.resize(img, (n_h, n_w))
        label = cv.resize(lab, (n_h, n_w))
        label = self.label_(label)
        # self.ct += 1
        # cv.imwrite('D:/dataset/{}_img.jpg'.format(self.ct), image)
        # cv.imwrite('D:/dataset/{}_lab.jpg'.format(self.ct), label)
        # print('img.shape', img.shape)
        # print('lab.shape', lab.shape)
        # print('image.shape', image.shape)
        # print('label.shape', label.shape)
        if random_scale < 1:
            new_image = np.ones((img_h,img_w,3)) * 128
            new_label = np.zeros((img_h,img_w,3))
            new_image[y:y+n_h, x:x+n_w, :] = image
            new_label[y:y+n_h, x:x+n_w, :] = label
        else:
            x = (n_w - img_w) // 2
            y = (n_h - img_h) // 2
            x,y = x-1, y-1
            x = np.maximum(x, 0)
            y = np.maximum(y, 0)
            # print('x={},y={}'.format(x,y))
            new_image = image[y:y+512,x:x+512,:]
            new_label = label[y:y+512,x:x+512,:]
        if 0.7 > random.random() >= 0.4:
            new_image = tf.image.flip_up_down(new_image).numpy()
            new_label = tf.image.flip_up_down(new_label).numpy()
        elif random.random() >= 0.7:
            new_image = tf.image.flip_left_right(new_image).numpy()
            new_label = tf.image.flip_left_right(new_label).numpy()
        return new_image, new_label

    def label_(self, lab):  # 标签resize后会，在建筑物边缘出现杂质，以125为阈值进行清除
        n_lab = np.where(lab > 125, 255, 0)
        return n_lab

    def two_img_fusion(self, img1, img2, lab1, lab2):
        # 两张图片融合， img1中建筑物占比大，img2中建筑物占比小，每张经过随机缩放后，img1 -> copy -> img2
        self.ct += 1
        quality = [int(cv.IMWRITE_PNG_COMPRESSION), 0]
        # cv.imwrite('D:/dataset/fusion/img/img_{}_1.png'.format(self.ct), img1, quality)
        # cv.imwrite('D:/dataset/fusion/img/img_{}_2.png'.format(self.ct), img2, quality)

        new_lab = lab1//255 + lab2//255
        new_lab = np.where(new_lab > 0, 255, 0)

        # lab2 = np.where(lab2>0,1,0)
        lab1 = np.array(lab1, dtype=np.uint8)
        lab2 = np.array(lab2, dtype=np.uint8)
        img1 *= (lab1//255)

        lab1 = np.where(lab1 > 0, 0, 1)
        lab1 = np.array(lab1, dtype=np.uint8)

        img2 *= lab1
        new_img = img1 + img2
        # cv.imwrite('D:/dataset/fusion/img/img_{}_3.png'.format(self.ct), img1, quality)
        # cv.imwrite('D:/dataset/fusion/img/img_{}_4.png'.format(self.ct), img2, quality)
        # cv.imwrite('D:/dataset/fusion/img/img_{}_5.png'.format(self.ct), new_img, quality)
        # cv.imwrite('D:/dataset/fusion/lab/lab_{}.png'.format(self.ct), new_lab, quality)

        return new_img, new_lab

    def mkdirs_path(self, new_path):    # 检查目录是否存在，若不存在则创建
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print('成功创建：{}'.format(new_path))

    def start_process(self, img, lab, img_path, lab_path, use_process=True):
        # 根据需求，选择是否启用多进程处理
        if use_process:
            myProcess = MyProcess()
            myProcess.getPara(img, lab, img_path, lab_path)
            myProcess.start()
        else:
            '''
            1.保存png图像，图像后缀必须为.png，图像质量0-9，默认为3，0质量最好，9最差。
                cv2.imwrite("123.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            2.保存jpg图像，图像后缀必须为.jpg，图像质量0-100，默认为95,100最好，0最差。
                cv2.imwrite("123.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
            3.使用摄像头时默认的图像尺寸为640*480，设置摄像头分辨率的方法如下。
                cap = cv2.VideoCapture(0)
                cap.set(3,1920)
                cap.set(4,1080)
                _, frame = cap.read()   
            '''
            cv.imwrite('{}'.format(img_path), img, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            cv.imwrite('{}'.format(lab_path), lab, [int(cv.IMWRITE_PNG_COMPRESSION), 9])


if __name__ == '__main__':
    data = Data_Enhance()
    run_status = [0,0,1]
    # data.copy_func()
    t1 = time.time()

    if run_status[1]:
        data.run()
        t2 = time.time()
        print('使用多进程耗时：{}'.format(t2-t1))

    if run_status[2]:
        data.run(use_process=False)
        t3 = time.time()
        print('不使用多进程耗时：{}'.format(t3-t1))

    if run_status[1]:
        data.random_copy()

    if run_status[1]:
        data.transform_format()