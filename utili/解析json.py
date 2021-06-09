import json
import numpy as np
import cv2 as cv
import os, glob, shutil, io, base64
import PIL
from PIL import Image, ImageDraw
'''
功能：
    把json文件转成label图片，
'''

class Peocess_Json():
    def __init__(self):
        self.label = ['background']

    def img_b64_to_arr(self, img_b64):  # 图片转格式
        f = io.BytesIO()
        f.write(base64.b64decode(img_b64))
        img_arr = np.array(PIL.Image.open(f))
        return img_arr

    def img_arr_to_b64(self, img_arr):  # 图片转格式
        img_pil = PIL.Image.fromarray(img_arr)
        f = io.BytesIO()
        img_pil.save(f, format='PNG')
        img_bin = f.getvalue()
        img_b64 = base64.encodebytes(img_bin)
        return img_b64

    def read_json(self, file_path):         # 读取JSON文件，获取坐标信息，保存二值化图
        with open(file_path) as f:          # 打开json文件
            json_list = json.load(f)        # 读取到json_list中

        shape = json_list['shapes']         # 获取shape字段内容
        fileName = json_list['imagePath']
        fileName = fileName.split('.')[:-1]
        fileName.append('label.png')
        fileName = '_'.join(fileName)

        img = self.img_b64_to_arr(json_list['imageData'])   # 图片转格式
        img_h, img_w, _ = img.shape        # 获取图片的h,w,c

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        masks = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(len(shape)):
            label = shape[i]['label']       # 获取每一个目标的名字(label)
            if label not in self.label:     # 统计名字类别
                self.label.append(label)
            index = self.label.index(label)
            points = shape[i]['points']     # 获取每一个目标的坐标点
            '''
            坐标点格式如下
            points = [
            [x1,y1],[x2,y2]...
            ]
            '''
            masks = self.polygons_to_mask(mask, polygons=points, index=index) + masks   # 绘图函数
        cv.imwrite('./{}'.format(fileName), masks)  # 保存


    def polygons_to_mask(self, mask, polygons, index):  # 创建MASK图
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        mask = np.where(mask, index, 0)
        return mask

    def proce_file(self):   # 复制JSON文件
        files = glob.glob('../200926-4/*.json')
        print(files[:10])
        print(len(files))
        for i in files:
            old_path = i.split('.json')[0]
            old_img_path = old_path + '.jpg'

            new_file = os.path.join('./data', old_path.split('\\')[-1])
            new_img_path = new_file + '.jpg'

            print('old_path {} -->> new_file {}'.format(i, new_file+ '.json'))
            print('old_img_path {} -->> new_img_path {}'.format(old_img_path, new_img_path))
            try:
                shutil.copyfile(i, new_file+ '.json')
                shutil.copyfile(old_img_path, new_img_path)
            except:
                print(i)
                raise ValueError('复制错误')

    def write_json(self):   # 写入JSON文件
        person_str = '{"title":"build", "points":["1123456","2adqwdac","3fwf856"]}'
        person_dic = json.loads(person_str)
        person_dic = json.dumps(person_dic, indent=4, sort_keys=True)
        return person_dic


if __name__ == '__main__':
    js = Peocess_Json()
    # js.read_json('./data/3135.0-375.0DOM.json')
    person_dic = js.write_json()

    print(person_dic)


