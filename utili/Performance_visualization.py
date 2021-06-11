import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
os.chdir(r'G:\result')  # 修改为自己的目录
mpl.rcParams['font.sans-serif'] = ['SimHei']

EPOCHS = 60
L_EP = 0
acc = []
loss = []
precision = []
recall = []
iou = []
miou = []
f1 = []
evalye_list = [loss, acc, precision, recall, f1, iou, miou]
evalyes = ['loss','acc','precision','recall','f1_score','iou','aiou',
         'val_loss','val_acc','val_precision','val_recall','val_f1_score','val_iou','val_aiou']
count = 0
for e in evalyes[7:]:
    with open('./{}.txt'.format(e), 'r', encoding='utf8') as f:
        for i in f.readlines():
            # print(i.strip())
            evalye_list[count].append(float(i.strip()))
    count += 1
for i in evalye_list:
    print(i)

fig3 = plt.figure(num='Acc/Loss', figsize=(8, 8), dpi=300, facecolor='#FFFFFF', edgecolor='#0000FF')
plt.rcParams['font.sans-serif']=['SimHei']  #修改字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 18})  #修改字体大小
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 14}) #设置图例文字大小
plt.xlabel('EPOCH',fontsize=18)
plt.ylabel('数值',fontsize=18)

plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18) #设置坐标轴刻度文字的尺寸

'''
plot参数介绍：
plot(x, y)
    可选参数：
        color='green'       颜色
        marker='o'          标记类型
        markersize=5        标记尺寸
        markevery=5         标记间隔，比如每5个点做一个标记
        linestyle='dashed'  线段类型
        linewidth=1         线段宽度
        
'''
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[0],marker='1', linestyle='-',markevery=5, label=evalyes[7])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[1],marker='2', linestyle='--',markevery=5, label=evalyes[8])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[2],marker='3', linestyle='-.',markevery=5, label=evalyes[9])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[3],marker='4', linestyle=':',markevery=5, label=evalyes[10])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[4],marker='p', linestyle='--',markevery=5, label=evalyes[11])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[5],marker='h', linestyle='-.',markevery=5, label=evalyes[12])
plt.plot(np.arange(L_EP, EPOCHS), evalye_list[6],marker='+', linestyle=':',markevery=5, label=evalyes[13])
plt.legend([evalyes[7], evalyes[8], evalyes[9], evalyes[10], evalyes[11], evalyes[12], evalyes[13]])
plt.savefig("./acc.jpg",bbox_inches = 'tight')
plt.show()