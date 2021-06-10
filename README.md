# Semantic-segmentation-network     语义分割网络  

# requirements  
tensorflow==2.0.0  
opencv-python==4.5.1.48  
numpy==1.18  

## backbone 常用主干网络  
  ResNet,  18，34，50，101，152  
  ResNeXt,  50，101，152  
  Res2et,  50， 101(待修复)，152(待修复)  

## Nets  
  VGG16  :  VGG16主干网络  
  MyModels.py  :  汇总，加有解码网络  
  PSPNET.py  :  以res50为主干网络的语义分割网络  

## utili  
  ASPP.py  ： ASPP+PSP模块  
  attentionModule.py  :  注意力机制  
  data_enhancement.py  : 数据增强方法  
  GPUSet.py : 设置GPU  
  LoadData.py : 加载数据集  
  modelTraining.py  : 部分训练技巧  
  multi_pool_module.py  : 多支路池化融合  
  My_LR.py  :  自定义学习率变化方法  
  myEvalue.py : 训练时模型评估方法  
  myLoss.py : 常用损失函数  
  scale_resize.py : 尺度缩放  
  select_path_module.py : 同multi_pool_module  
  
## 训练  
  train.py  

## 预测  
  predict.py  

