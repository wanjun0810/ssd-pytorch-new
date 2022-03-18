#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import sys

import time

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if not torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('./weights/ssd300_mAP_77.43_v2.pth')

image = cv2.imread('./data/000377.jpg', cv2.IMREAD_COLOR)#IMREAD_COLOR
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
plt.figure(figsize=(10,10))

x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)


xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if not torch.cuda.is_available():
    xx = xx.cuda()

#torch.cuda.synchronize()
start = time.time()
print(start)
y = net(xx)

#torch.cuda.synchronize()
end = time.time()
print(end)
print('Test time is %f' % (end-start))

from data import VOC_CLASSES as labels
top_k=10

# plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()#边框颜色
currentAxis = plt.gca()#画框

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)#显示小框大小
for i in range(detections.size(1)):#读类别
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)#显示类别和分数
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()#大小框的位置
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1#大小框的位置
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))#画框
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})#写字
        j+=1
plt.axis('off')
plt.imshow(rgb_image)
plt.savefig('./pic/000.jpg')
plt.show()
