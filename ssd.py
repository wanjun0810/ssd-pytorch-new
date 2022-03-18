import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *  # box_utils,multibox_loss,l2norm,prior_box,detection
from data import voc, coco
import os
import numpy as np



class SSD(nn.Module):  # 自定义SSD网络
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args（参数）:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    # 模型初始化
    # 初始化网络参数
    # def __init__(self, phase, size, base, extras, head, num_classes):
    def __init__(self, phase, size, base, extras, head, num_classes, output_channel, gcd, concat_in_channel, concat_channel):
        '''
        base_, extras_, head_ = multibox(vgg(base[str(size)], 3), add_extras(extras[str(size)], 1024), mbox[str(size)], num_classes)
        SSD(phase, size, base_, extras_, head_, num_classes)
        phase = test/train
        size = 300
        base = vgg(base['300'],3)
        extras = extras['300'], 1024)
        head = multibox( , , mbox['300'])
        num_classes = 21
        '''
        super(SSD, self).__init__()  # 调用父类的__init__()
        self.phase = phase  # test/train
        self.num_classes = num_classes  # 分类数
        self.cfg = (coco, voc)[num_classes == 21]  # [num_classes == 21] = 0 / 1
        self.priorbox = PriorBox(self.cfg)  # layers/functions/prior_box.py 生成priorbox
        self.priors = Variable(self.priorbox.forward(), volatile=True)  #  self.priorbox.forward()前向传播的激活值，volatile 固定priorbox层梯度不进行更新(from torch.autograd import Variable)
        self.size = size  # 输入图片尺寸

        # SSD network
        # 初始化网络结构
        self.vgg = nn.ModuleList(base)  # 将base网络添加到当前module(SSD)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)  # 进行L2标准化
        self.extras = nn.ModuleList(extras)  # 将add_extras网络层添加到当前module(SSD)
        '''add code'''
        # self.equal_channel_size = nn.ModuleList(equal_parameter(output_channel, gcd))
        self.equal_channel_size = nn.ModuleList(equal_parameter(output_channel, gcd)[0])
        self.equal_channel = nn.ModuleList(equal_parameter(output_channel, gcd)[1])  # [10]
        self.concat_operate = nn.ModuleList(concat_feature(concat_in_channel, concat_channel))  # [40]
        ''''''
        self.loc = nn.ModuleList(head[0])  # 将用于边框回归的卷积层添加到当前网络 head = (loc_layers, conf_layers)
        self.conf = nn.ModuleList(head[1])  # 将用于分类的卷积层添加到当前module

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # 将Softmax函数应用于n维输入Tensor对其进行重新缩放，以使n维输出Tensor的元素位于（0,1）范围内，并且总和为1
            # dim（int）:用来计算Softmax的尺寸（因此，沿着dim的每个切片的总和为1）。
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.
    
    # 网络前向传播操作, 将设计好的layers和ops应用到输入图片 x 上
    # def forward(self, x):
    def forward(self, x, output_channel, gcd):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes] × ---> [batch,num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4] × ---> [batch,num_priors,4]
                    3: priorbox layers, Shape: [2,num_priors*4] × ---> [num_priors,4]
        """
        sources = list()  # 存储的是参与预测的卷积层的输出, 6个指定的卷积层
        loc = list()  # 用于存储预测的边框信息
        conf = list()  # 用于存储预测的类别信息

        # apply vgg up to conv4_3 relu
        # 计算vgg直到conv4_3层relu结果的特征图
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)  # 将conv4_3层relu结果的特征图进行L2标准化
        sources.append(s)  # 将conv4_3后输出的特征层添加到sources中, 后面会根据sources中的元素进行预测

        # apply vgg up to fc7
        # 计算vgg从conv4_3的特征图之后到BaseNetwork结束的特征图conv7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)  # 将conv7输出的特征层添加到sources中, 后面会根据sources中的元素进行预测

        # apply extra layers and cache source layer outputs
        # 计算SSD额外加的层的特征图
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # 在extras_layers中, 第1,3,5,7(从第0开始)的卷积层的输出会用于预测box位置和类别, 将其添加到sources列表
                sources.append(x)
                
        '''add code'''

        '''对应计算元素尺寸相同,t0(Conv4_3),t3=t4(Conv7),t7=t8(Conv8_2),t11=t12(Conv9_2),t15=t16(Conv10_2),t19(Conv11_2)'''
        temp_source_0 = [sources[0], sources[1], sources[0], sources[1], sources[1], sources[2],
                         sources[1], sources[2], sources[2], sources[3], sources[2], sources[3],
                         sources[3], sources[4], sources[3], sources[4], sources[4], sources[5],
                         sources[4], sources[5]]  # 尺寸不同 通道不同
        '''用于计算协方差的元素'''
        # equal_channel = []
        # for i, j in zip(output_channel, gcd):
        #     equal_channel.append(nn.Parameter(torch.randn(i, j, 1, 1)))

        temp_source_result = []  # 通道相同
        for i in range(0, len(temp_source_0), 2):
            temp_source_group1 = int(temp_source_0[i].size(1) / gcd[int(i/2)])
            temp_source_group2 = int(temp_source_0[i + 1].size(1) / gcd[int(i/2)])
            temp_source_size1 = temp_source_0[i].size(3)
            temp_source_size2 = temp_source_0[i+1].size(3)
            temp_source_1 = torch.zeros(temp_source_0[i].size(0), output_channel[int(i / 2)], temp_source_size1, temp_source_size1)
            temp_source_2 = torch.zeros(temp_source_0[i + 1].size(0), output_channel[int(i/2)], temp_source_size2, temp_source_size2)
            for j in range(temp_source_group1):
                # temp_source_1 = F.conv2d(temp_source_0[i][:, j:j + gcd[int(i/2)], :, :], equal_channel[int(i/2)], bias=None, stride=1) + temp_source_1
                temp_source_1 = self.equal_channel[int(i/2)](temp_source_0[i][:, j:j + gcd[int(i/2)], :, :]) + temp_source_1
            temp_source_result.append(temp_source_1)
            for j in range(temp_source_group2):
                # temp_source_2 = F.conv2d(temp_source_0[i + 1][:, j:j + gcd[int(i/2)], :, :], equal_channel[int(i/2)], bias=None, stride=1) + temp_source_2
                temp_source_2 = self.equal_channel[int(i/2)](temp_source_0[i + 1][:, j:j + gcd[int(i/2)], :, :]) + temp_source_2
            temp_source_result.append(temp_source_2)


        '''协方差计算'''
        cov = []
        for i in range(0, len(temp_source_result), 2):
            cov.append(ComputeCov(F.adaptive_avg_pool2d(temp_source_result[i], (1, 1)).detach().cpu().numpy(), F.adaptive_avg_pool2d(temp_source_result[i+1], (1, 1)).detach().cpu().numpy()))

        num_images = temp_source_0[0].size(0)  # batch_size

        temp_source = [sources[0], self.equal_channel_size[0](sources[1]),
                       self.equal_channel_size[1](sources[0]), sources[1],
                       sources[1], self.equal_channel_size[2](sources[2]),
                       self.equal_channel_size[3](sources[1]), sources[2],
                       sources[2], self.equal_channel_size[4](sources[3]),
                       self.equal_channel_size[5](sources[2]), sources[3],
                       sources[3], self.equal_channel_size[6](sources[4]),
                       self.equal_channel_size[7](sources[3]), sources[4],
                       sources[4], self.equal_channel_size[8](sources[5]),
                       self.equal_channel_size[9](sources[4]), sources[5]
                       ]  # 尺寸相同 通道不同

        '''concat元素,t0(Conv4_3),t3=t4(Conv7),t7=t8(Conv8_2),t11=t12(Conv9_2),t15=t16(Conv10_2),t19(Conv11_2)'''
        for i in range(0, len(self.concat_operate), 2):  # concat元素 标准化后卷积 选择
            temp_source[int(i/2)] = self.concat_operate[i](temp_source[int(i/2)])
            temp_source[int(i/2)] = self.concat_operate[i+1](temp_source[int(i/2)])

        '''concat'''
        sources_1 = list()
        for j in range(num_images):
            if j == 0:
                # a ---> Conv4
                if cov[0][j] > 0:
                    sources_1.append((temp_source[0][j] + temp_source[1][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[0][j].unsqueeze(0))

                # b,c ---> Conv7
                if cov[1][j] > 0 and cov[2][j] > 0:
                    sources_1.append((temp_source[2][j] + temp_source[3][j] + temp_source[5][j]).unsqueeze(0))
                elif cov[1][j] > 0 and cov[2][j] <= 0:
                    sources_1.append((temp_source[2][j] + temp_source[3][j]).unsqueeze(0))
                elif cov[1][j] <= 0 and cov[2][j] > 0:
                    sources_1.append((temp_source[3][j] + temp_source[5][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[3][j].unsqueeze(0))

                # d,e ---> Conv8
                if cov[3][j] > 0 and cov[4][j] > 0:
                    sources_1.append((temp_source[6][j] + temp_source[7][j] + temp_source[9][j]).unsqueeze(0))
                elif cov[3][j] > 0 and cov[4][j] <= 0:
                    sources_1.append((temp_source[6][j] + temp_source[7][j]).unsqueeze(0))
                elif cov[3][j] <= 0 and cov[4][j] > 0:
                    sources_1.append((temp_source[7][j] + temp_source[9][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[7][j].unsqueeze(0))
                
                # f,g ---> Conv9
                if cov[5][j] > 0 and cov[6][j] > 0:
                    sources_1.append((temp_source[10][j] + temp_source[11][j] + temp_source[13][j]).unsqueeze(0))
                elif cov[5][j] > 0 and cov[6][j] <= 0:
                    sources_1.append((temp_source[10][j] + temp_source[11][j]).unsqueeze(0))
                elif cov[5][j] <= 0 and cov[6][j] > 0:
                    sources_1.append((temp_source[11][j] + temp_source[13][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[11][j].unsqueeze(0))

                # h,i ---> Conv10
                if cov[7][j] > 0 and cov[8][j] > 0:
                    sources_1.append((temp_source[14][j] + temp_source[15][j] + temp_source[17][j]).unsqueeze(0))
                elif cov[7][j] > 0 and cov[8][j] <= 0:
                    sources_1.append((temp_source[14][j] + temp_source[15][j]).unsqueeze(0))
                elif cov[7][j] <= 0 and cov[8][j] > 0:
                    sources_1.append((temp_source[15][j] + temp_source[17][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[15][j].unsqueeze(0))
                
                # j ---> Conv11
                if cov[9][j] > 0:
                    sources_1.append((temp_source[18][j] + temp_source[19][j]).unsqueeze(0))
                else:
                    sources_1.append(temp_source[19][j].unsqueeze(0))

            else:
                # a ---> Conv4
                if cov[0][j] > 0:
                    sources_1[0] = torch.cat((sources_1[0], ((temp_source[0][j] + temp_source[1][j])).unsqueeze(0)), 0)
                else:
                    sources_1[0] = torch.cat((sources_1[0], temp_source[0][j].unsqueeze(0)),0)

                # b,c ---> Conv7
                if cov[1][j] > 0 and cov[2][j] > 0:
                    sources_1[1] = torch.cat((sources_1[1], (temp_source[2][j] + temp_source[3][j] + temp_source[5][j]).unsqueeze(0)), 0)
                elif cov[1][j] > 0 and cov[2][j] <= 0:
                    sources_1[1] = torch.cat((sources_1[1], (temp_source[2][j] + temp_source[3][j]).unsqueeze(0)), 0)
                elif cov[1][j] <= 0 and cov[2][j] > 0:
                    sources_1[1] = torch.cat((sources_1[1], (temp_source[3][j] + temp_source[5][j]).unsqueeze(0)), 0)
                else:
                    sources_1[1] = torch.cat((sources_1[1], temp_source[3][j].unsqueeze(0)), 0)

                # d,e ---> Conv8
                if cov[3][j] > 0 and cov[4][j] > 0:
                    sources_1[2] = torch.cat((sources_1[2], (temp_source[6][j] + temp_source[7][j] + temp_source[9][j]).unsqueeze(0)), 0)
                elif cov[3][j] > 0 and cov[4][j] <= 0:
                    sources_1[2] = torch.cat((sources_1[2], (temp_source[6][j] + temp_source[7][j]).unsqueeze(0)), 0)
                elif cov[3][j] <= 0 and cov[4][j] > 0:
                    sources_1[2] = torch.cat((sources_1[2], (temp_source[7][j] + temp_source[9][j]).unsqueeze(0)), 0)
                else:
                    sources_1[2] = torch.cat((sources_1[2], temp_source[7][j].unsqueeze(0)), 0)
                
                # f,g ---> Conv9
                if cov[5][j] > 0 and cov[6][j] > 0:
                    sources_1[3] = torch.cat((sources_1[3], (temp_source[10][j] + temp_source[11][j] + temp_source[13][j]).unsqueeze(0)), 0)
                elif cov[5][j] > 0 and cov[6][j] <= 0:
                    sources_1[3] = torch.cat((sources_1[3], (temp_source[10][j] + temp_source[11][j]).unsqueeze(0)), 0)
                elif cov[5][j] <= 0 and cov[6][j] > 0:
                    sources_1[3] = torch.cat((sources_1[3], (temp_source[11][j] + temp_source[13][j]).unsqueeze(0)), 0)
                else:
                    sources_1[3] = torch.cat((sources_1[3], temp_source[11][j].unsqueeze(0)), 0)

                # h,i ---> Conv10
                if cov[7][j] > 0 and cov[8][j] > 0:
                    sources_1[4] = torch.cat((sources_1[4], (temp_source[14][j] + temp_source[15][j] + temp_source[17][j]).unsqueeze(0)), 0)
                elif cov[7][j] > 0 and cov[8][j] <= 0:
                    sources_1[4] = torch.cat((sources_1[4], (temp_source[14][j] + temp_source[15][j]).unsqueeze(0)), 0)
                elif cov[7][j] <= 0 and cov[8][j] > 0:
                    sources_1[4] = torch.cat((sources_1[4], (temp_source[15][j] + temp_source[17][j]).unsqueeze(0)), 0)
                else:
                    sources_1[4] = torch.cat((sources_1[4], temp_source[15][j].unsqueeze(0)), 0)
                
                # j ---> Conv11
                if cov[9][j] > 0:
                    sources_1[5] = torch.cat((sources_1[5], (temp_source[18][j] + temp_source[19][j]).unsqueeze(0)), 0)
                else:
                    sources_1[5] = torch.cat((sources_1[5], temp_source[19][j].unsqueeze(0)), 0)
        ''''''

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # apply multibox head to source layers
        '''修改代码'''
        # for (x, l, c) in zip(sources, self.loc, self.conf):
        for (x, l, c) in zip(sources_1, self.loc, self.conf):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # self.loc = nn.ModuleList(head[0])  # head = (loc_layers, conf_layers)
        # self.conf = nn.ModuleList(head[1])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W)(batch_size,channel,hight,width),这里的排列将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,而 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            
            # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes

        '''
        loc、conf是特征图
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        loc.view(loc.size(0), -1, 4)
        将卷积后的特征图转化为prior box与坐标的形式
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        conf.view(conf.size(0), -1, self.num_classes)
        将卷积后的特征图转化为prior box与类别的形式
        '''

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # torch.view将原有数据重新分配为一个新的张量
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4] num_boxes * 4 = w * h * c
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),   
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            # 将卷积后的特征图转化为prior box与坐标、类别的形式
            # loc[batch_size,num_priors,4]、conf[batch_size,num_priors,num_classes]、prior[num_priors,4]
        return output  # 返回loc坐标、conf结果、prior box坐标

    # 加载参数权重
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)  # 分离文件名(other)与扩展名(ext)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))  # 加载gpu训练好的模型参数到cpu
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# 定义BaseNetwork各层及操作
def vgg(cfg, i, batch_norm=False):
    '''
    base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
            '512': [],}  M为Max Pool，C为向上取整的Max Pool(75->38)
    cfg = base['300']  BaseNetwork操作及卷积核个数
    i = 3  输入图片通道数
    '''
    layers = []  # 定义输出的BaseNetwork各层
    in_channels = i  # 输入图片通道数
    for v in cfg:
        if v == 'M':  # 最大池化层，池化核尺寸2×2，步长为2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':  # 最大池化层，池化核尺寸2×2，步长为2，ceil_mode天花板模式，保留补充不足的边
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:  # 卷积层，输入通道数，输出通道数（卷积核个数），卷积核尺寸3×3，padding为1，卷积后图片尺寸不变
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:  # 要求归一化，将卷积层、归一化层、ReLU层组合到layers
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:  # 不要求归一化，将卷积层、ReLU层组合到到layers
                layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v  # 下次输入通道数=本次输出通道数
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 最大池化层，池化核尺寸3×3，步长为1，padding为1，池化后图片尺寸不变
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # 卷积层(空洞卷积)，卷积核尺寸3×3，padding为6，空格数+1为6，卷积后图片尺寸不变
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # 卷积层，卷积核尺寸为1×1，卷积后尺寸不变
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]  # 不归一化时，layers组合后共35层(0-34，ReLU也算一层)，Conv4_3为第21层，Conv7为倒数第2层
    # layers += [pool5, conv6, nn.ReLU(inplace=False), conv7, nn.ReLU(inplace=False)]
    # conv7--->19×19×1024
    return layers  # 返回组合后的BaseNetwork各层


# Extra layers added to VGG for feature scaling
# 定义BaseNetwork后增加的层及操作
def add_extras(cfg, i, batch_norm=False):
    '''
    extras = {  '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
                '512': [],}   S之后的卷积层步长为2
    cfg = extras['300']  额外添加层操作及卷积核个数
    i = 1024  输入特征通道数
    '''
    layers = []  # 定义输出的add_extras各层，只卷积，不激活
    in_channels = i  # 输入图片通道数
    flag = False  # false = 0
    for k, v in enumerate(cfg):  # enumerate同时列出数据下标和数据，cfg为支持迭代的对象，k为cfg数据下标，v为cfg数据
        if in_channels != 'S':  # S不是通道数，S对应的卷积核通道数为下标k+1
            if v == 'S':  # k对应的v为S时，对应的卷积层卷积核个数下标k+1，卷积步长为2，padding为1，卷积核尺寸为1×1或3×3
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:  # k对应的v为非S时，对应的卷积层卷积核个数下标为k，值为v
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag  # 卷积核尺寸为1×1与3×3交替，False时为1×1，True时为3×3，
        in_channels = v  # 本次输出通道数为下一层输入通道数
    return layers  # 返回组合后的add_extras各层


# 利用vgg与extra_layers网络结构定义边框回归卷积层与分类回归卷积层及操作，返回SSD网络结构
def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    mbox = {'300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
            '512': [], }  选定的特征图每个点提取的prior box数
    vgg = vgg.layers
    extra_layers = add_extras.layers
    cfg = mbox['300']
    num_classes = 21(VOC)/80(COCO)
    '''
    loc_layers = []  # 定义边框回归的卷积层
    conf_layers = []  # 定义分类回归的卷积层
    vgg_source = [21, -2]  # python列表负数索引从右向左，因此为vgg的第21层conv4_3与倒数第2层conv7
    for k, v in enumerate(vgg_source):  # BaseNetwork中特定特征图对应的边框回归卷积层与分类卷积层(0, 21; 1, -2)
        # 边框回归卷积层，conv4_3、conv7输出通道数，选定特征图每个点提取的prior box数×4(4个坐标),卷积核尺寸为3×3，padding为1
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        # 分类回归卷积层，conv4_3、conv7输出通道数，选定特征图每个点提取的prior box数×类别数(VOC=21，COCO=80),卷积核尺寸为3×3，padding为1
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):  # extra_layers中特定特征图对应的边框回归卷积层与分类卷积层，v取额外的特征图(双冒号，从下标为1开始，每次跳两个，如1->3)，k从2开始(vgg两个特征图用了k=0，1)
        # 边框回归卷积层，extra_layer输出通道数，选定特征图每个点提取的prior box数×4(4个坐标),卷积核尺寸为3×3，padding为1
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        # 分类回归卷积层，extra_layer输出通道数，选定特征图每个点提取的prior box数×类别数(VOC=21，COCO=80),卷积核尺寸为3×3，padding为1
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)  # 返回SSD网络结构，BaseNetwork_layers+add_layers+loc_layers+conf_layers


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [],
# }

mbox = {
    '300': [5, 5, 5, 5, 5, 5],  # number of boxes per feature map location
    '512': [],
}

output_channel = {
    '300': [512, 1024, 1024, 512, 512, 256, 256, 256, 256, 256],
    '512': [],
}

gcd = {
    '300': [512, 512, 512, 512, 256, 256, 256, 256, 256, 256],
    '512': [],
}

concat_in_channel = {
    '300': [512, 1024, 512, 1024, 1024, 512, 1024, 512, 512, 256, 512, 256, 256, 256, 256, 256, 256, 256, 256, 256],
    '512': []
}

concat_channel = {
    '300': [512, 512, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256],
    '512': []
}


def concat_feature(concat_in_channel, concat_channel):
    concat_operate = []
    for i in range(len(concat_channel)):
        concat_operate.append(nn.BatchNorm2d(concat_in_channel[i]))
        concat_operate.append(nn.Conv2d(concat_in_channel[i], concat_channel[i], kernel_size=3, stride=1, padding=1))
    return concat_operate


def equal_parameter(output_channel, gcd):
    equal_parameter = []
    for i, j in zip(output_channel, gcd):
        # equal_parameter.append(nn.Parameter(torch.randn(i, j, 1, 1)))
        equal_parameter.append(nn.Conv2d(in_channels=j, out_channels=i, kernel_size=1, stride=1))
    # 将特征图尺寸对应相等
    equal_size = []
    equal_size.append(nn.UpsamplingBilinear2d(scale_factor=2))  # Conv7 1
    equal_size.append(nn.AvgPool2d(kernel_size=2, stride=2))  # Conv4_3 0
    equal_size.append(nn.UpsamplingBilinear2d(scale_factor=19 / 10))  # Conv8_2 2
    equal_size.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # Conv7 1
    equal_size.append(nn.UpsamplingBilinear2d(scale_factor=2))  # Conv9_2 3
    equal_size.append(nn.AvgPool2d(kernel_size=2, stride=2))  # Conv8_2 2
    equal_size.append(nn.UpsamplingBilinear2d(scale_factor=5 / 3))  # Conv10_2 4
    equal_size.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # Conv9_2 3
    equal_size.append(nn.UpsamplingBilinear2d(scale_factor=3))  # Conv11_2 5
    equal_size.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False))  # Conv10_2 4
    return equal_size, equal_parameter
    # return equal_size


def ComputeCov(output1, output2):
    if output1.shape[2]==1:
        x = np.zeros(output1.shape[1])
        y = np.zeros(output2.shape[1])  # 两个空列表x,y,分别把各个通道1*1的值合并成一个列表 
        conv_mean_batch_1 =np.zeros(output1.shape[0])  # 存两个图中，各个通道的协方差
        # x = np.append(x, [[1,2,3,4]], axis = 0)
        for i in range(output1.shape[0]):
                       
            for j in range(output1.shape[1]):

                x[j] = output1[i, j, :, :]
                y[j] = output2[i, j, :, :]  # 将两个通道的特征图拍平后送进x,y

            z = np.vstack((x, y))  # 为了算协方差，将x,y拼接在一起
           
            corr_z = np.corrcoef(z)

            conv_mean_batch_1[i] = corr_z[0][1]  # 每一张图片的协方差均值
        return (conv_mean_batch_1)


    else:
        x = np.zeros(output1.shape[2]*output1.shape[3])
    # 每次定义两个空列表x,y
        y = np.zeros(output2.shape[2]*output2.shape[3])
        conv_mean_batch = np.zeros(output1.shape[0])  # 存batch个图中，每两张图中各个通道的协方差
    # x = np.append(x, [[1,2,3,4]], axis = 0)
        for i in range(output1.shape[0]):  # i为batch
            conv_matrix = np.zeros(output1.shape[1])  # 存两张图中各个通道的协方差
            for j in range(output1.shape[1]):

                x = output1[i, j, :, :].reshape(1, output1.shape[2]*output1.shape[3])
                
                # 将两个通道的特征图拍平后送进x,y
                y = output2[i, j, :, :].reshape(1, output2.shape[2]*output2.shape[3])
                
                z = np.vstack((x, y))  # 为了算协方差，将x,y拼接在一起
                corr_z = np.corrcoef(z)
            
                conv_matrix[j] =corr_z[0][1]
   
            conv_mean = conv_matrix.mean()  # 所有通道的相关系数均值

   
            conv_mean_batch[i] = conv_mean  # 每一张图片的相关系数均值
        return conv_mean_batch



# 定义网络结构，实现SSD模型
def build_ssd(phase, size=300, num_classes=21):
    '''
    phase=train/test
    size=300
    num_classes=21
    '''
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)  # 定义网络结构
                                     
    '''修改代码'''
    # return SSD(phase, size, base_, extras_, head_, num_classes)  #　实现SSD模型
    return SSD(phase, size, base_, extras_, head_, num_classes, output_channel['300'], gcd['300'],concat_in_channel['300'],concat_channel['300']) 
    