from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        '''
        cfg = (coco, voc)[numclass == 21]
        voc = { 'num_classes': 21,  类别数
                'lr_steps': (80000, 100000, 120000),  学习率衰减的迭代次数
                'max_iter': 120000,  最大迭代次数
                'feature_maps': [38, 19, 10, 5, 3, 1],  选定特征图尺寸
                'min_dim': 300,  输入图片尺寸
                'steps': [8, 16, 32, 64, 100, 300],  选定特征图尺寸与输入图像尺寸的对应关系，feature_maps = min_din(300) / steps
                'min_sizes': [30, 60, 111, 162, 213, 264],  priorbox尺度  s_k = min_size / mid_dim(300)
                'max_sizes': [60, 111, 162, 213, 264, 315],  priorbox尺度对应的特征图尺度  s_(k+1)
                'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  宽高比  w = s_k(min_size) * √aspect_rations  h = s_k(min_size) / √aspect_rations
                'variance': [0.1, 0.2],
                'clip': True,  是否将输出结果压缩到0-1
                'name': 'VOC',}
        '''
        

        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']  # 300×300
        # number of priors for feature map location (either 4 or 6)
        # feature map 上的每一个像素点都对应着6个prior boxes
        self.num_priors = len(cfg['aspect_ratios'])  # 6 没用上
        self.variance = cfg['variance'] or [0.1]  # 没用上
        self.feature_maps = cfg['feature_maps']  # 选定特征图尺寸  feature_maps = min_din(300) / steps
        self.min_sizes = cfg['min_sizes']  # priorbox尺度  s_k = min_size / mid_dim(300)
        self.max_sizes = cfg['max_sizes']  # priorbox尺度对应的特征图尺度 s_(k+1)
        self.steps = cfg['steps']  # 选定特征图尺寸与输入图像尺寸的对应关系，feature_maps = min_din(300) / steps
        self.aspect_ratios = cfg['aspect_ratios']  # 宽高比  w = s_k(min_size) * √aspect_rations  h = s_k(min_size) / √aspect_rations
        self.clip = cfg['clip']  # 是否将输出结果压缩到0-1
        self.version = cfg['name']  # VOC (/COCO)
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []  # 存储所有prior box坐标
        for k, f in enumerate(self.feature_maps):  # k : 0 ~ 5  f: 38, 19, 10, 5, 3, 1
            for i, j in product(range(f), repeat=2):  # 创建一个迭代器，生成表示f中的项目的笛卡尔积的元组，repeat表示重复生成序列的次数。i,j->[(38*38)],[(19*19)],...,[1]
                f_k = self.image_size / self.steps[k]  # 计算选定特征图尺寸  feature_maps = min_din(300) / steps
                # unit center x,y
                cx = (j + 0.5) / f_k  # 生成prior box中心坐标   x = (j + 0.5) / 特征图尺寸
                cy = (i + 0.5) / f_k  # 生成prior box中心坐标   y = (i + 0.5) / 特征图尺寸
                # i / f_k = i * 8 / mid_dim(300) 映射原图再归一化

                # s_k = min_size / mid_dim(300)

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size  # s_k = min_size / mid_dim(300)
                mean += [cx, cy, s_k, s_k]  # aspect_ratio = 1 时， w、h = s_k

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))  # aspect_ratio等于1时附加的尺寸，s_k_prime = sqrt(s_k * s_(k+1))
                mean += [cx, cy, s_k_prime, s_k_prime]  # aspect_ratio = 1 时， w、h = s_k_prime

                # w = s_k(min_size) * √aspect_rations  h = s_k(min_size) / √aspect_rations

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:  # 计算宽高比为2、3的prior box坐标
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]  # s_k * √aspect_rations , s_k / √aspect_rations
                    # mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]  # s_k * (1 / √aspect_rations) , s_k / (1 / √aspect_rations)
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)  # 将一个mean列表转换为[num_prior(8732), 4]格式，每行为一个prior box的四个坐标
        if self.clip:  #　将输出结果压缩到0-1
            output.clamp_(max=1, min=0)
        return output  # 返回所有prior box的坐标  [num_prior(8732), 4]
