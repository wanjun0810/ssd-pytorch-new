# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        '''
        num_classes = 21
        overlap_thresh = 0.5  正样本IoU阈值
        prior_for_matching = True 
        bkg_label = 0 背景标签
        neg_mining = True 难负样本挖掘
        neg_pos = 3 正负样本比例
        neg_overlap = 0.5  负样本IoU阈值
        encode_target = False 
        use_gpu = True  使用gpu
        '''
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu  # True
        self.num_classes = num_classes  # 21
        self.threshold = overlap_thresh  # 0.5
        self.background_label = bkg_label  # 0
        self.encode_target = encode_target  # False
        self.use_prior_for_matching = prior_for_matching  # True
        self.do_neg_mining = neg_mining  # True
        self.negpos_ratio = neg_pos  # 3 正负样本比例
        self.neg_overlap = neg_overlap  # 0.5
        self.variance = cfg['variance']  # 'variance': [0.1, 0.2]

    def forward(self, predictions, targets):
        '''
        predictions(tuple) = loc[batch_size,num_priors,4]、conf[batch_size,num_priors,num_classes]、prior[num_priors,4]
                           = SSD(phase, size, base_, extras_, head_, num_classes) = out = net()
        targets(tensor) = [batch_size,num_objs,5]
                          num_objs -> 当前图片中包含的物体个数
                          5 -> 当前图片中包含的物体ground truth坐标(4)+当前图片中包含的物体类别(1)
        '''
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions  # loc[batch_size,num_priors(8732),4]、conf[batch_size,num_priors(8732),num_classes]、prior[num_priors(8732),4]
        num = loc_data.size(0)  # batch_size
        priors = priors[:loc_data.size(1), :]  # loc_data.size(1) = num_priors(8732) priors = loc_data计算的结果数 
        num_priors = (priors.size(0))  # 8732
        num_classes = self.num_classes  # 21

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)  # [batch_size, num_priors, 4] 存储batch_size张图片中的每一张图片的8732个num_priors匹配的ground truth坐标
        conf_t = torch.LongTensor(num, num_priors)  #  [batch_size, num_priors] 存储batch_size张图片中的每一张图片的8732个num_priors匹配的ground truth类别
        for idx in range(num):  # idx为当前处理的batch_size中的第几张图片
            truths = targets[idx][:, :-1].data  # [idx][num_objs, :-1] -> [idx][num_objs, 0-3]  ground truth 坐标
            labels = targets[idx][:, -1].data  # [idx][num_objs, 4]  ground truth 类别
            defaults = priors.data  # prior box 坐标
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)  # 匹配， 获得loc_t和conf_t
            # [batch_size, num_priors, 4] 存储batch_size张图片中的每一张图片的8732个num_priors对应的ground truth的回归坐标
            # [batch_size, num_priors] 存储batch_size张图片中的每一张图片的8732个num_priors对应的ground truth类别
        if self.use_gpu:  # 使用gpu， 将数据送入gpu
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)  # 用可计算梯度的变量存储数据
        conf_t = Variable(conf_t, requires_grad=False)  # 用可计算梯度的变量存储数据

        pos = conf_t > 0  # [batch_size, num_priors] ，>0 -> True , <=0 -> False，与ground truth的IoU小于阈值的样本为0
        num_pos = pos.sum(dim=1, keepdim=True)  # [batch_size, num_priors] -> [0, 1] 计算第一维度(1)num_priors中True的个数，即每行的正样本数目，即每张图片的匹配后的正样本数

        # Localization Loss (Smooth L1)  边框回归损失
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [batch,num_priors,4]  [batch_size, num_priors]-unsqueeze(pos.dim=2)->[batch_size, num_priors, 1]-expand_as->[batch,num_priors,4]
        loc_p = loc_data[pos_idx].view(-1, 4)  # [_, 4] _为正样本个数(True) loc_data[pos_idx]为列表，变形为[_, 4] 匹配的正样本预测坐标
        loc_t = loc_t[pos_idx].view(-1, 4)  # [_, 4] _为正样本个数(True) loc_t[pos_idx]为列表，变形为[_, 4] 匹配的正样本的回归目标坐标
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # 计算smooth_L1损边框回归损失 [_, 4] [_, 4] size_average设置为False，则每个小批次的损失将被相加。


        # Compute max conf across batch for hard negative mining
        # 计算难负样本挖掘的最大的置信度,计算softmax损失
        batch_conf = conf_data.view(-1, self.num_classes)  # [batch_size * num_priors(8732), num_classes] batch_size张图片的所有prior box一起做难负样本挖掘
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # ⬆ [batch_size * num_priors(8732), 1]  softmax交叉熵损失  y_i=1

        ''' ⬆ loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        log_sum_exp(batch_conf): [batch_size * num_priors(8732), 1]
        log_sum_exp(x) ---> torch.log(torch.sum(torch.exp(x-x.data.max()), 1, keepdim=True)) + x.data.max()

        batch_conf.gather(1, conf_t.view(-1, 1)): [batch_size * num_priors(8732), 1]
        conf_t: [batch_size, num_priors] ---> conf_t.view(-1, 1)): [batch_size * num_priors(8732), 1]  (index ---> batch_size * num_priors个1~num_objests的数)
        batch_conf: [batch_size * num_priors(8732), num_classes]
        torch.gather(input, dim, index, out=None) → Tensor 若dim=1，按行来，index为列坐标
        batch_conf.gather(1, conf_t.view(-1, 1))  dim=1,按行计算,conf_t.view(-1, 1)为index,取每行index下标的值

        softmax交叉熵损失：-y_i*log(f(x_i))
        log(f(x_i))=(x-x_max)-log(sum(exp(x-x_max)))
        -log(f(x_i))=log(sum(exp(x-x_max)))+(x_max-x)
        -y_i=1/0 负样本为1，正样本为0
        '''

        # Hard Negative Mining
        # 难负样本挖掘
        loss_c = loss_c.view(num, -1) # [batch_size,num_prior]  loss_c[batch_size * num_priors(8732), 1] ---> [batch_size,num_prior]
        loss_c[pos] = 0  # 将正样本的损失置为0 pos = conf_t > 0 [batch_size, num_priors]
        loss_c = loss_c.view(num, -1)  # [batch_size,num_prior]
        _, loss_idx = loss_c.sort(1, descending=True)  # [batch_size,num_prior] 按损失降序排列的prior box的位置(即损失排名第几(loss_idx下标)的是第几个prior box(存储内容)) dim=1 按行降序排列,保留降序数据的原下标
        _, idx_rank = loss_idx.sort(1)  # [batch_size,num_prior] 存储prior box损失的排名(第几个prior box(idx_rank下标)的损失排名第几(第idx_rank下标个prior box在loss_idx中的位置，loss_idx按损失降序排列))
        num_pos = pos.long().sum(1, keepdim=True)  # [batch_size,1] 每张图片的先验框中正样本的数目 pos.long() 将true转为1，false转为0 .sum() 按行求和
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)  # [batch_size, 1] 计算每张图片的负样本数目，最小为正样本数*negpos_ratio(3),最大为先验框数-1
        neg = idx_rank < num_neg.expand_as(idx_rank)  # [batch_size,num_prior] 将排名在num_neg位之前的prior box保留为难负样本 num_neg [batch_size, 1] ---> idx_rank [batch_size,num_prior]

        ''' ⬆ idx_rank < num_neg.expand_as(idx_rank) 
        loss_idx [batch_size,num_prior] 将num_prior个损失按降序排列，保存原始位置，原始位置为prior box位置(prior box标号，第几个prior box)
        idx_rank [batch_size,num_prior] 将num_prior个按损失降序排列的prior box位置(prior box标号)按prior box标号的升序排列，
                                             保存prior box标号在按损失降序排列的prior box标号中的原始位置，即该prior box按损失降序排列的排名
        num_neg  [batch_size, 1] ---> [batch_size,num_prior] 每张图片应取的对应的负样本数量*num_prior
        idx_rank < num_neg.expand_as(idx_rank) [batch_size,num_prior] num_prior个prior box按损失降序排列的排名与每张图片应取的对应的负样本数量比较，
                                                排名在每张图片应取的对应的负样本数量之前的样本为难负样本
        '''

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # [batch_size, num_priors, classes]  pos = conf_t > 0 [batch_size, num_priors]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # [batch_size, num_priors, classes]  neg = idx_rank < num_neg.expand_as(idx_rank) [batch_size,num_prior]
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)  # [_ , classes] _为pos_idx+neg_idx在batch_size*num_priors中选中的样本数(True) pos_idx+neg_idx ---> [batch_size, num_priors, classes] False+True=True=1>0
        targets_weighted = conf_t[(pos+neg).gt(0)]  # [_] _为pos+neg在batch_size*num_priors中选中的样本数(True)  pos+neg ---> [batch_size, num_priors] False+True=True=1>0
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)  # 计算正负样本的交叉熵损失 [_, classes] [_] size_average设置为False，则每个小批次的损失将被相加

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        # N = num_pos.data.sum()
        N = num_pos.data.sum().float()  # num_pos [batch_size,1] ---> bitch张图片中所有的正样本数
        loss_l /= N  # L_loc(x,l,g) / N
        loss_c /= N  # L_conf(x, c) / N
        return loss_l, loss_c  # 返回边框回归损失和置信损失
