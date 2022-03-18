import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        '''
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        num_classes = 21(VOC)
        bkg_label = 0 背景标签
        top_k = 200         考虑的最大预测边框数量
        conf_thresh = 0.01  置信阈值
        nms_thresh = 0.45   nms阈值
        '''
        self.num_classes = num_classes  # 21
        self.background_label = bkg_label  # 0
        self.top_k = top_k  # 200
        # Parameters used in nms.
        self.nms_thresh = nms_thresh  # 0.45
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh  # 0.01
        self.variance = cfg['variance']  # [0.1, 0.2]

    def forward(self, loc_data, conf_data, prior_data):
        '''
        self.softmax = nn.Softmax(dim=-1)
        output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1,self.num_classes)), self.priors.type(type(x.data)))
        loc_data = loc.view(loc.size(0), -1, 4)                                     [batch_size,num_priors,4]
        conf_data = self.softmax(conf.view(conf.size(0), -1,self.num_classes))      [batch_size * num_priors,num_classes]  conf.view(conf.size(0), -1,self.num_classes) [batch_size, num_priors,num_classes] ---> [batch_size * num_priors,num_classes]
        prior_data = self.priors.type(type(x.data)) = self.priors                   [num_priors, 4]
        '''
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch_size
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, self.num_classes, self.top_k, 5)  # [batch_size, 21, 200, 5]
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)  # [batch_size,num_classes,num_priors] [batch_size,21,8732]

        # Decode predictions into bboxes.
        for i in range(num):  # 将bitch_size张图片预测的坐标变换复原并转换为左上右下坐标
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # [num_priors,4] 将预测的坐标变换复原并转换为左上右下坐标 (loc_data[i]:[num_priors,4],prior_data:[num_priors,4] variance:[0.01,0.02])
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # [num_classes,num_priors] conf_preds[i] [num_classes,num_priors] 将预测置信度结果复制到conf_scores

            for cl in range(1, self.num_classes):  # 一个类对应num_prior个box
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # [num_priors] conf_scores:[num_classes,num_priors] conf_scores[i]:[num_priors] softmax置信度大于阈值的置为True 小于阈值的置为False
                scores = conf_scores[cl][c_mask]  # [_] _为num_prior个box中softmax置信度大于阈值的prior box总数 score存储置信度
                if scores.size(0) == 0:  # 如果这个类的prior box中没有置信度大于阈值的box，计算下一个类
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # [num_priors,4](True/False) c.mask:[num_priors](True/False) ---> decoded_boxes:[num_priors,4]
                boxes = decoded_boxes[l_mask].view(-1, 4)  # [_, 4] _为l_mask中True对应的box坐标行数(box数) decoded_boxes:[num_priors,4] l_mask:[num_priors,4](True/False)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)  # ([_, 4], [_], 0.45, 200)  idx：预测NMS后剩余的目标边框置信度分数位置，count：预测NMS后剩余的目标边框个数
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                boxes[ids[:count]]), 1)  # [batch_size, 21, 200, 5] 保留NMS后的边框对应置信度及边框坐标
        flt = output.contiguous().view(num, -1, 5)  # output ---> [batch_size, 21 * 200, 5]
        # view共享tensor, 因此, 对flt的修改也会反应到output上面
        _, idx = flt[:, :, 0].sort(1, descending=True)  # flt[:, :, 0] [batch_size, 21 * 200, 0]  NMS后边框的置信度，按降序排列，保存NMS后边框按降序排列的原位置(第几个边框)
        _, rank = idx.sort(1)  # [batch_size, 21 * 200] NMS后边框降序排列的排名
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0) 
        return output  # [batch_size, 21 * 200, 5]
