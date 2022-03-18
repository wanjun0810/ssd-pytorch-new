# -*- coding: utf-8 -*-
import torch


def point_form(boxes):  # 将prior box坐标转换为左上右下形式
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin 
                     boxes[:,:2] + boxes[:, 2:] / 2), 1)  # xmax, ymax
    # xmin = cx-w/2, ymin = cy-h/2, xmax = cx+w/2, ymax = cy+w/2


def center_size(boxes):  # 将左上右下形式坐标转换为中心坐标加宽高
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h
    # cx = (xmin+xmax)/2, cy = (ymin+ymax)/2, w = xmax-xmin, h = ymax-ymin


def intersect(box_a, box_b):  # 计算prior box与ground truth的交集
    '''
    box_a [num_objects,4]
    box_b [num_priors,4]
    '''
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)  # A = num_objects
    B = box_b.size(0)  # B = num_priors
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))  # ground truth和prior box中xmax,ymax的最小值
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))  # ground truth和prior box中xmin,ymin的最大值
    inter = torch.clamp((max_xy - min_xy), min=0)  # ground truth和prior box的交集(x,y->宽, 高)
    return inter[:, :, 0] * inter[:, :, 1]  # ground truth和prior box的交集(xy->宽*高) [num_objects, num_priors]


def jaccard(box_a, box_b):  # 计算prior box与ground truth的交并比
    '''
    box_a [num_objects,4]
    box_b [num_priors,4]
    '''
    """
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)  # box_a和box_b的交集
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # ground truth (xmax-xmin)(ymax-ymin) [num_objects, num_priors]
    # (box_a[:, 2]-box_a[:, 0]) *  (box_a[:, 3]-box_a[:, 1])                                  [num_objects]
    # (box_a[:, 2]-box_a[:, 0]) *  (box_a[:, 3]-box_a[:, 1]).unsqueeze(1)                     [num_objects, 1]
    # (box_a[:, 2]-box_a[:, 0]) *  (box_a[:, 3]-box_a[:, 1]).unsqueeze(1).expand_as(inter)    [num_objects, num_priors]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # prior box (xmax-xmin)(ymax-ymin) [num_objects, num_priors]
    # (box_b[:, 2]-box_b[:, 0]) *  (box_b[:, 3]-box_b[:, 1])                                  [num_priors]
    # (box_b[:, 2]-box_b[:, 0]) *  (box_b[:, 3]-box_b[:, 1]).unsqueeze(0)                     [1, num_priors]
    # (box_b[:, 2]-box_b[:, 0]) *  (box_b[:, 3]-box_b[:, 1]).unsqueeze(0).expand_as(inter)    [num_objects, num_priors]
    union = area_a + area_b - inter  # 求ground truth和prior box的并集
    return inter / union   # ground truth和prior box的交并比 [num_objects, num_priors]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''
    match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
    threshold = 0.5
    truths = targets[idx][:, :-1].data    [idx][num_objs, :-1] -> [idx][num_objs, 0-3]  ground truth 坐标
    priors = defaults = priors.data       [num_priors, 4] prior box 坐标
    variances = cfg['variance']           [0.1, 0.2]
    labels = targets[idx][:, -1].data     [idx][num_objs, 4]  ground truth 类别
    loc_t = torch.Tensor(num, num_priors, 4)    [batch_size, num_priors, 4] 存储batch_size张图片中的每一张图片的8732个num_priors匹配的ground truth坐标
    conf_t = torch.LongTensor(num, num_priors)  [batch_size, num_priors] 存储batch_size张图片中的每一张图片的8732个num_priors匹配的ground truth类别
    idx = 图片索引
    '''
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(priors))  # [num_objects, num_priors] 计算ground truth与prior box的IoU阈值
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # [num_objects, 1] 1->按行  best_prior_idx：ground truth匹配的最好的prior box下标  best_prior_overlap：ground truth与匹配的最好的prior box的阈值
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # [1, num_priors]  0->按列  best_truth_idx：prior box匹配的最好的ground truth下标  best_truth_overlap：prior box与匹配的最好的ground truth的阈值
    best_truth_idx.squeeze_(0)                      # [num_priors]
    best_truth_overlap.squeeze_(0)                  # [num_priors]
    best_prior_idx.squeeze_(1)                      # [num_objects]
    best_prior_overlap.squeeze_(1)                  # [num_objects]
    '''确保每个ground truth有匹配的最好的prior box，将best_prior代入到best_truth中，best_prior为中间变量⬇'''
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior  将ground truth匹配的最好的prior box(索引为best_prior_idx)与匹配的ground truth的阈值置为2, 在所有prior box与匹配的最好的ground truth的阈值中
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):  # best_prior_idx.size(0) -> num_objects
        best_truth_idx[best_prior_idx[j]] = j  # best_prior_idx[j] -> 将ground truth(j)匹配的最好的prior box(best_prior_idx[j])匹配的最好的ground truth下标(best_truth_idx)置为ground truth下标(j)
    '''确保每个ground truth有匹配的最好的prior box，将best_prior代入到best_truth中，best_prior为中间变量⬆'''
    matches = truths[best_truth_idx]          # [num_priors,4] 将ground truth坐标按照best_truth_idx的内容填充，存储到matches，num_priors个ground truth坐标
    conf = labels[best_truth_idx] + 1         # [num_priors] 将ground truth类别按照best_truth_idx的内容填充，存储到conf，num_priors个ground truth类别
    # [num_priors]或[num_priors, 4]中的第i行内容对应第i个prior box，i<num_priors
    conf[best_truth_overlap < threshold] = 0  # label as background  在num_priors个ground truth类别中，将prior box与匹配的最好的ground truth的阈值(best_truth_overlap)小于0.5的conf置为0,标为背景
    loc = encode(matches, priors, variances)  # [num_priors,4] 将 box 信息编码成小数形式, 方便网络训练，同时获得边框回归目标，中心坐标+宽高
    # ground truth + prior box -> loc truth
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn 存储ground truth 与 prior box对应的要学习的边框坐标
    conf_t[idx] = conf  # [num_priors] top class label for each prior 存储prior box对应的ground truth的类别，0为背景

    ''' ⬆ best_truth_overlap.index_fill_(0, best_prior_idx, 2) + best_truth_idx[best_prior_idx[j]] = j for j in range(best_prior_idx.size(0))
    确保每个ground truth有匹配的最好的prior box，将best_prior代入到best_truth中，best_prior为中间变量
    best_prior_overlap, best_prior_idx  [num_objects]   best_prior_overlap：ground truth与匹配的最好的prior box的阈值
                                                        best_prior_idx：ground truth匹配的最好的prior box下标
    best_truth_overlap, best_truth_idx  [num_priors]    best_truth_overlap：prior box与匹配的最好的ground truth的阈值
                                                        best_truth_idx：prior box匹配的最好的ground truth下标
    best_truth_overlap.index_fill_(0, best_prior_idx, 2) 在所有prior box与匹配的最好的ground truth的阈值(best_truth_overlap)中,
                                                         将与ground truth匹配的最好的prior box(索引为best_prior_idx)与匹配的ground truth的阈值置为2
    for j in range(best_prior_idx.size(0)):     best_prior_idx.size(0) = num_objects
        best_truth_idx[best_prior_idx[j]] = j   将ground truth(j)匹配的最好的prior box(best_prior_idx[j])匹配的最好的ground truth下标(best_truth_idx)置为ground truth下标(j)
    '''


def encode(matched, priors, variances):  # 将 box 信息编码成小数形式, 方便网络训练，中心坐标+宽高
    '''
    matches = truths[best_truth_idx]  [num_priors,4] num_priors个ground truth坐标
    priors = defaults = priors.data   [num_priors, 4] num_priors个prior box 坐标
    variances = cfg['variance']       [0.1, 0.2]
    '''
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]  # ground_truth(xmin+xmax,ymin+ymax)/2-prior_box(xmax,ymax)
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])    # (ground_truth(xmin+xmax,ymin+ymax)/2-prior_box(xmax,ymax))/(0.1*prior_box(xmin,ymin))
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # (ground_truth(xmax,ymax)-ground_truth(xmin,ymin))/prior_box(xmax,ymax)
    g_wh = torch.log(g_wh) / variances[1]  # log((ground_truth(xmax,ymax)-ground_truth(xmin,ymin))/prior_box(xmax,ymax))/0.2
    # log 底数为 e
    # return target for smooth_l1_loss 
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4] 边框回归损失目标，中心坐标+宽高


''' ⬇ decode(loc, priors, variances)
中心坐标+宽高<--->左上右下坐标
(cx,cy)=(xmin+xmax,ymin+ymax)/2
(w,h)=(xmax-xmin,ymax-ymin)
(xmin,ymin)=((cx,cy)-(w,h))/2=(xmin+xmax-(xmax-xmin),ymin+ymax-(ymax-ymin))/2=(2xmin,2ymin)/2=(xmin,ymin)
(xmax,ymax)=(w,h)-(xmin,ymin)=(xmax-xmin+xmin,ymax-ymin+ymin)=(xmax,ymax)

loc_data （中心坐标+宽高+变换）计算方式与复原方式：
g_wh计算：
g_wh1 = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
g_wh2 = log(g_wh1)/0.2           log 底数为 e , e^(g_wh2*0.2)=g_wh1
g_wh复原：
e^(loc_data(w,h)*0.2) = g_wh1
prior_box(w,h)*(e^(loc_data(w,h)*0.2)) = prior_box(w,h) * g_wh1 = priors[:, 2:] * g_wh1 = (matched[:, 2:] - matched[:, :2])
g_cxcy计算：
g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
g_cxcy1 = g_cxcy / (0.1 * priors[:, 2:]) = g_cxcy / 0.1 / priors[:, 2:]
g_cxcy复原：
loc_data(cx,cy) * 0.1 * (prior_box(cx,cy)) = loc_data(cx,cy) * 0.1 * priors[:, 2:] = g_cxcy
prior_box(cx,cy)+loc_data(cx,cy)*0.1*(prior_box(cx,cy)) = prior_box(cx,cy) + g_cxcy = priors[:, :2] + g_cxcy = (matched[:, :2] + matched[:, 2:])/2

坐标复原：
boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],priors[:, 2:]*torch.exp(loc[:, 2:]*variances[1])), 1)
      = prior_box(cx,cy)+loc_data(cx,cy)*0.1*(prior_box(cx,cy)),prior_box(w,h)*(e^(loc_data(w,h)*0.2))
      = (matched[:, :2] + matched[:, 2:])/2, (matched[:, 2:] - matched[:, :2])
boxes[:, :2] -= boxes[:, 2:] / 2  # xmin,ymin  # (matched[:, :2] + matched[:, 2:])/2, (matched[:, 2:] - matched[:, :2])
boxes[:, :2] = (matched[:, :2] + matched[:, 2:])/2 - (matched[:, 2:] - matched[:, :2])/2
             = ((xmin,ymin)+(xmax,ymax))/2 - ((xmax,ymax)-(xmin,ymin))/2 
             = ((xmin+xmax,ymin+ymax)-(xmax-xmin,ymax-ymin))/2
             = (2xmin,2ymin)/2
             = (xmin,ymin)
boxes[:, 2:] += boxes[:, :2]  # xmax,ymax  # (xmin,ymin),(matched[:, 2:] - matched[:, :2])
boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
             = (matched[:, 2:] - matched[:, :2]) + (xmin,ymin)
             = ((xmax,ymax) - (xmin,ymin)) + (xmin,ymin)
             = (xmax,ymax)
'''


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):  # 将中心坐标+宽高转换为左上右下坐标
    '''
    decoded_boxes = decode(loc_data[i], prior_data, self.variance)
    loc = loc_data[i]     [num_priors,4] num_priors个loc坐标(定位卷积层输出)
    priors = prior_data   [num_priors, 4] num_priors个prior box 坐标
    variances = cfg['variance']       [0.1, 0.2]
    '''
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:]*torch.exp(loc[:, 2:]*variances[1])), 1)  # (matched[:, :2] + matched[:, 2:])/2, (matched[:, 2:] - matched[:, :2])
    boxes[:,:2] -= boxes[:, 2:] / 2  # xmin,ymin
    # (xmin,ymin),(matched[:, 2:] - matched[:, :2])
    boxes[:, 2:] += boxes[:, :2]  # xmax,ymax
    return boxes  # [num_priors,4] 返回左上右下坐标


def log_sum_exp(x):
    #　batch_conf [batch_size * num_priors(8732), num_classes]
    # 计算softmax损失，同时防止上下溢出
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()  # 取[batch_size * num_priors(8732), num_classes]中最大一个值
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max  # [batch_size * num_priors(8732), 1]  log(softmax(x))


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    '''
    ids, count = nms(boxes, scores, self.nms_thresh(0.45), self.top_k(200))
    boxes           [_, 4] _为l_mask中True对应的box坐标行数(num_prior个box中softmax置信度大于阈值的prior box总数) / [num_priors,4]
    scores          [_] _为num_prior个box中softmax置信度大于阈值的prior box总数 保存置信度 / [num_priors]
    overlap=0.45    nms的IoU阈值为0.45 / 0.5
    top_k=200       考虑的最大预测边框数量
    '''
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()  # [_] _为num_prior个box中softmax置信度大于阈值的prior box总数(与boxes行数一样) 内容为0
    if boxes.numel() == 0:  # boxes.numel() 返回tensor变量内所有元素个数，若为0则无坐标 返回0
        return keep
    x1 = boxes[:, 0]  # [_] _个box坐标的xmin
    y1 = boxes[:, 1]  # [_] _个box坐标的ymin
    x2 = boxes[:, 2]  # [_] _个box坐标的xmax
    y2 = boxes[:, 3]  # [_] _个box坐标的ymax
    area = torch.mul(x2 - x1, y2 - y1)  # [_] _个box的面积 (x2-x1)*(y2-y1) ---> (xmax-xmin)*(ymax-ymin)
    v, idx = scores.sort(0)  # sort in ascending order [_], [_] 将_个box置信度按升序排列，idx保存升序置信度的原下标，idx的下标即为保存的box的置信度排名
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals 倒序取idx的top_k个置信度的原下标，即取top_k个置信度较高的box的置信度
    xx1 = boxes.new()  # [] 空tensor
    yy1 = boxes.new()  # []
    xx2 = boxes.new()  # []
    yy2 = boxes.new()  # []
    w = boxes.new()  # []
    h = boxes.new()  # []

    # keep = torch.Tensor()
    count = 0  # 处理后的置信度分数对应的位置的个数
    while idx.numel() > 0:  # 还有置信度较高的分数对应的位置没处理完
        i = idx[-1]  # index of current largest val  i = 最大的置信度的位置
        # keep.append(i)
        keep[count] = i  # 将最大的置信度的位置加到keep
        count += 1  # count+1
        if idx.size(0) == 1:  # 如果置信度较高的分数对应的位置只有一个，处理结束，跳出循环
            break
        idx = idx[:-1]  # remove kept element from view  把处理过的置信度分数对应的位置移出idx数组
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)  # [_-count] 将x1(xmin)中的数据按idx中存储的置信度分数对应的位置放到xx1
        torch.index_select(y1, 0, idx, out=yy1)  # [_-count] 将y1(ymin)中的数据按idx中存储的置信度分数对应的位置放到yy1
        torch.index_select(x2, 0, idx, out=xx2)  # [_-count] 将x2(xmax)中的数据按idx中存储的置信度分数对应的位置放到xx2
        torch.index_select(y2, 0, idx, out=yy2)  # [_-count] 将y2(ymax)中的数据按idx中存储的置信度分数对应的位置放到yy2
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])  # [_-count] 控制数据范围，最小为x1[i]，x1[i]为置信度最高的box的xmin
        yy1 = torch.clamp(yy1, min=y1[i])  # [_-count] 控制数据范围，最小为y1[i]，y1[i]为置信度最高的box的ymin
        xx2 = torch.clamp(xx2, max=x2[i])  # [_-count] 控制数据范围，最小为x2[i]，x2[i]为置信度最高的box的xmax
        yy2 = torch.clamp(yy2, max=y2[i])  # [_-count] 控制数据范围，最小为y2[i]，y2[i]为置信度最高的box的ymax
        '''xx1,yy1,xx2,yy2 确定当前各个box与置信度最高的box的交集范围的左上右下坐标'''
        w.resize_as_(xx2)  # [_-count] 0
        h.resize_as_(yy2)  # [_-count] 0
        w = xx2 - xx1  # [_-count] xmax-xmin 当前各个box与置信度最高的box的交集的宽
        h = yy2 - yy1  # [_-count] ymax-ymin 当前各个box与置信度最高的box的交集的高
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)  # [_-count] 控制数据范围，最小为0
        h = torch.clamp(h, min=0.0)  # [_-count] 控制数据范围，最小为0
        inter = w*h  # [_-count]  当前各个box与当前置信度最高的box的交集面积 w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # [_-count] area:[_] 将area中的_个box面积按当前idx中存储的置信度分数对应的位置放到rem_area
        union = (rem_areas - inter) + area[i]  # [_-count] _-count个box与当前置信度最高的box的并集面积 面积-交集面积+最高的面积
        IoU = inter/union  # store result in iou [_-count] _-count个box与当前置信度最高的box的交并比
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]  # 保留IoU阈值小于overlap的置信度分数位置
    return keep, count  # keep：预测NMS后剩余的目标边框置信度分数位置，count：预测NMS后剩余的目标边框个数
