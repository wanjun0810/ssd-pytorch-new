from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import visdom


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



output_channel = {
    '300': [512, 1024, 1024, 512, 512, 256, 256, 256, 256, 256],
    '512': [],
}

gcd = {
    '300': [512, 512, 512, 512, 256, 256, 256, 256, 256, 256],
    '512': [],
}

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')  # data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate,pin_memory=True)
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')  # elif args.dataset == 'VOC':
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')  # data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate,pin_memory=True)
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')  # ssd_net.load_weights(args.resume)
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')  # data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate,pin_memory=True) 使用四个线程加载数据集
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')  # optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')  # optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')  # optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')  # adjust_learning_rate(optimizer, args.gamma, step_index)
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')  # 可视化
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.visdom:  # 可视化
    import visdom
    viz = visdom.Visdom()

def train():
    if args.dataset == 'COCO':  # 读取数据
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':  # 读取数据
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))



    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])  # (train,300,21) 建立SSD网络模型类
    net = ssd_net  # SSD网络模型

    if args.cuda:  # 调用GPU
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:  # 恢复训练点或直接加载权重
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)  # 恢复训练点
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)  # 加载VGG权重
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:  # 调用GPU
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)  # 初始化额外层卷积核权重       
        '''add code'''
        ssd_net.equal_channel_size.apply(weights_init)
        ssd_net.equal_channel.apply(weights_init)
        ssd_net.concat_operate.apply(weights_init)
        ''''''
        ssd_net.loc.apply(weights_init)  # 初始化定位层卷积核权重
        ssd_net.conf.apply(weights_init)  # 初始化分类层卷积核权重


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)  # 优化器初始化
    ''' optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        net.parameters()
        lr=args.lr=1e-4
        momentum=args.momentum=0.9
        weight_decay=args.weight_decay=5e-4
    '''
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)  # 损失函数初始化
    ''' MultiBoxLoss.__init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True)
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

    net.train()  # 网络的训练模式
    # loss counters
    loc_loss = 0  # 定位损失初始化为0
    conf_loss = 0  # 分类损失初始化为0
    epoch = 0  # 数据集完整迭代次数初始化
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size  # 每个epoch的batch块大小 数据集长度/batch_size(4/32/...)并向下取整
    print('Training SSD on:', dataset.name)  # VOC/COCO
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:  # 可视化
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)  # 加载数据集 , drop_last=True
    '''
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate,pin_memory=True)
    dataset=VOC/COCO
    args.batch_size=4/32
    num_workers=args.num_workers=4
    shuffle=True
    collate_fn=detection_collate   data/__init__.py def detection_collate(batch)
    pin_memory=True
    '''
    # create batch iterator
    batch_iterator = iter(data_loader)  # 创造迭代器 将一个epoch中所有图片每batch张图片组成一个迭代块
    for iteration in range(args.start_iter, cfg['max_iter']):  # 迭代次数，args.start_iter=0，cfg['max_iter']=120000(VOC)
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):  # 可视化
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0  # 重置定位损失
            conf_loss = 0  # 重置分类损失
            epoch += 1  # 迭代次数+1

        if iteration in cfg['lr_steps']:  # 'lr_steps': (80000, 100000, 120000) 迭代次数为lr_steps调整学习率
            step_index += 1  
            adjust_learning_rate(optimizer, args.gamma, step_index)  # 调整学习率
            '''
            optimizer=optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            args.gamma=0.1
            step_index
            '''

        # load train data
        # images, targets = next(batch_iterator)
        # 加载训练数据
        try:
            images, targets = next(batch_iterator)  # 从迭代器中获得训练图片和ground truth以及分类结果
        except StopIteration:  # 一个epoch训练结束开始下一次迭代
            batch_iterator = iter(data_loader)  # 开始下一个epoch迭代
            images, targets = next(batch_iterator)  # 从迭代器中获得训练图片和ground truth以及分类结果

        if args.cuda:  # 使用GPU加速计算
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()  # 获得开始时间
        '''修改代码'''
        # out = net(images)  # 前向传播，预测
        # with torch.no_grad():
        out = net(images, output_channel['300'], gcd['300'])  # 前向传播，预测
        # backprop
        optimizer.zero_grad()  # 将反向传播的梯度置为0
        loss_l, loss_c = criterion(out, targets)  # 计算预测结果与目标结果之间的损失 MultiBoxLoss()
        loss = loss_l + loss_c  # 计算定位和分类的总损失
        loss.backward()  # 损失反向传播结果
        optimizer.step()  # 更新梯度
        t1 = time.time()  # 获得结束时间
        # loc_loss += loss_l.data[0]
        loc_loss += loss_l.item()  # 计算定位总损失
        # conf_loss += loss_c.data[0]
        conf_loss += loss_c.item()  # 计算分类总损失

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))  # batch张图片的训练时间
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')  # 输出总损失

        if args.visdom:  # 可视化
            # update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
            #                 iter_plot, epoch_plot, 'append')
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:  # 每训练3000次保存一次权重
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')  # 训练结束保存权重


def adjust_learning_rate(optimizer, gamma, step):  # 调整学习率
    '''
    optimizer=optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    args.gamma=0.1
    step_index
    '''
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))  # lr = rgs.lr(1e-4)*(0.1^step)
    for param_group in optimizer.param_groups:  # 更改optimizer(优化器)的学习率
        param_group['lr'] = lr


def xavier(param):  # xavier初始化
    init.xavier_uniform(param)  # xavier初始化


def weights_init(m):  # 权重初始化
    if isinstance(m, nn.Conv2d):  # 初始化卷积核权重及偏置
        xavier(m.weight.data)  # 初始化卷积核权重
        m.bias.data.zero_()  # 初始化卷积核偏置


def create_vis_plot(_xlabel, _ylabel, _title, _legend):  # 可视化
    return viz.line(
        # X=torch._C.zeros((1,)).cpu(),
        # Y=torch._C.zeros((1, 3)).cpu(),
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):  # 可视化
    viz.line(
        # X=torch._C.ones((1, 3)).cpu()*iteration,
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            # X=torch._C.zeros((1, 3)).cpu(),
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
