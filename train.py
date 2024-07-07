from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 500
VAL_FREQ = 2000


def sequence_loss(flow_preds, flow_gt, valid,loss_feature, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    # print(valid.shape,valid.max(),valid.min())
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss_feature': loss_feature.item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
def show_image(image,name):
    print(image.shape)
    image = image.cpu().numpy()
    image = (image).astype(np.uint8)
    # print(image.shape)
    filename = name
    cv2.imwrite(filename, image)
    # cv2.imshow('image', image)
    cv2.waitKey(1)
def processdepth(depths):
    # print('1',depths.max(),depths.min(),depths.mean())
    depths = depths.unsqueeze(1)
    # depths = depths.repeat(1,3,1,1)
    depthsss = depths
    # print(depthsss.max(),depthsss.mean(),depthsss.min())
    depthsss[depthsss > 900] = 0.02
    # print('2',depths.max(),depths.min(),depths.mean())
    means = depthsss[depthsss > 0.01].mean()
    depthsss = depthsss / means
    return depthsss
def load_model_state_dict(model, checkpoint_path, requires_grad=False):
    loaded_state_dict = torch.load(checkpoint_path, map_location='cpu')
    # 从加载的state_dict()中提取现有模型中存在的参数
    state_dict = {k: v for k, v in loaded_state_dict.items() if k in model.state_dict()}
    # 输出加载了哪些参数
    print("The following parameters are loaded from checkpoint:")
    for name, param in model.named_parameters():
        if name in state_dict:
            param.requires_grad = requires_grad  # 设置为需要/不需要梯度
            print(name)
        else:
            param.requires_grad = True
            print(f"Parameter {name} is not found in checkpoint.")
    # 将提取出来的参数加载到现有模型中
    model.load_state_dict(state_dict, strict=False)
def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        # load_model_state_dict(model, args.restore_ckpt, requires_grad=True)
    model.cuda()
    model.train()
    print({sum([x.nelement() for x in model.parameters()])/1000000.} )
    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 2000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid,depth1,depth2 = [x.cuda() for x in data_blob]
            # print('1',i_batch)
            # print(image1.shape,flow.shape,valid.shape)#[6, 3, 400, 720],[6, 2, 400, 720],[6, 400, 720]
            # if args.add_noise:
            # show_image(depth1[0,:,:],'2-image0.png')
            # show_image(image1[1,:,:,:],'2-image1.png')
            # show_image(image1[2,:,:,:],'2-image2.png')
            # show_image(image1[3,:,:,:],'2-image3.png')
            # show_image(image1[4,:,:,:],'2-image4.png')
            # show_image(image1[5,:,:,:],'2-image5.png')

            # stdv1 = np.random.randint(0, 10)#uniform
            # if stdv1 == 0:
            #     image1 = image1 * stdv1
            #     image2 = image2 * stdv1
            # else:
            #      image1 = image1 / stdv1
            #      image2 = image2 / stdv1
            # show_image(image1[0,:,:,:],'2-image1.jpg')
            # show_image(image1[1,:,:,:],'2-image1.png')
            # show_image(image1[2,:,:,:],'2-image2.png')
            # show_image(image1[3,:,:,:],'2-image3.png')
            # show_image(image1[4,:,:,:],'2-image4.png')
            # show_image(image1[5,:,:,:],'2-image5.png')

            stdv = np.random.randint(0, 35)#uniform
            image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            # show_image(image1[0,:,:,:],'2-image0noise.png')
            # show_image(image1[1,:,:,:],'2-image1noise.png')
            # show_image(image1[2,:,:,:],'2-image2noise.png')
            # show_image(image1[3,:,:,:],'2-image3noise.png')
            # show_image(image1[4,:,:,:],'2-image4noise.png')
            # show_image(image1[5,:,:,:],'2-image5noise.png')
            # show_image(image1[3,:,:,:])
            # image1 = image1/stdv1
            # image2 = image2/stdv1
            depth1 = processdepth(depth1)
            depth2 = processdepth(depth2)
            # print(image1.shape,depth1.shape)#torch.Size([6, 3, 400, 720]) torch.Size([6, 400, 720])
            flow_predictions,loss_feature = model(image1, image2,depth1,depth2, iters=args.iters)

            # print(flow_predictions[0].shape)#[6, 2, 400, 720]

            # loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            loss, metrics = sequence_loss(flow_predictions, flow, valid,loss_feature.sum(), args.gamma)
            # loss = loss + 0.2*loss_feature.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        # print(0)
                        # results.update(evaluate.validate_kitti(model.module))
                        # if total_steps / VAL_FREQ > 13:
                        evaluate.validate_flything3D(model.module)
                        # evaluate.validate_kitti(model.module)
                        evaluate.validate_flything3Done(model.module)
                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)