import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()
        self.channel_wise_adaptations = nn.Linear(256, 256) 
        self.spatial_wise_adaptations = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, f_s, f_t):
        # distance_loss = self.cosine_similarity_loss(f_s, f_t)
        other_loss = self.new_loss(f_s, f_t)
        return other_loss
    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        pixel_wise_euclidean_distance = torch.norm(output_net - target_net, p=2, dim=1)
        mean_pixel_wise_euclidean_distance = pixel_wise_euclidean_distance.sum()
        return 0.0001*mean_pixel_wise_euclidean_distance
    def new_loss(self, teacher_feat, student_feat):#torch.Size([2, 256, 50, 90]) torch.Size([2, 256, 50, 90])

        dist_loss = self.cosine_similarity_loss(teacher_feat, student_feat)
        # kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        loss_dict = dict()

        student_B, student_C, student_H, student_W = student_feat.size()
        teacher_B, teacher_C, teacher_H, teacher_W = teacher_feat.size()
        assert student_B == teacher_B
        B = student_B

        # FIXME how to project student channel to teacher channel in a natural manner?
        kd_channel_loss += torch.dist(torch.mean(teacher_feat, [2, 3]),self.channel_wise_adaptations(torch.mean(student_feat, [2, 3])))*0.2 #* kd_spatial_loss_weight # 4e-3 * 6
        t_spatial_pool = torch.mean(teacher_feat, [1], keepdim=True).view(teacher_B, 1, teacher_H, teacher_W)
        s_spatial_pool = torch.mean(student_feat, [1], keepdim=True).view(student_B, 1, student_H, student_W)
        kd_spatial_loss += torch.dist(t_spatial_pool,
                            self.spatial_wise_adaptations(s_spatial_pool))*0.02 #* kd_spatial_loss_weight # 4e-3 * 6
        # print(dist_loss,kd_channel_loss,kd_spatial_loss)
        return dist_loss + kd_channel_loss + kd_spatial_loss

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        # args.small = True
        self.correlations = PKT()
        if args.small:
            print(1)
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False
        # self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        # self.cnet = BasicEncoder(output_dim=256, norm_fn='none')#256
        # feature network, context network, and update block
        if args.small:
            print(2)
            self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            # print(0)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)#batch
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.conv_layers2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_layers3 = nn.Conv2d(512, 256, kernel_size=1)
            # print(1)
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2,depth1,depth2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        # print(image1.shape,depth1.shape,depth1.max())
        image1 = [image1,depth1]
        image2 = [image2,depth2]
        hdim = self.hidden_dim
        cdim = self.context_dim
        # print(image1.shape)
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
########################################################################            
            fmap1_a,fmap1_b = self.fnet(image1)
            fmap2_a,fmap2_b = self.fnet(image2)
            fmap1 = torch.cat([fmap1_a, fmap1_b], 1)
            fmap2 = torch.cat([fmap2_a, fmap2_b], 1)
            fmap1 = self.conv_layers2(fmap1.float())
            fmap2 = self.conv_layers2(fmap2.float())
            normalized_fmap1_a = F.normalize(fmap1_a, p=2, dim=1)
            normalized_fmap1_b = F.normalize(fmap1_b, p=2, dim=1)
            normalized_fmap2_a = F.normalize(fmap2_a, p=2, dim=1)
            normalized_fmap2_b = F.normalize(fmap2_b, p=2, dim=1)
            print(normalized_fmap1_a.shape,normalized_fmap1_b.shape)
            loss_feature = self.correlations(normalized_fmap1_a,normalized_fmap1_b) + self.correlations(normalized_fmap2_a,normalized_fmap2_b)
########################################################################
        #loss_feature = 0.0
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet1,cnet2 = self.cnet(image1)
            cnet = torch.cat([cnet1, cnet2], 1)
            cnet = self.conv_layers3(cnet.float())
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # print('3',net.shape,inp.shape)
        # coords0, coords1 = self.initialize_flow(image1[0])
        coords0, coords1 = self.initialize_flow(image1[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions,loss_feature
