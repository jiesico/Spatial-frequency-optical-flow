# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import csv
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from skimage import exposure,io
from PIL import ImageEnhance
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.depth_list = []
        self.cam_list = []
        # self.intrinsics_list = []



    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            # print(img1.shape)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            # print(self.flow_list[index])
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        depth1 = frame_utils.read_gen(self.depth_list[index][0])
        depth2 = frame_utils.read_gen(self.depth_list[index][1])
        # print('1',self.depth_list[index][0],self.depth_list[index][1])
        # print('2',self.image_list[index][0],self.image_list[index][1])

        depth1 = np.array(depth1).astype(np.float32)
        depth2 = np.array(depth2).astype(np.float32)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            # print(0)
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow, depth1, depth2 = self.augmentor(img1, img2, flow,depth1,depth2)
                # img1, img2, depth1, depth2 = self.augmentor(img1, img2, depth1,depth2)
        # print('1',depth1.shape,flow.shape,img1.shape)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()


        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        # print(depth1.min(),depth1.max(),depth1.mean(),np.median(depth1))
        return img1, img2, flow, valid.float(),depth1,depth2
        # return img1, img2,depth1,depth2

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/zhou/file8T/raft/Sintel/MPI-Sintel-complete', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        depth_root = osp.join(root, split, 'depth')
        image_root = osp.join(root, split, dstype)
        cam_root = osp.join(root, split, 'camdata_left')

        if split == 'test':
            self.is_test = True
        # os.path.exists(depth_root, scene, '某个具体文件')
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            # cam_list = sorted(glob(osp.join(cam_root, scene, '*.cam')))
            # print(os.path.exists(os.path.join(depth_root, scene, 'frame_0001.dpt')))
            # print(depth_list)
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]
                # self.extra_info += [ (scene, i) ] # scene and frame_id
                # print(cam_list[i])
                self.depth_list += [[depth_list[i], depth_list[i+1]]]
            # if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                # self.cam_list += [[cam_list[i], cam_list[i + 1]]]
                # self.depth_list += sorted(glob(osp.join(depth_root, scene, '*.dpt')))
        # print('1',self.depth_list)

# class MpiSintel(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='/home/zhou/file8T/raft/Sintel/MPI-Sintel-complete', dstype='clean'):
#         super(MpiSintel, self).__init__(aug_params)
#         flow_root = osp.join(root, split, 'flow')
#         depth_root = osp.join(root, split, 'depth')
#         image_root = osp.join(root, split, dstype)
#
#         if split == 'test':
#             self.is_test = True
#         # for idir, fdir, ddir in zip(image_dirs, flow_dirs, depth_dirs):
#         #     images = sorted(glob(osp.join(idir, '*.png')))
#         #     flows = sorted(glob(osp.join(fdir, '*.pfm')))
#         #     depths = sorted(glob(osp.join(ddir, '*.pfm')))
#
#         for scene,scenes,sceness in zip(image_root,depth_root,flow_root):
#             print('2',scene)
#             image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
#             depth_list = sorted(glob(osp.join(depth_root, scenes, '*.dpt')))
#             flow_list = sorted(glob(osp.join(flow_root, sceness, '*.flo')))
#             for i in range(len(flow_list)-1):
#                 self.image_list += [ [image_list[i], image_list[i+1]] ]
#                 # self.extra_info += [ (scene, i) ] # scene and frame_id
#                 self.depth_list += [ [depth_list[i], depth_list[i+1]] ]
#                 self.flow_list += [ [fow_list[i], flow_list[i+1]] ]
#             # if split != 'test':
#             #     self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/home/zhou/file8T/raft/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                depth_dirs = sorted(glob(osp.join(root,'disparity/TRAIN/*/*')))
                depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir,ddir in zip(image_dirs, flow_dirs,depth_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    depths= sorted(glob(osp.join(ddir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.depth_list += [ [depths[i], depths[i+1]] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.depth_list += [ [depths[i+1], depths[i]] ]

class FlyingThings3Dtest(FlowDataset):
    def __init__(self, aug_params=None, root='/home/zhou/file8T/raft/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3Dtest, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TESTtwo/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                depth_dirs = sorted(glob(osp.join(root,'disparity/TESTtwo/*/*')))
                depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TESTtwo/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir,ddir in zip(image_dirs, flow_dirs,depth_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    depths= sorted(glob(osp.join(ddir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.depth_list += [ [depths[i], depths[i+1]] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.depth_list += [ [depths[i+1], depths[i]] ]

class Tartanairtest(FlowDataset):
    def __init__(self, aug_params=None, root='/home/zhou/file8T/raft/FlyingThings3D', dstype='realtest4' ):
        super(Tartanairtest, self).__init__(aug_params)
        # print(split)
        for cam in ['left']:
            for direction in ['into_future']:
                image_dirs = sorted(glob(osp.join(root, 'frames_cleanpass',dstype, '*/*')))
                # print(image_dirs)
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                
                depth_dirs = sorted(glob(osp.join(root, 'disparity',dstype, '*/*')))
                depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])
                # print(image_dirs,depth_dirs)
                # flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TESTthree/*/*')))
                # flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                # print(len(image_dirs),len(depth_dirs))
                for idir, ddir in zip(image_dirs, depth_dirs):

                    images = sorted(glob(osp.join(idir, '*.png')) )
                    # print(images)
                    # flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    depths= sorted(glob(osp.join(ddir, '*.npy')) )
                    # print(images,depths)
		            
                    for i in range(len(images)-1):
                        # print(i)
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            # self.flow_list += [ flows[i] ]
                            self.depth_list += [ [depths[i], depths[i+1]] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            # self.flow_list += [ flows[i+1] ]
                            self.depth_list += [ [depths[i+1], depths[i]] ]
        print('0',len(self.image_list),len( self.depth_list))


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/zhou/file8T/raft/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        depth1 = sorted(glob(osp.join(root, 'disp/*_10.png')))
        depth2 = sorted(glob(osp.join(root, 'disp/*_11.png')))

    

        # calib_list = sorted(glob(osp.join(root, "calib_cam_to_cam/*.txt")))


        # for calib_file in calib_list:
        #     with open(calib_file) as f:
        #         reader = csv.reader(f, delimiter=' ')
        #         for row in reader:
        #             if row[0] == 'K_02:':
        #                 K = np.array(row[1:], dtype=np.float32).reshape(3,3)
        #                 kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        #                 self.intrinsics_list.append(kvec)

        for img1, img2,deps1,deps2 in zip(images1, images2,depth1,depth2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]
            self.depth_list += [ [deps1, deps2] ]
            # depth1 = frame_utils.read_gen(deps1)
            # depth2 = frame_utils.read_gen(deps2)
            # depth1 = np.array(depth1).astype(np.float32)
            # depth2 = np.array(depth2).astype(np.float32)
            # print('1',depth1.min(),depth1.max(),depth2.min(),depth2.max(),depth2.mean())
        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset =  clean_dataset + final_dataset
        # print(clean_dataset, final_dataset, train_dataset)
    if args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        # things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final
        # if TRAIN_DS == 'C+T+K+S+H':
        #     kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        #     hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        #     train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        # elif TRAIN_DS == 'C+T+K/S':
        #     train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

