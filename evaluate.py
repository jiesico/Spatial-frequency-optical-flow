import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import datasets
from utils import flow_viz
from utils import frame_utils
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
from  torchvision import utils as vutils

import numpy as np
import cv2
# import Image

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

def processdepthssintel(depths):
    depths = torch.from_numpy(depths).cuda()
    # print(depths.max(),depths.min(),depths.mean())
    depths = depths.unsqueeze(0)
    depths = depths.unsqueeze(1)
    # depths = depths.repeat(1,3,1,1)
    depthsss = depths
    depthsss[depthsss > 100] = 2
    # means = depthsss[depthsss > 0.001].mean()
    # depthsss = depthsss / means
    return depthsss
@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean','final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt,valid_gt, depth1, depth2 = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            # print(depth1.max(),depth1.min(),depth1.mean())
            # print(depth1.shape,depth2.shape)
            depth1 = cv2.resize(depth1, (image1.shape[3], image1.shape[2]))
            depth2 = cv2.resize(depth2, (image2.shape[3], image2.shape[2]))
            depth1 = processdepthssintel(depth1)
            depth2 = processdepthssintel(depth2)
            # print('1',depth1.max(), depth1.min(), depth1.mean())
            # print('2',depth2.max(), depth2.min(), depth2.mean())
            stdv1 = val_id % 10  # uniform
            # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            # print(stdv1)
            # if stdv1 == 0:
            #     image1 = image1 * stdv1
            #     image2 = image2 * stdv1
            # else:
            #     image1 = image1 / stdv1
            #     image2 = image2 / stdv1
            flow_low, flow_pr = model(image1, image2, depth1, depth2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results
def calculate_average_diff_from_file(visresult_path):
    result_file = os.path.join(visresult_path, "result.txt")

    # 检查结果文件是否存在
    if not os.path.exists(result_file):
        print("结果文件不存在")
        return None

    # 读取结果文件中的数据
    median_diff_values = []
    std_diff_values = []
    average_diff_values = []
    median_diff_all_values = []
    std_diff_all_values = []
    average_diff_all_values = []

    with open(result_file, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) == 7:  # 假设每行有七个值
                median_diff_values.append(float(parts[1]))  # 提取剩余像素差的中位数
                std_diff_values.append(float(parts[2]))  # 提取剩余像素差的标准差
                average_diff_values.append(float(parts[3]))  # 提取剩余像素差的平均值
                median_diff_all_values.append(float(parts[4]))  # 提取全部像素差的中位数
                std_diff_all_values.append(float(parts[5]))  # 提取全部像素差的标准差
                average_diff_all_values.append(float(parts[6]))  # 提取全部像素差的平均值

    # 计算平均值
    if len(median_diff_values) > 0 and len(std_diff_values) > 0 and len(average_diff_values) > 0:
        median_diff_mean = np.mean(median_diff_values)
        std_diff_mean = np.mean(std_diff_values)
        average_diff_mean = np.mean(average_diff_values)
        median_diff_all_mean = np.mean(median_diff_all_values)
        std_diff_all_mean = np.mean(std_diff_all_values)
        average_diff_all_mean = np.mean(average_diff_all_values)

        print('Median Difference Mean (Filtered):', median_diff_mean)
        print('Standard Deviation Difference Mean (Filtered):', std_diff_mean)
        print('Average Difference Mean (Filtered):', average_diff_mean)
        print('Median Difference Mean (All):', median_diff_all_mean)
        print('Standard Deviation Difference Mean (All):', std_diff_all_mean)
        print('Average Difference Mean (All):', average_diff_all_mean)

        return median_diff_mean, std_diff_mean, average_diff_mean, median_diff_all_mean, std_diff_all_mean, average_diff_all_mean

    return None

import numpy as np
import os

def compute_pixel_difference(projected_frame, image2, i, visresult_path):
    # 计算像素差
    pixel_diff = np.abs(projected_frame - image2).astype(float)

    # 计算剩余像素差的阈值
    threshold = np.quantile(pixel_diff, 0.85)

    # 保留小于阈值的像素差
    pixel_diff_filtered = pixel_diff[pixel_diff <= threshold]

    # 计算剩余像素差的中位数
    median_diff = np.median(pixel_diff_filtered)

    # 计算剩余像素差的标准差
    std_diff = np.std(pixel_diff_filtered)

    # 计算剩余像素差的平均值
    average_diff = np.mean(pixel_diff_filtered)

    # 检查是否存在结果文件夹，如果不存在则创建
    if not os.path.exists(visresult_path):
        os.makedirs(visresult_path)

    # 构造结果文件路径
    result_file = os.path.join(visresult_path, 'result.txt')

    # 检查结果文件是否存在，如果不存在则创建
    if not os.path.exists(result_file):
        with open(result_file, 'w') as file:
            file.write("")

    # 将结果写入 txt 文件
    with open(result_file, "a") as file:
        file.write(f"{i} {median_diff} {std_diff} {average_diff} ")

    # 计算全部像素差的中位数
    median_diff_all = np.median(pixel_diff)

    # 计算全部像素差的标准差
    std_diff_all = np.std(pixel_diff)

    # 计算全部像素差的平均值
    average_diff_all = np.mean(pixel_diff)

    with open(result_file, "a") as file:
        file.write(f"{median_diff_all} {std_diff_all} {average_diff_all}\n")

    # print("Median Difference (Filtered):", median_diff)
    # print("Standard Deviation Difference (Filtered):", std_diff)
    # print("Average Difference (Filtered):", average_diff)

    # print("Median Difference (All):", median_diff_all)
    # print("Standard Deviation Difference (All):", std_diff_all)
    # print("Average Difference (All):", average_diff_all)

    return median_diff, std_diff, average_diff, median_diff_all, std_diff_all, average_diff_all




def project_pixels_with_optical_flow(frame1, frame2, flow, i,visresult_path):
    # 转换为NumPy数组
    frame1 = frame1.detach().cpu().numpy()[0].transpose((1, 2, 0))
    frame2 = frame2.detach().cpu().numpy()[0].transpose((1, 2, 0))
    flow = flow.detach().cpu().numpy()
    i = 0
    # 转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 创建一个与第二帧相同大小的图像，用于存储投影结果
    projected_frame = np.zeros_like(gray2)

    # 遍历第一帧的每个像素位置
    for y in range(gray1.shape[0]):
        for x in range(gray1.shape[1]):
            # 获取光流向量
            u, v = flow[:, y, x]

            # 计算在第二帧上的新像素位置
            new_x = int(x + u)
            new_y = int(y + v)

            # 判断新位置是否在第二帧的范围内
            if 0 <= new_x < frame2.shape[1] and 0 <= new_y < frame2.shape[0]:
                # 将第一帧的像素值投影到第二帧上
                projected_frame[new_y, new_x] = gray1[y, x]
            else:
                i = i + 1

    # 创建结果保存文件夹
    # result_folder = 'resultvis'
    # if not os.path.exists(result_folder):
    #     os.makedirs(result_folder)

    # 构造结果图像路径
    result_path = os.path.join(visresult_path, f'result_{i:04d}.jpg')
    compute_pixel_difference(projected_frame, gray2, i,visresult_path)
    print(i)
    # 保存结果图像
    cv2.imwrite(result_path, projected_frame)



def processdepths(depths):
    depths = torch.from_numpy(depths).cuda()
    # print(depths.max(),depths.min(),depths.mean())
    depths = depths.unsqueeze(0)
    depths = depths.unsqueeze(1)
    # depths = depths.repeat(1,3,1,1)
    depthsss = depths
    depthsss[depthsss > 30000] = 2
    means = depthsss[depthsss > 0.001].mean()
    depthsss = depthsss / means
    return depthsss
def processdepthss(depths):
    depths = torch.from_numpy(depths).cuda()
    # print(depths.max(),depths.min(),depths.mean())#tensor(23085) tensor(0) tensor(2138.1704)
    depths = depths.unsqueeze(0)
    depths = depths.unsqueeze(1)
    # depths = depths.repeat(1,3,1,1)
    depthsss = depths
    # mins = depthsss[depthsss > 0.001].min()
    
    depthsss[depthsss <1000] = 1000
    depthsss[depthsss >3000] =3000
    # means = depthsss[depthsss <100000].mean()
    # depthsss = depthsss / means
    means = depthsss.mean()
    depthsss = depthsss/means
    depthsss = 1.0/depthsss
    # print(depthsss.max(),depthsss.min(),depthsss.mean())
    # depthsss = means/depthsss
    return depthsss
@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    out_list, epe_list,epe_lists = [], [],[]
    iss = 0
    for val_id in range(len(val_dataset)):
        # isss = isss + 1
        # print(isss)
        image1, image2, flow_gt, valid_gt,depth1,depth2 = val_dataset[val_id]
        # print(valid_gt.shape,valid_gt.max())
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        # print('0',image1.shape,depth1.shape)
        image1 = image1[:,:,:368,:1216]
        depth1 = depth1[:368,:1216]
        image2 = image2[:,:,:368,:1216]
        depth2 = depth2[:368,:1216]
        valid_gt = valid_gt[:368,:1216].contiguous()
        flow_gt = flow_gt[:,:368,:1216]
        # print('kitti',flow_gt.shape,valid_gt.shape)
        # depth1 = depth1[None].cuda()
        # depth2 = depth2[None].cuda()
        # print('1',image1.shape,depth1.shape)
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        # print('2',image1.shape,depth1.shape)

        # depth1, depth2 = padder.pad(torch.FloatTensor(depth1), torch.FloatTensor(depth2))
        depth1 = cv2.resize(depth1,(image1.shape[3],image1.shape[2]))
        depth2 = cv2.resize(depth2,(image2.shape[3],image2.shape[2]))
        # print(image1.shape,depth1.shape)torch.Size([1, 3, 376, 1248]) (376,1248)
        depth1 = processdepths(depth1)
        depth2 = processdepths(depth2)
        # print(image1.shape,depth1.shape)
        # stdv = iss%35  # uniform 28
        # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
        # image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
        stdv1 = val_id % 10 # uniform
        if stdv1 == 0:
            image1 = image1*stdv1
            image2 = image2*stdv1
        else:
            image1 = image1 / stdv1
            image2 = image2 / stdv1
        # image1 = image1 / stdv1
        # image2 = image2 / stdv1
        iss = iss + 1
        flow_low, flow_pr = model(image1, image2,depth1,depth2, iters=iters, test_mode=True)
        # print('2',flow_pr.shape)

        flow = padder.unpad(flow_pr[0]).cpu()
        # print('3',flow.shape)
#####################################################
        # flow = padder.unpad(flow_pr[0]).cpu()
        # print('2',flow_pr[0].shape)
        flowss = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        # print('1',flowss.shape)
        flow_img = flow_viz.flow_to_image(flowss)
        # print('3',flow_img.shape)
        # flow_image = Image.fromarray(flow_img)
        # # frame_id: "{scene}_{img0_idx}", without suffix.
        # flow_image.save(f'kitti/{iss}.png')
#####################################################
        # print('1',flow.shape,flow_gt.shape)
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        number = str(iss)
        val = valid_gt.view(-1) >= 0.5
        epess = epe.view(-1).numpy()
        error = np.mean(epess[val])
        err = str(error)
        file_out = os.path.join('/home/zhou/fourier-2d/setting3',number + '-' + str(1) + '-' + err + '.png')
        valids = valid_gt>= 0.5
        vutils.save_image(epe * valids, file_out, normalize=True)
        # print('2',epe.shape,epe.max(),epe.min())
        # epeimage = flow_viz.flow_to_images(epe)
        # epeimage = Image.fromarray(epe)
        # frame_id: "{scene}_{img0_idx}", without suffix.
        number = str(iss)
        file_out = os.path.join('./epe',number+'.png')
        # print(file_out)
        # vutils.save_image(image1, file_out, normalize=True)
        # epe.save(f'epe/{iss}.png')
        epe_lists.append(epe.view(-1).numpy())
        mag = torch.sum(flow_gt**2, dim=0).sqrt()
        # print('01', epe.shape, mag.shape)
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        # print('12', epe.shape, val.shape)
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    # epe_all = np.concatenate(epe_lists)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    # acc1 = np.mean(epe_all < 1)
    # acc3 = np.mean(epe_all < 3)
    # acc5 = np.mean(epe_all < 5)
    acc1 = 100 *np.mean(epe_list < 1)
    acc3 = 100 *np.mean(epe_list < 3)
    acc5 = 100 *np.mean(epe_list < 5)
    # print(acc1,acc3,acc5)
    print("Validation KITTI: %f, %f,%f, %f,%f" % (epe, f1, acc1, acc3, acc5))
    return {'kitti-epe': epe, 'kitti-f1': f1,'kitti-acc1':acc1,'kitti-acc3':acc3,'kitti-acc5':acc5}
def processdepth(depths):
    iss = 9

    depths = torch.from_numpy(depths).cuda()
    # print(depths.max(),depths.min(),depths.mean())
    depths = depths.unsqueeze(0)
    depths = depths.unsqueeze(1)
    # depths = depths.repeat(1,3,1,1)
    depthsss = depths
    # depthsss[depthsss > 900] = 0.02
    # means = depthsss[depthsss > 0.02].mean()
    depthsss[depthsss > 900] = 100000
    means = depthsss[depthsss <100].mean()
    depthsss = depthsss / means
    return depthsss
def show_image(image,name):
    print(image.shape)
    image = image.cpu().numpy()
    image = (image).astype(np.uint8)
    # print(image.shape)
    filename = name
    cv2.imwrite(filename, image)

def validate_Tartanair(model,split = 'realtest4'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    # val_dataset = datasets.Tartanairtest(root='/home/zhou/file8T/raft/FlyingThings3D', dstype='frames_cleanpass',split)
    # val_dataset = datasets.Tartanairtest('/home/zhou/file8T/raft/FlyingThings3D', 'frames_cleanpass', split)
    val_dataset = datasets.Tartanairtest(root='/home/zhou/file8T/raft/FlyingThings3D', dstype=split)
    iters=24
    iss = 0
    print(split,len(val_dataset)-20)
    out_list, epe_list = [], []
    if not os.path.exists(split):
        os.makedirs(split, exist_ok=True)
    flow_folder = os.path.join(split, 'flow')
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder, exist_ok=True)
    visresult_folder = os.path.join(split, 'visresult')
    if not os.path.exists(visresult_folder):
        os.makedirs(visresult_folder, exist_ok=True)
    for val_id in range(len(val_dataset)):

        image1, image2,depth1,depth2 = val_dataset[val_id]#img1, img2, flow, valid.float(),depth1,depth2
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # print(depth1.shape)
        mask = depth1< 2500
        # print(mask.shape)
        # image1s = image1
        # image2s = image2
        # stdv1 = iss % 10 # uniform
        # if stdv1 == 0:
        # image1 = image1*0
        # image2 = image2*0
        # else:
        #     image1 = image1 / stdv1
        #     image2 = image2 / stdv1
        depth1 = cv2.resize(depth1,(image1.shape[3],image1.shape[2]))
        depth2 = cv2.resize(depth2,(image2.shape[3],image2.shape[2]))
        depth1 = processdepthss(depth1)
        depth2 = processdepthss(depth2)
        # print("depth",depth1.mean())
        iss = iss + 1
        flow_low, flow_pr = model(image1,image2,depth1,depth2, iters=iters, test_mode=True)
        flows = padder.unpad(flow_pr[0]).cpu()
        # print('3',flows.shape,image1.shape)
        # visresult_path = os.path.join(visresult_folder, f'result_{iss:04d}.jpg')
        # if iss > 20:
        #     project_pixels_with_optical_flow(image1,image2,flows,iss,visresult_folder)
        
        flowss = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        # print(flowss.mean())
        # mask = mask.cpu().numpy()
        mask = np.expand_dims(mask, axis=2)
        flowss = flowss*mask
        flow_img = flow_viz.flow_to_image(flowss)
        flow_image = Image.fromarray(flow_img)
        # flow_image.save(f'ours-2d-flow/{iss}.png')
        flow_path = os.path.join(flow_folder, f'{iss}.png')
        # print(flow_path)
        flow_image.save(flow_path)
    # calculate_average_diff_from_file(visresult_folder)
    return {0,0}
@torch.no_grad()
def validate_flything3D(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.FlyingThings3Dtest(root='/home/zhou/file8T/raft/FlyingThings3D', dstype='frames_cleanpass')
    iss = 0
    totaltime = 0.0
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _,depth1,depth2 = val_dataset[val_id]#img1, img2, flow, valid.float(),depth1,depth2
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        # depth1 = depth1[None].cuda()
        # depth2 = depth2[None].cuda()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # padder = InputPadder(depth1.shape)
        # depth1, depth2 = padder.pad(depth1, depth2)
        depth1 = cv2.resize(depth1,(image1.shape[3],image1.shape[2]))
        depth2 = cv2.resize(depth2,(image2.shape[3],image2.shape[2]))
        depth1 = processdepth(depth1)
        depth2 = processdepth(depth2)
        # stdv1 = val_id % 10 # uniform
        # stdv = val_id%35  # uniform
        # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
        # image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
        # stdv1 = iss % 10 # uniform
        # if stdv1 == 0:
        #     image1 = image1*stdv1
        #     image2 = image2*stdv1
        # else:
        #     image1 = image1 / stdv1
        #     image2 = image2 / stdv1
        # image1 = image1 * 0
        # image2 = image2 * 0
        # depth1 = depth1 * 0
        # depth2 = depth2 * 0
        iss = iss + 1
        start_time = time.time()
        flow_low, flow_pr = model(image1,image2,depth1,depth2, iters=iters, test_mode=True)

        # print('1',flow_pr.shape)
        flow = padder.unpad(flow_pr[0]).cpu()
        # print('2',flow_pr[0].shape)
        # flowss = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        # flow_img = flow_viz.flow_to_image(flowss)
        # flow_image = Image.fromarray(flow_img)
        # frame_id: "{scene}_{img0_idx}", without suffix.
        # flow_image.save(f'ours-2d-flow/{iss}.png')

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        end_time = time.time()
        run_time = end_time - start_time
        totaltime = totaltime + run_time
        if iss == 210:#208
            # print('epe2d',iss,image1.shape)
        #     number = str(iss)
        #     show_image(image1[0], 'setting3.png')
            # image1 = image1/9
            print('epe2d',iss,np.mean(epe.view(-1).numpy()),np.mean(epe.view(-1).numpy() < 1))
            file_out = os.path.join('/home/zhou/','setting3.png')
        # print(file_out)
            vutils.save_image(epe, file_out, normalize=True)
        # print(epe.shape,epe.unsqueeze(-1).shape)
            # mg2 = Image.fromarray(np.array(epe))
        # cv2.imwrite('./two.png',mg2)
            
            # vutils.save_image(epe, './{iss}.png', normalize=True)

            # RGBs = cv2.cvtColor(mg2, cv2.COLOR_GRAY2BGR)
            # vutils.save_image(RGBs, './two.png', normalize=True)
        # print('1',epe.view(-1).numpy())
        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # if epe.view(-1).numpy() < 100000:
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all[epe_all<10000])
    # epe1 = np.mean(epe_all[epe_all < 900])
    epe2 = np.mean(epe_all[epe_all < 100])
    # epe3 = np.mean(epe_all[epe_all < 800])
    # epe4 = np.mean(epe_all[epe_all < 700])
    # epe5 = np.mean(epe_all[epe_all < 600])
    # epe6 = np.mean(epe_all[epe_all < 500])
    # epe7 = np.mean(epe_all[epe_all < 400])
    # epe8 = np.mean(epe_all[epe_all < 300])
    # px = np.mean(epe_all>1000)
    # px0 = np.mean(epe_all<0.5)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)
    # print(epe1, epe2, epe3, epe4,epe5, epe6, epe7, epe8)
    print( epe, epe2,px1, px3, px5,totaltime/len(val_dataset))
    # print("Validation KITTI: %f, %f,%f, %f,%f" % (epe, f1, acc1,acc3,acc5))
    return {0,0}
@torch.no_grad()
def validate_flything3Done(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.FlyingThings3Dtest(root='/home/zhou/file8T/raft/FlyingThings3D', dstype='frames_finalpass')
    iss = 0
    run_time = 0
    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _,depth1,depth2 = val_dataset[val_id]#img1, img2, flow, valid.float(),depth1,depth2
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        # depth1 = depth1[None].cuda()
        # depth2 = depth2[None].cuda()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        # print(image1.shape)
        # padder = InputPadder(depth1.shape)
        # depth1, depth2 = padder.pad(depth1, depth2)
        depth1 = cv2.resize(depth1,(image1.shape[3],image1.shape[2]))
        depth2 = cv2.resize(depth2,(image2.shape[3],image2.shape[2]))
        depth1 = processdepth(depth1)
        depth2 = processdepth(depth2)

        # stdv1 = iss % 10 # uniform
        # if stdv1 == 0:
        #     image1 = image1*stdv1
        #     image2 = image2*stdv1
        # else:
        #     image1 = image1 / stdv1
        #     image2 = image2 / stdv1
        # stdv = val_id%35  # uniform
        # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
        # image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
        iss = iss + 1
        start_time = time.perf_counter()
        # flops = FlopCountAnalysis(model,(image1,image2,depth1,depth2))
        # print(flop_count_table(flops))
        # print(image1.shape)
        flow_low, flow_pr = model(image1,image2,depth1,depth2, iters=iters, test_mode=True)
        end_time = time.perf_counter()

        run_time = run_time + end_time - start_time
        # timelists.append(run_time)
        flow = padder.unpad(flow_pr[0]).cpu()

        # flowss = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        # flow_img = flow_viz.flow_to_image(flowss)
        # flow_image = Image.fromarray(flow_img)
        # # frame_id: "{scene}_{img0_idx}", without suffix.
        # flow_image.save(f'ours-2d-flow/{iss}.png')

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # print('1',epe.view(-1).numpy())
        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # if epe.view(-1).numpy() < 100000:
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    # time_all = np.concatenate(timelists)
    # timess = np.mean(time_all)
    epe = np.mean(epe_all[epe_all<10000])
    # epe1 = np.mean(epe_all[epe_all < 900])
    epe2 = np.mean(epe_all[epe_all < 100])
    # epe3 = np.mean(epe_all[epe_all < 800])
    # epe4 = np.mean(epe_all[epe_all < 700])
    # epe5 = np.mean(epe_all[epe_all < 600])
    # epe6 = np.mean(epe_all[epe_all < 500])
    # epe7 = np.mean(epe_all[epe_all < 400])
    # epe8 = np.mean(epe_all[epe_all < 300])

    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)
    # print(epe1, epe2, epe3, epe4,epe5, epe6, epe7, epe8)
    print( epe, epe2,px1, px3, px5,run_time/iss)
    # print("Validation KITTI: %f, %f,%f, %f,%f" % (epe, f1, acc1,acc3,acc5))
    return {0,0}
#
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--split', default='realtest4')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    print({sum([x.nelement() for x in model.parameters()])/1000000.} )
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e6}")
    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'flything3D':
            # validate_flything3D(model.module) 
            # validate_flything3Done(model.module)
            validate_Tartanair(model.module,args.split)
