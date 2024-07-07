import open3d as o3d
import cv2
import numpy as np
from run import DepthDetect
import matplotlib.pyplot as plt

def RGBD2Point(depth_img, rgb_img):
    # 相机内参
    k_x = 5.775910000000000082e+02
    k_y = 5.787300000000000182e+02
    u_0 = 3.189049999999999727e+02
    v_0 = 2.426839999999999975e+02
    factor = 1
    K = [[k_x, 0, u_0], [0, k_y, v_0], [0, 0, 1]]

    # 逐点处理，此过程可以使用numpy优化
    m, n = depth_img.shape  # 480 640
    color_map, point_cloud = [], []
    for v in range(m):      # 行相当于y坐标
        for u in range(n):  # 列相当于x坐标
            if depth_img[v, u] == 0:
                continue
            rgb = rgb_img[v, u]
            rgb = [rgb[0], rgb[1], rgb[2]]
            rgb_info = np.array(rgb) / 255.0  # 颜色归一化到0-1之间
            rgb_info = rgb_info[::-1]  # cv2读取数据格式为BGR
            color_map.append(rgb_info)
            depth = depth_img[v, u]
            # 矩阵运算速度较慢
            # x_c, y_c, z_c = np.transpose(np.dot(np.linalg.inv(K), np.transpose(depth * np.array([u, v, 1]))))
            z_c = depth / factor
            x_c = (u - u_0) * z_c / k_x
            y_c = (v - v_0) * z_c / k_y
            point_cloud.append([x_c, y_c, z_c])
    point_cloud = np.array(point_cloud)
    color_map = np.array(color_map)
    return point_cloud, color_map  # shape都是(212342, 3) point为(x,y,z) color为(r,g,b)


depth_path = "test_imgs/0000_depth.npy"
color_path = "test_imgs/0000_color.png"

s_depth = np.load(depth_path)  # 480 640

s_color = cv2.imread(color_path)     # (480, 640, 3)
detector = DepthDetect()
s_depth = detector.run(color_path,depth_path)

points, color = RGBD2Point(s_depth, s_color)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(points)
pc.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pc])      # 可视化
o3d.io.write_point_cloud(color_path.replace("color.png","pointcloud_completion.ply"), pc)  # 保存文件

