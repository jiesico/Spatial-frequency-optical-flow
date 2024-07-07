import cv2
import numpy as np
import torch

image = cv2.imread('/home/zhou/file8T/raft/FlyingThings3D/0012.png')
# rint(image.max())
# image = fromnumpy()
# image = (image).astype(np.uint8)
image = image*0
# stdv = 9  # uniform
# image = (image + stdv * torch.randn(*image.shape)).clamp(0.0, 255.0)
cv2.imwrite('set3-8.png',image)
