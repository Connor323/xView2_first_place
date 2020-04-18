import os
from os import path as osp 
import cv2
import glob
import numpy as np 

root = "submission"
res_root = "show"
files = glob.glob(osp.join(root, "*.png"))

for f in files:
    image = cv2.imread(f, 0)
    if "damage" in f:
        h, w = image.shape[:2]
        show = np.zeros([h, w, 3], float)
        show[image == 1] += [127, 255, 0]
        show[image == 2] += [255, 127, 0]
        show[image == 3] += [127, 0, 255]
        show[image == 4] += [255, 0, 127]
        show /= 4
        minv = show.min()
        maxv = show.max()
        if maxv != 0 or minv != 0:
            show = (show - minv) / (maxv - minv + 1e-5) * 255
        image = cv2.cvtColor(show.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        image[image != 0] = 255
    cv2.imwrite(osp.join(res_root, osp.basename(f)), image)