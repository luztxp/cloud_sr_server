import argparse
import torch
import os
import numpy as np
from utils import utils
import skimage.color as sc
import skimage.io as sio
from model import model
import cv2
import time 


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cuda = True
is_y = False
upscale_factor = 2
checkpoint = '../pretrained_model/180p.pth'

device = torch.device('cuda' if cuda else 'cpu')

model = model.model_rtc(upscale=upscale_factor)
model_dict = utils.load_state_dict(checkpoint)
model.load_state_dict(model_dict, strict=True)

if cuda:
    model = model.to(device)


def process(imname, model):

    im_l = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # im_l = sio.imread(opt.test_lr_folder + '/' + imname.split('/')[-1])  # RGB
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        print("use cuda!!!")
        im_input = im_input.to(device)

    with torch.no_grad():
        start = time.time()*1000
        out = model(im_input)
        torch.cuda.synchronize()
        time_cost = time.time()*1000 - start
        print(f"time cost = {time_cost} ms")

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)

#     cv2.imshow("input", im_l[:, :, [2, 1, 0]])
#     cv2.imshow("output", out_img[:, :, [2, 1, 0]])
#     cv2.waitKey(0)


img_list = [i*10 for i in range(1, 10)]

for i in range(len(img_list)):
    imname = f"../datasets/test_data/lr_img/x2/{img_list[i]}.jpg"
    process(imname, model)