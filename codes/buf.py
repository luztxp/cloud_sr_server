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
import queue
import threading


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

frame_queue = queue.Queue()

def sr_process(model):

    # im_l = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB

    while(True):
        image = frame_queue.get()

        start = time.time()*1000
        im_l = image[:, :, [2, 1, 0]]


        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        if cuda:
            im_input = im_input.to(device)

        print(f"time cost 01 = {time.time()*1000-start} ms")
        # with torch.no_grad():
            
        out = model(im_input)
        torch.cuda.synchronize()
        print(f"time cost 02 = {time.time()*1000-start} ms")
        
        out_img = utils.tensor2np(out.detach()[0])
        print(f"time cost 03 = {time.time()*1000-start} ms")
#         cv2.imshow("output", image)
        # cv2.imshow("output", out_img[:, :, [2, 1, 0]])

#         key = cv2.waitKey(25)
#         if key == 27:  # 按键esc
#             break


        # return out_img

    # cv2.imshow("input", im_l[:, :, [2, 1, 0]])
        
    # cv2.waitKey(0)


# img_list = [10]

# for i in range(len(img_list)):
#     imname = f"../datasets/test_data/lr_img/x2/{img_list[i]}.jpg"
#     process(imname, model)


def read_frame():

    cap = cv2.VideoCapture("./videos/demo_180p.mp4")

    count = 0
    isOpened = cap.isOpened()

    while(isOpened and count<100000):
        flag, frame = cap.read()
        count += 1

        frame_queue.put(frame)

        # frame = process(frame, model)
        # cv2.imshow("video", frame[:, :, [2, 1, 0]])
        
        isOpened = cap.isOpened()

#         key = cv2.waitKey(2)
#         if key == 27:  # 按键esc
#             break


threads = [
    threading.Thread(target=read_frame, args=()),
    threading.Thread(target=sr_process, args=(model,))
]
[thread.setDaemon(True) for thread in threads]
[thread.start() for thread in threads]

threads[0].join()

