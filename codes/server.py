import queue
import threading
from subprocess import Popen
import subprocess
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


### init the SR model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cuda = True
is_y = False
upscale_factor = 2
checkpoint = '../pretrained_model/epoch_91.pth'

device = torch.device('cuda' if cuda else 'cpu')

model = model.model_rtc(upscale=upscale_factor)
model_dict = utils.load_state_dict(checkpoint)
model.load_state_dict(model_dict, strict=True)

if cuda:
    model = model.to(device)
    
    
def process(image):

    im_l = image[:, :, [2, 1, 0]]  # BGR to RGB
    # im_l = sio.imread(opt.test_lr_folder + '/' + imname.split('/')[-1])  # RGB
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
#         print("use cuda!!!")
        im_input = im_input.to(device)

    with torch.no_grad():
#         start = time.time()*1000
        out = model(im_input)
        torch.cuda.synchronize()
#         time_cost = time.time()*1000 - start
#         print(f"time cost = {time_cost} ms")

    out_img = utils.tensor2np(out.detach()[0])
#     print(f"time cost = {time.time()*1000 - start} ms")
    return out_img[:, :, [2, 1, 0]]
#     crop_size = upscale_factor
#     cropped_sr_img = utils.shave(out_img, crop_size)
    
    

class Live(object):
    def __init__(self, rtmpUrl_source, rtmpUrl_dest_sr, rtmpUrl_dest_resize):
        self.frame_queue_sr = queue.Queue()
        self.frame_queue_resize = queue.Queue()
        self.command_sr = ""
        self.rtmpUrl_source = rtmpUrl_source
        self.rtmpUrl_dest_sr = rtmpUrl_dest_sr
        self.rtmpUrl_dest_resize = rtmpUrl_dest_resize
        
        self.img_id = 0
        
    def read_frame(self):  
        cap = cv2.VideoCapture(self.rtmpUrl_source)

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         print(f"!!! input frame size = {width},{height}",)
        # ffmpeg command
        self.command_sr = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width*2,height*2),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
#                 '-g', '1',
                self.rtmpUrl_dest_sr]
        
        self.command_resize = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width*2,height*2),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
#                 '-g', '1',
                self.rtmpUrl_dest_resize]
             
        # read webcamera
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            self.img_id += 1
            #print(f"img id: {self.img_id}")

            resize_frame = cv2.resize(frame, (width*2,height*2))

#             if(self.img_id%1==0):
#                 out_img = process(frame)
#             else:
#                 out_img = buf
            sr_frame = process(frame)
                     
            self.frame_queue_sr.put(sr_frame)
            print(f"queue.size() = {self.frame_queue_sr.size()}")
#             self.frame_queue_resize.put(resize_frame)

    def push_frame(self):
        while(True):
            if(len(self.command_sr)>0):
                p = Popen(self.command_sr, stdin=subprocess.PIPE)
                break

        while True:
#             if self.frame_queue.empty() != True:

            try:
                frame = self.frame_queue_sr.get(block=False)
            except queue.Empty:
                time.sleep(0.005)
                continue
                
            p.stdin.write(frame.tostring())
            
    def trans(self):
        cap = cv2.VideoCapture(self.rtmpUrl_source)

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         print(f"!!! input frame size = {width},{height}",)
        # ffmpeg command
        self.command_sr = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width*2,height*2),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
#                 '-g', '1',
                self.rtmpUrl_dest_sr]
        
        self.command_resize = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width*2,height*2),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
#                 '-g', '1',
                self.rtmpUrl_dest_resize]
        
        p_sr = Popen(self.command_sr, stdin=subprocess.PIPE)
        p_resize = Popen(self.command_resize, stdin=subprocess.PIPE)
             
        # read webcamera
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            self.img_id += 1
            #print(f"img id: {self.img_id}")

            resize_frame = cv2.resize(frame, (width*2,height*2))

#             if(self.img_id%1==0):
#                 out_img = process(frame)
#             else:
#                 out_img = buf
            sr_frame = process(frame)
                     
            p_sr.stdin.write(sr_frame.tostring())
            p_resize.stdin.write(resize_frame.tostring())
        
                
    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,))
#             threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]

        threads[0].join()
#         threads[0].detach()


LiveObj = Live('rtmp://192.168.0.138:1935/live/001', 'rtmp://192.168.0.138:1935/live/002','rtmp://192.168.0.138:1935/live/003')
# LiveObj.run()
LiveObj.trans()