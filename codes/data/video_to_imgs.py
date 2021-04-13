import cv2 as cv
import argparse
import os


parser = argparse.ArgumentParser(description="video_to_imgs")

parser.add_argument("--video_path", default="/home/zhijian/Desktop/workspace/videos/demo.mp4",
                    help="path of the input video")

parser.add_argument("--output_path", default="../../datasets/train_data/source_img/HR/",
                    help="path to save output images")

args = parser.parse_args()

video_path = args.video_path
output_path = args.output_path

print(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

cap = cv.VideoCapture(video_path)

isOpened = cap.isOpened()

fps = cap.get(cv.CAP_PROP_FPS)  ##获取帧率
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   ###获取宽度
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))   ###获取高度
print(fps,width,height)


count = 0
img_id = 0

while(isOpened and count<100000):

	flag, frame = cap.read()
	count += 1
	if(count%5 != 0):
		continue

	img_id+=1
	
	fileName = str(img_id)+".jpg"
	cv.imwrite(os.path.join(output_path, fileName),frame) 
# 	cv.imshow("video", frame)
	
	isOpened = cap.isOpened()

	key = cv.waitKey(25)
	if key == 27:  # 按键esc
		break

