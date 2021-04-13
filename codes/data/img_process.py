import os
import cv2
import argparse

# Training settings
parser = argparse.ArgumentParser(description="agora_sr_2020")
parser.add_argument("--input_path", default="../../datasets/train_data/source_img/HR/",
                    help="path to input_path")
parser.add_argument("--target_path_train_hr", default="../../datasets/train_data/hr_img",
                    help="path to save trainset hr images")

parser.add_argument("--target_path_test_hr", default="../../datasets/test_data/hr_img",
                    help="path to save testset hr images")

parser.add_argument("--target_path_train_lr", default="../../datasets/train_data/lr_img/x2",
                    help="path to save trainset lr images")

parser.add_argument("--target_path_test_lr", default="../../datasets/test_data/lr_img/x2",
                    help="path to save testset lr images")

args = parser.parse_args()

input_path = args.input_path
target_path_train_hr = args.target_path_train_hr
target_path_test_hr = args.target_path_test_hr
target_path_train_lr = args.target_path_train_lr
target_path_test_lr = args.target_path_test_lr

new_paths = [target_path_train_hr, target_path_test_hr, target_path_train_lr, target_path_test_lr]

for path in new_paths:
    if not os.path.exists(path):
        os.makedirs(path)


img_sum = len(os.listdir(input_path))

# trainSetScale = 0.95


test_list = [str(i)+".jpg" for i in range(0, img_sum, 10)]



for root, dirs, files in os.walk(input_path):
    for file in sorted(files):
        if file in test_list:
            target_path_hr = target_path_test_hr
            target_path_lr = target_path_test_lr
        else:
            target_path_hr = target_path_train_hr
            target_path_lr = target_path_train_lr

        print(os.path.join(root, file))
        bgr = cv2.imread(os.path.join(root, file))
        print('Processing img: ', file, bgr.shape)

        # gen downscale lr images
        bgr = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(target_path_lr, file), bgr)
        # copy hr images
        os.system('cp %s %s' %(os.path.join(root, file), target_path_hr))