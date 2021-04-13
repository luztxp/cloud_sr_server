# Downscale HR images with factor x2 and skip 1112.JPG.

python video_to_imgs.py --video_path /home/zhijian/Desktop/workspace/videos/demo_360p.mp4 --output_path ../../datasets/train_data/source_img/HR/

python img_process.py  --input_path ../../datasets/train_data/source_img/HR/ 

python png2npy.py --pathFrom ../../datasets/train_data/hr_img/ --pathTo ../../datasets/train_data/hr_npy/

python png2npy.py --pathFrom ../../datasets/train_data/lr_img/ --pathTo ../../datasets/train_data/lr_npy/


python png2npy.py --pathFrom ../../datasets/test_data/hr_img/x2/ --pathTo ../../datasets/test_data/hr_npy/x2/

python png2npy.py --pathFrom ../../datasets/test_data/lr_img/x2/ --pathTo ../../datasets/test_data/lr_npy/x2/