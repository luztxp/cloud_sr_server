## 云端超分中转服务器

该项目的超分模型使用的是EfficientSR：https://github.com/zdyshine/RTC2020_EfficientSR

### 部署流程

##### 1. 配置Nginx-rtmp服务器
##### pull nginx-rtmp 镜像:
`sudo docker pull tiangolo/nginx-rtmp`
##### 启动 nginx-rtmp 镜像:
`sudo docker run -d -p 1935:1935 --name nginx-rtmp tiangolo/nginx-rtmp`

##### 2. 修改 ./codes/server.py中的rtmp地址，默认值说明：

* 主播端推流地址：rtmp://192.168.0.138:1935/live/001
* 超分后转发的RTMP地址：rtmp://192.168.0.138:1935/live/002
* 原始视频流转发的RTMP地址：rtmp://192.168.0.138:1935/live/003

##### 3. 运行 server 程序
```
    python server.py
```

##### 4.主播端进行推流，观众端进行拉流播放


### 超分模型训练

##### 模型结构
<div align="center"> <img src="https://i.loli.net/2021/04/10/YuFKQsLNf9Wm6zw.jpg" width = 700 height = 400 /> </div>


##### 1. 训练环境

 torch 1.5.0  
 scikit-image 0.16.2  
 opencv-python 4.5.1.48  

##### 2. 训练、测试数据生成

 * 原始图像存放位置：..datasets/train_datasource_imgHR  
 * 训练集存放位置：..datasets/train_data （其中hr_img为高清图像，lr_img为低清图像）
 * 训练集存放位置：..datasets/test_data （其中hr_img为高清图像，lr_img为低清图像）

 从视频文件提取生成数据集：

```
    修改video_to_imgs.py中的视频文件位置
    python .codes/data/video_to_imgs.py
```

 生成npy格式的训练数据  
 
```bash
    sh .codes/data/gen_data.sh
```

##### 3. 训练模型
 
等待数据全部生成后，开始训练  

```bash
python .codes/train.py
```

##### 4. 测试模型

```bash
python .codes/test.py
```



