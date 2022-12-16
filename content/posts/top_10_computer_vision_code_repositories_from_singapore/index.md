---
title: "Top 10 Computer Vision Code Repositories from Singapore"
date: 2020-12-16T08:00:00+08:00
tags: ["Github", "Deep Learning", "Singapore", "Machine Learning", "Source Code", "PyTorch", "TensorFlow", "Computer Vision"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "cv_repo_nuscenes.jpg"
    relative: true
    alt: "nuScenes dataset from Motional (formerly nuTonomy) featuring 1000 driving scenes in Boston and Singapore. Retrieved from Github."
    caption: "nuScenes dataset from Motional (formerly nuTonomy) featuring 1000 driving scenes in Boston and Singapore."
---

> **tl;dr** We feature 10 of the top Computer Vision (CV) code repositories from Singapore. These include popular 
> implementations of YOLO3, EfficientDet, DeepLab, FaceBoxes and other models ranging from activity recognition to eye tracking.  
> The ranking is decided based on the total Github stars of the repositories.

## 10. Dual Path Networks

{{< figure src="cv_repo_dpns.png" title="Architecture of DPNs. Retrieved from Github." >}}

This repository contains the code and trained models of Dual Path Networks which won the 1st place in Object Localization 
Task in ILSVRC 2017, and was a Top 3 team with on all competition tasks (Team: NUS-Qihoo_DPNs).

The paper, [Dual Path Networks](https://arxiv.org/abs/1707.01629), is published at NeurIPS 2017.

|          |                                                                    |
|----------|--------------------------------------------------------------------|
|Repository|[cypw/DPNs](https://github.com/cypw/DPNs)                           |
|License   |Not Specified                                                       |
|Author    |[Yunpeng](https://cypw.github.io/) ([cypw](https://github.com/cypw))|
|Vocation  |Senior research scientist at YITU Teach                            |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  534|  150|         16|


## 9. PointPillars for KITTI object detection

{{< figure src="cv_repo_pointpillars.png" title="Detecting objects from Point Clouds. Retrieved from Github." >}}

This repo demonstrates how to reproduce the results from [PointPillars: Fast Encoders for Object Detection from 
Point Clouds (CVPR 2019)](https://arxiv.org/abs/1812.05784).

PointPillars is a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical 
columns (pillars). While the encoded features can be used with any standard 2D convolutional detection architecture, 
we further propose a lean downstream network. Extensive experimentation shows that PointPillars outperforms previous 
encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection 
pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and 
bird's eye view KITTI benchmarks. 

|          |                                                                                    |
|----------|------------------------------------------------------------------------------------|
|Repository|[nutonomy/second.pytorch](https://github.com/nutonomy/second.pytorch)               |
|License   |[MIT License](https://api.github.com/licenses/mit)                                  |
|Author    |[Motional](motional.com) ([nutonomy](https://github.com/nutonomy))                  |
|Vocation  |We're making self-driving vehicles a safe, reliable, and accessible reality. at None|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  544|  143|          1|


## 8. 👀 Eye Tracking Library

{{< figure src="cv_repo_eye_tracking.jpg" title="Eye/gaze tracking in real-time." >}}

A Python (2 and 3) library that provides a webcam-based eye tracking system. 
It gives you the exact position of the pupils and the gaze direction, in real time.

|          |                                                                                                      |
|----------|------------------------------------------------------------------------------------------------------|
|Repository|[antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking)                               |
|License   |[MIT License](https://api.github.com/licenses/mit)                                                    |
|Author    |[Antoine Lamé](http://antoinelame.fr) ([antoinelame](https://github.com/antoinelame))                 |
|Vocation  |Software Engineer. Laravel, Python, Go. Currently building high frequency trading algorithms.|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  574|  183|         32|


## 7. PyTorch Video Activity Recognition.

{{< figure src="cv_repo_video.jpg" title="Detecting the 'applying lipstick' activity in a video." >}}

PyTorch implemented C3D, R3D, R2Plus1D models for video activity recognition. The models are trained on the on UCF101 
and HMDB51 datasets. 

|          |                                                                                             |
|----------|---------------------------------------------------------------------------------------------|
|Repository|[jfzhang95/pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)|
|License   |[MIT License](https://api.github.com/licenses/mit)                                           |
|Author    |[Pyjcsx](http://jeff95.me) ([jfzhang95](https://github.com/jfzhang95))                       |
|Vocation  |PhD student at NUS at National University of Singapore                                       |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  640|  170|         32|


## 6. PyTorch Implementation of FaceBoxes

{{< figure src="cv_repo_faceboxes.jpg" title="The architecture of FaceBoxes. Retrieved from Github." >}}

A PyTorch implementation of the [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234).
The original was implemented in [caffe](https://github.com/sfzhang15/FaceBoxes). FaceBoxes is a novel CPU-based face detector, 
with superior performance on both speed and accuracy. The speed of FaceBoxes is invariant to the number of faces. 

|          |                                                                         |
|----------|-------------------------------------------------------------------------|
|Repository|[zisianw/FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)|
|License   |[MIT License](https://api.github.com/licenses/mit)                       |
|Author    |Zi Sian Wong ([zisianw](https://github.com/zisianw))             |
|Vocation  |Computer Vision & Deep Learning                                |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  659|  173|         10|


## 5. :angel: Morph faces with Python, Numpy, Scipy

{{< figure src="cv_repo_face_morpher.png" title="Morphing a face automatically from a source to destination image. Retrieved from Github." >}}

Warp, average and morph human faces!
1. Locates face points
2. Align faces by resizing, centering and cropping to given size
3. Given 2 images and its face points, warp one image to the other
    - Triangulates face points
    - Affine transforms each triangle with bilinear interpolation
4a. Morph between 2 or more images
4b. Average faces from 2 or more images
5. Optional blending of warped image:
    - Weighted average
    - Alpha feathering
    - Poisson blend

|          |                                                                                |
|----------|--------------------------------------------------------------------------------|
|Repository|[alyssaq/face_morpher](https://github.com/alyssaq/face_morpher)                 |
|License   |Not Specified                                                                   |
|Author    |[Alyssa Quek](https://alyssaq.github.io) ([alyssaq](https://github.com/alyssaq))|
|Vocation  |Software Engineer at Apple                                                      |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  705|  196|         32|


## 4. nuScenes Dataset Devkit

{{< figure src="cv_repo_nuscenes2.png" title="" >}}

The nuScenes dataset is a public large-scale dataset for autonomous driving developed by the team at Motional 
(formerly nuTonomy). The dataset is meant to support public research into computer vision and autonomous driving.
The dataset contains 1000 driving scenes in Boston and Singapore, two cities that are known for their dense traffic and 
highly challenging driving situations. The scenes of 20 second length are manually selected to show a diverse and 
interesting set of driving maneuvers, traffic situations and unexpected behaviors. To facilitate common computer vision 
tasks, such as object detection and tracking, the dataset contains annotations for 23 object classes with accurate 
3D bounding boxes at 2Hz over the entire dataset.

This repository contains the devkit of the nuImages and nuScenes dataset.

|          |                                                                                    |
|----------|------------------------------------------------------------------------------------|
|Repository|[nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)             |
|License   |[Other](None)                                                                       |
|Author    |[Motional](motional.com) ([nutonomy](https://github.com/nutonomy))                  |
|Vocation  |[Motional](motional.com) (formerly nutonomy), making self-driving vehicles|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  777|  287|          2|


## 3. PyTorch Implementation of EfficientDet

{{< figure src="cv_repo_efficientdet_archi.png" title="Architecture of EfficientDet including a weighted bi-directional feature pyramid network (BiFPN)." >}}

A PyTorch implementation of the 2019 __Computer Vision__ paper [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) from Google Brain.
The [official implementation](https://github.com/google/automl/tree/master/efficientdet) by Google Brain is in TensorFlow.

|   |   |
|-	|-	|
|Repository  	    |[toandaominh1997/EfficientDet.Pytorch](https://github.com/toandaominh1997/EfficientDet.Pytorch)  	|
|License            |[MIT](https://github.com/toandaominh1997/EfficientDet.Pytorch/blob/master/LICENSE)   |
|Author  	        |[Đào Minh Toàn](https://twitter.com/toandaominh1997) ([toandaominh1997](https://github.com/toandaominh1997))  	|
|Vocation  	        |[Data Scientist](https://www.linkedin.com/in/toandaominh1997) at [VinID](https://medium.com/vinid)  	|

|Language   |Stars      |Forks      |Watchers   |Open Issues    |
|-	        |-	        |-	        |-          |-              |
|Python     |1,337      |300        |44         |112            |


## 2. Object Detection: YOLO3 

{{< figure src="cv_repo_yolo3.png" title="Object bounding boxes on video footage as predicted by YOLO3. Retrieved from the official YOLO site." >}}

A __Computer Vision__ repository with code for training and evaluation of a YOLO3 model for the __Object Detection__ task. 
[YOLO](https://pjreddie.com/darknet/yolo/), You Only Look Once, is a state-of-the-art, real-time object detection model.
Its claim to fame is its extremely fast and accurate and you can trade-off speed and accuracy without re-training 
by changing the model size. Multi-GPU training is also implemented.

|   |   |
|-	|-	|
|Repository  	    |[experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3)  	|
|License            |[MIT](https://github.com/experiencor/keras-yolo3/blob/master/LICENSE)   |
|Author  	        |[Huynh Ngoc Anh](https://experiencor.github.io/) ([experiencor](https://github.com/experiencor))  	|
|Vocation  	        |[Machine Learning Engineer](https://sg.linkedin.com/in/ngoca) at [Grab](https://engineering.grab.com/)  	|

|Language   |Stars      |Forks      |Watchers   |Open Issues    |
|-	        |-	        |-	        |-          |-              |
|Python     |1,362      |753        |54         |217            |


## 1. DeepLab v3+ model in PyTorch

{{< figure src="cv_repo_pytorch_deeplab.png" title="Some results of the deep labelling model on various datasets. Retrieved from Github." >}}

A computer vision repository which started with an early PyTorch implementation (circa 2018) of DeepLab-V3-Plus (in PyTorch 0.4.1). 
DeepLab is a series of __image semantic segmentation__ models whose latest version, v3+, is state-of-art on the semantic segmentation task. 
It can use Modified Aligned Xception and ResNet as backbone. 
The authors train DeepLab V3 Plus using Pascal VOC 2012, SBD and Cityscapes datasets. Pre-trained models on ResNet, 
MobileNet and DRN are provided.

|   |   |
|-	|-	|
|Repository  	    |[jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)  	|
|License            |[MIT](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/LICENSE)   |
|Author  	        |[Jianfeng Zhang](http://jeff95.me/) ([jfzhang95](https://github.com/jfzhang95))  	|
|Vocation  	        |PhD Student at NUS  	|

|Language   |Stars      |Forks      |Watchers   |Open Issues    |
|-	        |-	        |-	        |-          |-              |
|Python     |2,057      |634        |47         |95             |

## More CV repositories

Visit [machinelearning.sg](https://machinelearning.sg/repo/) to view a full list of ML repositories from Singapore.
