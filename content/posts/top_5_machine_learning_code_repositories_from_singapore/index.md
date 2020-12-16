---
title: "Top 5 Machine Learning Code Repositories from Singapore"
date: 2020-12-10T08:00:00+08:00
tags: ["Github", "Deep Learning", "Singapore", "Machine Learning", "Source Code", "PyTorch", "TensorFlow", "Computer Vision", "Natural Language Processing", "Finance"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "repo_efficientdet.gif"
    alt: "Object bounding boxes on video footage as predicted by EfficientDet, a family of scalable and efficient object detectors."
    caption: "Object bounding boxes on video footage as predicted by EfficientDet, a family of scalable and efficient object detectors."
---

> **tl;dr** We feature 5 of the top machine learning code repositories on Github from Singapore. The Top 5 is made up of 
> popular implementations of state-of-the-art Computer Vision (CV) and Natural Language Processing (NLP) models and 
> even a high-frequency trading project. The ranking is decided based on the total stars (stargazer count) of the 
> repositories.

## 5. PyTorch Implementation of EfficientDet

{{< figure src="repo_efficientdet_archi.png" title="Architecture of EfficientDet including a weighted bi-directional feature pyramid network (BiFPN)." >}}

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

## 4. Object Detection: YOLO3 

{{< figure src="repo_yolo3.png" title="Object bounding boxes on video footage as predicted by YOLO3. Retrieved from the official YOLO site." >}}

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

## 3. Chinese Named Entity Recognition and Relation Extraction

{{< figure src="repo_information_extraction_chinese.jfif" title="Visualization of 3x3, 7x7 and 15x15 receptive fields produced by 1, 2 and 4 dilated convolutions by the IDCNN model." >}}

An __NLP__ repository including state-of-art deep learning methods for various tasks in chinese/mandarin language (中文): 
named entity recognition (__NER__/实体识别), relation extraction (__RE__/关系提取) and word segmentation.

|   |   |
|-	|-	|
|Repository  	    |[crownpku/Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese)  	|
|License            |Not Specified   |
|Author  	        |[Wang Guan](http://www.crownpku.com/) ([crownpku](https://github.com/crownpku))  	|
|Vocation  	        |[Senior Data Scientist, VP](https://www.linkedin.com/in/crownpku/) at [Swiss Re](https://www.linkedin.com/company/swiss-re/)  	|

|Language   |Stars      |Forks      |Watchers   |Open Issues    |
|-	        |-	        |-	        |-          |-              |
|Python     |1,692      |748        |90         |102            |

## 2. High-frequency Trading Model using the Interactive Brokers API

{{< figure src="repo_high_frequency_trading.gif" title="Demo of setting up the model using docker-compose. Retrieved from Github." >}}

A high-frequency trading model using Interactive Brokers API with pairs and mean-reversion in Python. 
It was last updated with v3.0 in June 2019.
The author describes the model as utilizing statistical arbitrage incorporating these methodologies:

- Bootstrapping the model with historical data to derive usable strategy parameters
- __Resampling__ inhomogeneous time series to homogeneous time series
- Selection of __highly-correlated tradable pair__
- The ability to short one instrument and long the other.
- Using __volatility ratio__ to detect up or down trend.
- Fair valuation of security using __beta__, or the mean over some past interval.
- One pandas DataFrame to store historical prices

|   |   |
|-	|-	|
|Repository  	    |[jamesmawm/High-Frequency-Trading-Model-with-IB](https://github.com/jamesmawm/High-Frequency-Trading-Model-with-IB)  	|
|License            |[MIT](https://github.com/jamesmawm/High-Frequency-Trading-Model-with-IB/blob/master/LICENSE)   |
|Author  	        |[James Ma](https://linkedin.com/in/jamesmawm) ([jamesmawm](https://github.com/jamesmawm))  	|
|Vocation  	        |Full-stack software engineer and author of `Mastering Python for Finance`.  	|

|Language   |Stars      |Forks      |Watchers   |Open Issues    |
|-	        |-	        |-	        |-          |-              |
|Python     |1,734      |525        |231        |7              |

## 1. DeepLab v3+ model in PyTorch

{{< figure src="repo_pytorch_deeplab.png" title="Some results of the deep labelling model on various datasets. Retrieved from Github." >}}

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

## More ML repositories

Visit [machinelearning.sg](https://machinelearning.sg/repo/) to view a full list of ML repositories from Singapore.
