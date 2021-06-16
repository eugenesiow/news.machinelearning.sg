---
title: "Object Detection with YOLOv5: Detecting People in Images"
date: 2021-05-30T16:00:00+08:00
tags: ["Computer Vision", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Object Detection", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "object_detection.jpg"
    alt: "Object Detection with YOLOv5: Detecting People in Images"
---

> **tl;dr** A step-by-step tutorial to detect people in photos automatically using the ultra-fast You-Only-Look-Once (YOLOv5) model.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Detect_Persons_From_Image_YOLOv5.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

# Detect Persons From An Image with YOLOv5 Object Detection

Notebook to detect persons from a image and to export clippings of the persons and an image with bounding boxes drawn. It can detect classes other than persons as well.

[Object detection](https://paperswithcode.com/task/object-detection) is the task of detecting instances of objects of a certain class within an image. The state-of-the-art methods can be categorized into two main types: one-stage methods and two stage-methods. Pre-trained YOLOv5 models are used in this one-stage method that prioritizes inference speed.

The [model used](https://pytorch.org/hub/ultralytics_yolov5/) is one of the pre-trained `ultralytics/yolov5` models. It was trained on the COCO train2017 for object detection set, it has [80 classes](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-datasetyaml).

The notebook is structured as follows:
* Setting up the Environment
* Getting Data
* Using the Model (Running Inference)

# Setting up the Environment

#### Dependencies and Runtime

If you're running this notebook in Google Colab, all the dependencies are already installed and we don't need the GPU for this particular example. 

If you decide to run this on many (>thousands) images and want the inference to go faster though, you can select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`.

# Getting Data

#### Getting Foreground Images

* Foreground image #1: A photo of a [baseball team](https://unsplash.com/photos/GudTmd-Q3Cg) lined up by Wade Austin Ellis. From Unsplash.
* Foreground image #2: An anime fanart image of a [gundam pilot](https://danbooru.donmai.us/posts/4549761) by basedheero. From Danbooru.

We'll save these images to our local storage and view a preview of them in our notebook.


```
import cv2
from urllib.request import urlretrieve
from google.colab.patches import cv2_imshow

# save the foreground and background to our local storage
urlretrieve('https://images.unsplash.com/photo-1526497127495-3b388dc87620?auto=format&fit=crop&w=640&q=80', '/content/foreground1.jpg')
urlretrieve('https://danbooru.donmai.us/data/original/c0/4e/__tatsumi_hori_gundam_and_2_more__c04e8425ff3685202a67386027ea555d.png', '/content/foreground2.jpg')

# display the images in the notebook
cv2_imshow(cv2.imread('/content/foreground1.jpg'))
cv2_imshow(cv2.imread('/content/foreground2.jpg'))
```


![png](Detect_Persons_From_Image_YOLOv5_files/Detect_Persons_From_Image_YOLOv5_6_0.png)



![png](Detect_Persons_From_Image_YOLOv5_files/Detect_Persons_From_Image_YOLOv5_6_1.png)


# Using the Model (Running Inference)

First, we need to install the required dependencies for YOLOv5 by running the code below.

We will also print out the torch version at the end and if we are using CPU or GPU. Both are fine for the inference because YOLOv5 is very fast. If we are running this on thousands of images, we might want to use a GPU. See the `Setting up the Environment` section above for more details.


```
!pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

import torch
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```


Now we want to load and run the model on our 2 foregrounds. We only need to use the [yolov5s model checkpoints](https://github.com/ultralytics/yolov5/releases/tag/v5.0), which is very [fast](https://github.com/ultralytics/yolov5#pretrained-checkpoints) (just 2ms on a V100) and small (just 14.1mb). If you want higher object detection accuracy, try the larger models, they come in s, m, l and x sizes. 

![YOLOv5 Model Comparison](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png)

Specifically we are running the following steps:

* `torch.hub.load()` - Loads the pre-trained model from torchhub. In particular, we specify to use the small model, `yolov5s`.
* create a list called `imgs` with the 2 file paths of the 2 foregrounds in the list.
* `model()` - We run the inference using the model we loaded on the `imgs` list.
* `results.print()` - We print out a summary of the inference run and results from the `results` object which is returned from the previous step.


```
import torch
import torch
from IPython.display import clear_output

# Load the model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Setup the paths of the 2 foregrounds in a list to run the model on 
dir = '/content/'
imgs = [dir + f for f in ('foreground1.jpg', 'foreground2.jpg')]

# Inference, run the model on the 2 foreground images
results = model(imgs, size=640)
clear_output()

# Print out a summary of the inference job, we ran it on 2 images, at what speed and what detections were made
results.print()
```


We can also convert each of the results returned into a pandas DataFrame. We run the `results.pandas().xyxy[0]` to return the pandas DataFrame for the first foreground result (foreground #1). Note that foreground #2 is stored in the second element of the list `.xyxy[1]`.

We see that the pandas DataFrame consists of rows of the 11 persons and 1 baseball glove detected in foreground #1. Each row contains the bounding box (xmin, ymin, xmax and ymax), the confidence of the detection and the class of the detection (0 is person and 35 is baseball glove).


```
results.pandas().xyxy[0]
```


Now let's clip out the first row result of foreground #1 and display the detected person.


```
# get the tensor for foreground #1 detection results 
# and convert to an integer numpy array, return the positions x0, y0, x1, y1
x0, y0, x1, y1, _, _ = results.xyxy[0][0].numpy().astype(int) 

# crop/clip the image and show it
cropped_image = results.imgs[0][y0:y1, x0:x1]
cv2_imshow(cropped_image)
```


![png](Detect_Persons_From_Image_YOLOv5_files/Detect_Persons_From_Image_YOLOv5_15_0.png)


If we wanted a view of the bounding boxes, classes and original image all rendered, we can just call `results.render()` and then show the resulting image from the `.imgs` list in results.

We show an example of calling `render()` and then displaying foreground #2 with the rendered bounding boxes and labels.


```
results.render()
cv2_imshow(results.imgs[1])
```


![png](Detect_Persons_From_Image_YOLOv5_files/Detect_Persons_From_Image_YOLOv5_17_0.png)


We can save the rendered image to disk using the `imwrite()` function, with the output filepath and the result image array as parameters.


```
cv2.imwrite('/content/foreground2_results.jpg', results.imgs[1])
```

We can connect to Google Drive with the following code. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the output files which are saved in the `/content/` directory to the root of your Google Drive.


```
import shutil
shutil.move('/content/foreground2_results.jpg', '/content/drive/My Drive/foreground2_results.jpg')
```

We can also play around with some settings when we run the model inference. For example, we can set the confidence threshold very high (>0.8) to ensure that only detections which we are 90% or more confident of will be shown.

To detect only persons, we can specify only the `0` class in `model.classes`.


```
model.conf = 0.8  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

results = model(imgs, size=640)
results.pandas().xyxy[0]
```

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
