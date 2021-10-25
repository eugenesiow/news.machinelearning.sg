---
title: "OCR (Optical Character Recognition) from Images with Transformers"
date: 2021-10-21T10:00:00+08:00
tags: ["Computer Vision", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Optical Character Recognition", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "ocr_from_images.jpg"
    alt: "Optical Character Recognition from Images with TrOCR"
---

> **tl;dr** A step-by-step tutorial to automatically recognise text (OCR) from images of handwritten and printed text using transformer encoder-decoder models. 

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/OCR_from_Images_with_Transformers.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

## OCR (Optical Character Recognition) from Images with Transformers

Notebook to recognise text automaticaly from an input image with either handwritten or printed text.

[Optical Character Recognition](https://paperswithcode.com/task/optical-character-recognition) is the task of converting images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo (for example the text on signs and billboards in a landscape photo, license plates in cars...) or from subtitle text superimposed on an image (for example: from a television broadcast).

The [transformer models used](https://malaya-speech.readthedocs.io/en/latest/tts-singlish.html) are from Microsoft's TrOCR. The TrOCR models are encoder-decoder models, consisting of an image Transformer as encoder, and a text Transformer as decoder. We utilise the versions hosted on [huggingface.co](https://huggingface.co/models?search=microsoft/trocr) and use the awesome transformers library, for longevity and simplicity.

The notebook is structured as follows:
* Setting up the Environment
* Using the Model (Running Inference)

## Setting up the Environment

#### Dependencies and Runtime

If you're running this notebook in Google Colab, most of the dependencies are already installed and we don't need the GPU for this particular example. 

If you decide to run this on many (>thousands) images and want the inference to go faster though, you can select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`.

We need to install huggingface `transformers` for this example to run, so execute the command below to setup the dependencies. We use the version compiled directly from the latest source (at the time of writing this is the only way to access the transforemrs TrOCR model code).


```
!pip install -q git+https://github.com/huggingface/transformers.git
```

## Using the Model (Running Inference)

Let's define a function for us to get images from the web. We execute this function to download an image with a line of handwritten text and display it.


```
import requests
from IPython.display import display
from PIL import Image

def show_image(url):
  img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
  display(img)
  return img

handwriting1 = show_image('https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg')
```


![png](OCR_from_Images_with_Transformers_files/OCR_from_Images_with_Transformers_8_0.png)


Now we want to load the model to recognise handwritten text.

Specifically we are running the following steps:

* Load the processor, `TrOCRProcessor`, which processes our input image and converts it into a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. The processor also adds absolute position embeddings and this sequence is fed to the layers of the Transformer encoder.
* Load the model, `VisionEncoderDecoderModel`, which consists of the image encoder and the text decoder.
* Define `ocr_image` function - We define the function for inferencing which takes our `src_img`, the input image we have downloaded. It will then run both the processor and the model inference and produce the output OCR text that has been recognised from the image.


```
import transformers
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def ocr_image(src_img):
  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

We now run our `ocr_image` function on the line of handwritten text in the image we have downloaded previously (and stored in `handwriting1`).


```
ocr_image(handwriting1)
```




    'industry, " Mr. Brown commented icily. " Let us have a'



Lets try on another image with handwritten text.


```
ocr_image(show_image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoolxi9yWGAT5SLZShv8vVd0bz47UWRzQC19fDTeE8GmGv_Rn-PCF1pP1rrUx8kOjA4gg&usqp=CAU'))
```


![png](OCR_from_Images_with_Transformers_files/OCR_from_Images_with_Transformers_14_0.png)





    'Hope you have done it.'




```
import transformers
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def ocr_print_image(src_img):
  pixel_values = print_processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = print_model.generate(pixel_values)
  return print_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

We download an image with noisy printed text, a scanned receipt.


```
receipt = show_image('https://github.com/zzzDavid/ICDAR-2019-SROIE/raw/master/data/img/000.jpg')
```


![png](OCR_from_Images_with_Transformers_files/OCR_from_Images_with_Transformers_17_0.png)


As the model processes a line of text, we crop the image to include on of the lines of text in the receipt and send it to our model.


```
receipt_crop = receipt.crop((0, 80, receipt.size[0], 110))
display(receipt_crop)
ocr_print_image(receipt_crop)
```


![png](OCR_from_Images_with_Transformers_files/OCR_from_Images_with_Transformers_19_0.png)





    'BOOK TA,K(TAMAN DAYA) SDN BHD'

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
