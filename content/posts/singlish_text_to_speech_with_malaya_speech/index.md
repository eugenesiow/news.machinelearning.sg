---
title: "Singlish Text to Speech with Malaya Speech"
date: 2021-10-21T08:00:00+08:00
tags: ["Speech", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Text-to-Speech", "Jupyter Notebook", "Colab", "Natural Language Processing"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "singlish_text_to_speech.jpg"
    alt: "Singlish Text to Speech with Malaya Speech"
---

> **tl;dr** A step-by-step tutorial to generate spoken singlish audio from text automatically using a pipeline of a Malaya Speech model and applying speech enhancement. 

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Singlish_Text_to_Speech_with_Malaya_Speech.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Singlish Text to Speech with Malaya Speech

Notebook to convert an input piece of text into an speech audio file automatically.

[Text-To-Speech synthesis](https://paperswithcode.com/task/text-to-speech-synthesis) is the task of converting written text in natural language to speech.

The [model used](https://malaya-speech.readthedocs.io/en/latest/tts-singlish.html) is one of the pre-trained models from `malaya_speech`.

The notebook is structured as follows:
* Setting up the Environment
* Using the Model (Running Inference)
* Apply Speech Enhancement/Noise Reduction

## Setting up the Environment

#### Dependencies and Runtime

If you're running this notebook in Google Colab, most of the dependencies are already installed and we don't need the GPU for this particular example. 

If you decide to run this on many (>thousands) images and want the inference to go faster though, you can select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`.

We need to install `malaya` and `malaya_speech` for this example to run, so execute the command below to setup the dependencies.


```
!pip install -q malaya malaya_speech
```

## Using the Model (Running Inference)

Now we want to load and run the specific singlish text-to-speech model.

Specifically we are running the following steps:

* Define `load_model` - We define the `load_model` function, which will download the fastspeech2 and melgan models of `model_name` using the `malaya_speech` library. 
* Define `predict` - We define the `predict` function for inferencing. It will take in the models that we loaded and an `input_text` string. In this function/method, we run the `.predict` of the fastspeech model and then pass it through the `melgan` model to get the audio output vector.
* Run `load_model`


```
import malaya_speech


def load_models(model_name):
  fs2 = malaya_speech.tts.fastspeech2(model = model_name)
  melgan = malaya_speech.vocoder.melgan(model = model_name)
  return fs2, melgan

def predict(input_text, singlish, melgan):
  r_singlish = singlish.predict(input_text)
  y_ = melgan(r_singlish['postnet-output'])
  data = malaya_speech.astype.float_to_int(y_)
  return data

fs2, melgan = load_models('female-singlish')
```

Now we define the `input_text` variable, a piece of text that we want to convert to a speech audio file. Next, we synthesize/generate the audio file.

The notebook will then display the audio sample produced for us to playback.


```
from IPython.display import Audio, display

sample_rate = 22050
input_text = 'The second Rental Support Scheme payout will be disbursed about a month early to ensure businesses get cash flow relief as soon as possible, say IRAS and MOF.'
audio = predict(input_text, fs2, melgan)
display(Audio(audio, rate=sample_rate))
```

Other available models can be seen by running the code below:


```
malaya_speech.tts.available_fastspeech2()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size (MB)</th>
      <th>Quantized Size (MB)</th>
      <th>Combined loss</th>
      <th>understand punctuations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>male</th>
      <td>125</td>
      <td>31.7</td>
      <td>1.846</td>
      <td>False</td>
    </tr>
    <tr>
      <th>male-v2</th>
      <td>65.5</td>
      <td>16.7</td>
      <td>1.886</td>
      <td>False</td>
    </tr>
    <tr>
      <th>female</th>
      <td>125</td>
      <td>31.7</td>
      <td>1.744</td>
      <td>False</td>
    </tr>
    <tr>
      <th>female-v2</th>
      <td>65.5</td>
      <td>16.7</td>
      <td>1.804</td>
      <td>False</td>
    </tr>
    <tr>
      <th>husein</th>
      <td>125</td>
      <td>31.7</td>
      <td>0.6411</td>
      <td>False</td>
    </tr>
    <tr>
      <th>husein-v2</th>
      <td>65.5</td>
      <td>16.7</td>
      <td>0.7712</td>
      <td>False</td>
    </tr>
    <tr>
      <th>haqkiem</th>
      <td>125</td>
      <td>31.7</td>
      <td>0.5663</td>
      <td>True</td>
    </tr>
    <tr>
      <th>female-singlish</th>
      <td>125</td>
      <td>31.7</td>
      <td>0.5112</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



We notice that there is some noise in the previously generated sample which can easily be reduced to enhance the quality of speech using a speech enhancement model. We try this in the next section. This is entirely optional.

## Apply Speech Enhancement/Noise Reduction

We use the simple and convenient LogMMSE algorithm (Log Minimum Mean Square Error) with the [logmmse library](https://github.com/wilsonchingg/logmmse).


```
!pip install -q logmmse
```

Run the LogMMSE algorithm on the generated audio `audio[0]` and  display the enhanced audio sample produced in an audio player.


```
import numpy as np
from logmmse import logmmse

enhanced = logmmse(np.array(audio), sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
display(Audio(enhanced, rate=sample_rate))
```

Save the enhanced audio to file.


```
from scipy.io.wavfile import write

write('/content/audio.wav', sample_rate, enhanced)
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
shutil.move('/content/audio.wav', '/content/drive/My Drive/audio.wav')
```

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
