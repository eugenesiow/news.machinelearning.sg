---
title: "Mandarin Text to Speech with Coqui TTS"
date: 2021-11-03T13:00:00+08:00
tags: ["Speech", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Text-to-Speech", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "tts_coqui.jpg"
    alt: "Mandarin Text to Speech with Coqui TTS."
---

> **tl;dr** A step-by-step tutorial to generate spoken audio from text automatically using the enterprise-grade SileroTTS model and applying speech enhancement.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Mandarin_Text_to_Speech_with_Coqui_TTS.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Mandarin Text to Speech with Coqui TTS

Notebook to convert an input piece of text into an speech audio file automatically.

[Text-To-Speech synthesis](https://paperswithcode.com/task/text-to-speech-synthesis) is the task of converting written text in natural language to speech.

The mandarin model used is one of the pre-trained [Coqui TTS](https://github.com/coqui-ai/TTS) model. This model was from the Mozilla TTS days (of which Coqui TTS is a hard-fork). The model was trained on data from the [中文标准女声音库](https://www.data-baker.com/data/index/source/) with 10000 sentences from [DataBaker Technology](https://www.data-baker.com/).

The notebook is structured as follows:
* Setting up the Environment
* Using the Model (Running Inference)
* Apply Speech Enhancement/Noise Reduction (Optional)

## Setting up the Environment

#### Dependencies and Runtime

If you're running this notebook in Google Colab, most of the dependencies are already installed and we **don't need the GPU** for this particular example. 

We need to install the Coqui TTS library called `TTS` for this example to run, so execute the command below to setup the dependencies.


```
!pip install -q TTS==0.4.1
```


# Using the Model (Running Inference)

Now we want to load the specific mandarin speaker model. You can browse the full set of [available models](https://github.com/coqui-ai/TTS/blob/main/TTS/.models.json) from Coqui.

Specifically we are running the following steps:

* `manager.download_model()` - Downloads the `tts_models/zh-CN/baker/tacotron2-DDC-GST` pre-trained model from Coqui. This model is a female `zh-cn` (mandarin) language speaker.
* `Synthesizer()` - Setup a `Sythesizer` from our model.


```
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

manager = ModelManager()
model_path, config_path, model_item = manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")
synthesizer = Synthesizer(
    model_path, config_path, None, None, None,
)
```


Now we define the `example_text` variable, a piece of mandarin text that we want to convert to a speech audio file. This particular example text asks "How are you? I'm doing fine.".

Next, we synthesize/generate the audio file with the `synthezier.tts()` function.

The notebook will then display the audio sample produced for us to playback.


```
from IPython.display import Audio, display

example_text = '你好吗？我很好。'

wavs = synthesizer.tts(example_text)

display(Audio(wavs, rate=synthesizer.output_sample_rate))
```              


We notice that there is actually very little noise in the generated sample. If we want to try to further enhance the quality of speech using a speech enhancement model we can move on to the next section. This is entirely optional.

# Apply Speech Enhancement/Noise Reduction

We use the simple and convenient LogMMSE algorithm (Log Minimum Mean Square Error) with the [logmmse library](https://github.com/wilsonchingg/logmmse).


```
!pip install -q logmmse
```

Run the LogMMSE algorithm on the generated audio `audio[0]` and  display the enhanced audio sample produced in an audio player.


```
import numpy as np
from logmmse import logmmse

enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
display(Audio(enhanced, rate=synthesizer.output_sample_rate))
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
