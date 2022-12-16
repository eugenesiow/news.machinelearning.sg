---
title: "Text to Speech with Silero"
date: 2021-05-31T16:00:00+08:00
tags: ["Speech", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Text-to-Speech", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "tts_silero.jpg"
    relative: true
    alt: "Text to Speech with Silero. Image from Unsplash by Volodymyr Hryshchenko."
---

> **tl;dr** A step-by-step tutorial to generate spoken audio from text automatically using the enterprise-grade SileroTTS model and applying speech enhancement.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Text_to_Speech_with_Silero.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

# Text to Speech with Silero

Notebook to convert an input piece of text into an speech audio file automatically.

[Text-To-Speech synthesis](https://paperswithcode.com/task/text-to-speech-synthesis) is the task of converting written text in natural language to speech.

The [model used](https://pytorch.org/hub/snakers4_silero-models_tts/) is one of the pre-trained `silero_tts` model. It was trained on a private dataset.

Do note that the Silero models are [licensed](https://habr.com/ru/post/549482/) under a [GPU A-GPL 3.0 License](https://github.com/snakers4/silero-models/blob/master/LICENSE) where you have to provide source code if you are using it for commercial purposes.

The notebook is structured as follows:
* Setting up the Environment
* Using the Model (Running Inference)
* Apply Speech Enhancement/Noise Reduction

# Setting up the Environment

#### Dependencies and Runtime

If you're running this notebook in Google Colab, most of the dependencies are already installed and we don't need the GPU for this particular example. 

If you decide to run this on many (>thousands) images and want the inference to go faster though, you can select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`.

We need to install `torchaudio` and `omegaconf` for this example to run, so execute the command below to setup the dependencies.


```
!pip install -q torchaudio omegaconf
```

# Using the Model (Running Inference)

Now we want to load and run the specific Silero 16khz english speaker model. The full set of [available models](https://github.com/snakers4/silero-models#text-to-speech) include models in German and Russian.

Specifically we are running the following steps:

* `torch.hub.load()` - Downloads and loads the pre-trained model from torchhub. In particular, we specify to use the `silero_tts` model with the `en` (English) language speaker `lj_16khz`.
* `model.to(device)` - We load the model to the `CPU` (the default) or `GPU` (if you set this up in the previous section) for inferencing.


```
import torch

language = 'en'
speaker = 'lj_16khz'
device = torch.device('cpu')
model, symbols, sample_rate, _, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
model = model.to(device)
``` 


Now we define the `example_text` variable, a piece of text that we want to convert to a speech audio file. Next, we synthesize/generate the audio file.

The notebook will then display the audio sample produced for us to playback.


```
from IPython.display import Audio, display

example_text = 'What is umbrage? According to the Oxford Languages dictionary, Umbrage is a noun that means offence or annoyance.'

audio = apply_tts(texts=[example_text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)

display(Audio(audio[0], rate=sample_rate))
```

We notice that there is some noise in the generated sample which can easily be reduced to enhance the quality of speech using a speech enhancement model. We try this in the next section. This is entirely optional.

# Apply Speech Enhancement/Noise Reduction

We use the simple and convenient LogMMSE algorithm (Log Minimum Mean Square Error) with the [logmmse library](https://github.com/wilsonchingg/logmmse).


```
!pip install -q logmmse
```

Run the LogMMSE algorithm on the generated audio `audio[0]` and  display the enhanced audio sample produced in an audio player.


```
import numpy as np
from logmmse import logmmse

enhanced = logmmse(np.array(audio[0]), sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
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
