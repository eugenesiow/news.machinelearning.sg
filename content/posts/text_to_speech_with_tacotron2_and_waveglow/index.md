---
title: "Text to Speech with Tacotron2 and WaveGlow"
date: 2021-05-31T18:00:00+08:00
tags: ["Speech", "Deep Learning", "Machine Learning", "Source Code", "PyTorch", "Text-to-Speech", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "tts_waveglow.jpg"
    relative: true
    alt: "Text to Speech with Tacotron2 and WaveGlow. Image from Unsplash by Hrayr Movsisyan."
---

> **tl;dr** A step-by-step tutorial to generate spoken audio from text automatically using a pipeline of Nvidia's Tacotron2 and WaveGlow models and applying speech enhancement.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Text_to_Speech_with_Tacotron2_and_WaveGlow.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

# Text to Speech with Tacotron2 and WaveGlow

Notebook to convert (synthesize) an input piece of text into a speech audio file automatically.

[Text-To-Speech synthesis](https://paperswithcode.com/task/text-to-speech-synthesis) is the task of converting written text in natural language to speech.

The models used combines a pipeline of a [Tacotron 2](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/) model that produces mel spectrograms from input text using an encoder-decoder architecture and a [WaveGlow](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/) flow-based model that consumes the mel spectrograms to generate speech. 

Both steps in the pipeline will utilise pre-trained models from the PyTorch Hub by NVIDIA. Both the Tacotron 2 and WaveGlow models are trained on a publicly available [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset.

Do note that the models are under a [BSD 3 License](https://opensource.org/licenses/BSD-3-Clause).

The notebook is structured as follows:
* Setting up the Environment
* Using the Model (Running Inference)
* Apply Speech Enhancement/Noise Reduction

# Setting up the Environment

#### Ensure we have a GPU runtime

If you're running this notebook in Google Colab, select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`. This will allow us to use the GPU to train the model subsequently.

#### Setup Dependencies

We need to install `unidecode` for this example to run, so execute the command below to setup the dependencies.


```
!pip install -q unidecode
```


# Using the Model (Running Inference)

Now we want to load the Tacotron2 and WaveGlow models from PyTorch hub and prepare the models for inference.

Specifically we are running the following steps:

* `torch.hub.load()` - Downloads and loads the pre-trained model from torchhub. In particular, we specify to use the `silero_tts` model with the `en` (English) language speaker `lj_16khz`.
* `.to(device)` - We load both the models to the `GPU` for inferencing.


```
import torch

tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2 = tacotron2.to('cuda')

waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
```

    Using cache found in /root/.cache/torch/hub/nvidia_DeepLearningExamples_torchhub
    Using cache found in /root/.cache/torch/hub/nvidia_DeepLearningExamples_torchhub


Now we define the `example_text` variable, a piece of text that we want to convert to a speech audio file. Next, we synthesize/generate the audio file.

* `tacotron2.text_to_sequence()` - Creates a tensor representation of the input text sequence (`example_text`).
* `tacotron2.infer()` - Tacotron2 generates mel spectrogram given tensor representation from the previous step (`sequence`).
* `waveglow.infer()` - Waveglow generates sound given the mel spectrogram
* `display()` - The notebook will then display a playback widget of the audio sample, `audio_numpy`.


```
from IPython.display import Audio, display
import numpy as np

example_text = 'What is umbrage? According to the Oxford Languages dictionary, Umbrage is a noun that means offence or annoyance.'

# preprocessing
sequence = np.array(tacotron2.text_to_sequence(example_text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050

display(Audio(audio_numpy, rate=rate))
```


We notice that there is some slight noise in the generated sample which can easily be reduced to enhance the quality of speech using a speech enhancement model. We try this in the next section. This is entirely optional.

# Apply Speech Enhancement/Noise Reduction

We use the simple and convenient LogMMSE algorithm (Log Minimum Mean Square Error) with the [logmmse library](https://github.com/wilsonchingg/logmmse).


```
!pip install -q logmmse
```

Run the LogMMSE algorithm on the generated audio `audio[0]` and  display the enhanced audio sample produced in an audio player.


```
import numpy as np
from logmmse import logmmse

enhanced = logmmse(audio_numpy, rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
display(Audio(enhanced, rate=rate))
```

Save the enhanced audio to file.


```
from scipy.io.wavfile import write

write('/content/audio.wav', rate, enhanced)
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
