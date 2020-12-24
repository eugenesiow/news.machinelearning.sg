---
title: "Train a Named Entity Recognition (NER) Model Using FLAIR"
date: 2020-12-23T09:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Named Entity Recognition", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "ner.png"
    alt: "Named Entity Recognition (NER) Model Using FLAIR"
---

> **tl;dr** A step-by-step tutorial to train a state-of-the-art model for named entity recognition (NER), 
>the task of identifying persons, organizations and locations from a piece of text.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_CoNLLpp.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

## Named Entity Recognition on the CoNLL++ Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_CoNLLpp.ipynb "Open in Colab")

Notebook to train a [flair](https://github.com/flairNLP/flair) model using stacked embeddings (with word and flair 
contextual embeddings) to perform named entity recognition (NER). The dataset used is the [CoNLL 2003](https://www.aclweb.org/anthology/W03-0419.pdf) dataset for NER (train, dev) with a manually corrected (improved/cleaned) test set from the [CrossWeigh paper](https://arxiv.org/abs/1909.01441) called [CoNLL++](https://github.com/ZihanWangKi/CrossWeigh#data). The current state-of-the-art model on this dataset is from the CrossWeigh paper (also using flair) by [Wang et al. (2019)](https://www.aclweb.org/anthology/D19-1519/) with F1-score of [94.3%](http://nlpprogress.com/english/named_entity_recognition.html). Without using pooled-embeddings, CrossWeigh and training to a max 50 instead of 150 epochs, we get a micro F1-score of 93.5%, within 0.7 of a percentage point of the SOTA.

The notebook is structured as follows:
* Setting up the GPU Environment
* Getting Data
* Training and Testing the Model
* Using the Model (Running Inference)

#### Task Description

> Named entity recognition (NER) is the task of tagging entities in text with their corresponding type. Approaches typically use BIO notation, which differentiates the beginning (B) and the inside (I) of entities. O is used for non-entity tokens.

## Setting up the GPU Environment

#### Ensure we have a GPU runtime

If you're running this notebook in Google Colab, select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`. This will allow us to use the GPU to train the model subsequently.

#### Install Dependencies


```
pip install -q flair
```

## Getting Data

We download the data (train, test and dev sets) in the BIO format (each token in each of the sentences are tagged with Begin, Inside or Outside labels) as text files from the [CoNLL++ repository](https://github.com/ZihanWangKi/CrossWeigh) and save them to the `/content/data/` folder.


```
import urllib.request
from pathlib import Path

def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_train.txt', '/content/data/conllpp_train.txt')
download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_dev.txt', '/content/data/conllpp_dev.txt')
download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_test.txt', '/content/data/conllpp_test.txt')
```

Now we will use flair's built in `ColumnCorpus` object to load in our `conllpp_train.txt`, `conllpp_test.txt` and `conllpp_dev.txt` files in the `/content/data/` folder.


```
from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0: 'text', 3: 'ner'}
corpus: Corpus = ColumnCorpus('/content/data/', columns,
                              train_file='conllpp_train.txt',
                              test_file='conllpp_test.txt',
                              dev_file='conllpp_dev.txt')
```

To check that the sentences/size of the train, test and development set tally exactly with the Table 1 (train: 14987, test: 3684, dev: 3466) in the [CoNLL 2003 paper](https://www.aclweb.org/anthology/W03-0419.pdf), we will get the `len()` of the `.train`, `.test` and `.dev` sets from the `ColumnCorpus` object and print it out as a table.


```
import pandas as pd
data = [[len(corpus.train), len(corpus.test), len(corpus.dev)]]
# Prints out the dataset sizes of train test and development in a table.
pd.DataFrame(data, columns=["Train", "Test", "Development"])
```

## Training and Testing the Model

#### Train the Model

To train the flair `SequenceTagger`, we use the `ModelTrainer` object with the corpus and the tagger to be trained. We use flair's sensible default options while specifying the output folder for the `SequenceTagger` model to be `/content/model/conllpp`. We also set the `embeddings_storage_mode` to be `gpu` to utilise the GPU. Note that if you run this with a larger dataset than CoNLL++ and you run out of GPU memory, be sure to set this option to `cpu`.

Be prepared to allow the training to run for a few hours.


```
import flair
from typing import List
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# For faster training and smaller models, we can comment out the flair embeddings.
# This will significantly affect the performance though.
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('/content/model/conllpp',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=50,
              embeddings_storage_mode='gpu')
```


We see that the output accuracy (F1-score) for our new model is **93.5%** (F1-score (micro) 0.9354).

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `tagger.predict(sentence)`.


```
from flair.data import Sentence
from flair.models import SequenceTagger

input_sentence = 'My name is Eugene, I currently live in Singapore, I work for DSO.'
tagger: SequenceTagger = SequenceTagger.load("/content/model/conllpp/final-model.pt")
sentence: Sentence = Sentence(input_sentence)
tagger.predict(sentence)
print(sentence.to_tagged_string())
```

We can connect to Google Drive with the following code to save any files you want to persist. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the model files from our local directory to your Google Drive.


```
import shutil
shutil.move('/content/model/conllpp/', "/content/drive/My Drive/model/")
```

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
