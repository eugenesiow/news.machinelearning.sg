---
title: "Named Entity Recognition on Weibo in Mandarin"
date: 2020-12-24T08:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Named Entity Recognition", "命名实体识别", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "ner_weibo.png"
    alt: "Named Entity Recognition on Weibo in Mandarin"
---

> **tl;dr** A step-by-step tutorial to train a state-of-the-art model with flair and BERT for named entity recognition 
>(NER) in mandarin, 中文命名实体识别, on a Weibo dataset. Our model beats the state-of-the-art by 20+ percentage points.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_Mandarin_Weibo.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Named Entity Recognition in Mandarin on a Weibo Social Media Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_Mandarin_Weibo.ipynb "Open in Colab")

Notebook to train a [flair](https://github.com/flairNLP/flair) model in mandarin using stacked embeddings (with word and BERT embeddings) to perform named entity recognition (NER). 

The [dataset](https://github.com/hltcoe/golden-horse) used contains 1,890 Sina Weibo messages annotated with four entity types (person, organization, location and geo-political entity), including named and nominal mentions from the paper [Peng et al. (2015)](https://www.aclweb.org/anthology/D15-1064/) and with revised annotated data from [He et al. (2016)](https://arxiv.org/abs/1611.04234).

The current state-of-the-art model on this dataset is from [Peng et al. (2016)](https://www.aclweb.org/anthology/P16-2025/) with an average F1-score of **47.0%** (Table 1) and from [Peng et al. (2015)](https://www.aclweb.org/anthology/D15-1064.pdf) with an F1-score of **44.1%** (Table 2). The authors say that the poor results on the test set show the "difficulty of this task" - which is true a sense because the dataset is really quite small for the NER task with 4 classes (x2 as they differentiate nominal and named entities) with a test set of only 270 sentences.

Our flair model is able to improve the state-of-the-art with an F1-score of **67.5%**, which is a cool 20+ absolute percentage points better than the current state-of-the-art performance.

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

The dataset, including the train, test and dev sets, has just been included in the `0.7 release` of flair, hence, we just use the `flair.datasets` loader to load the `WEIBO_NER` dataset into the flair `Corpus`. The [raw datasets](https://github.com/87302380/WEIBO_NER) are also available on Github.

```
import flair.datasets
from flair.data import Corpus
corpus = flair.datasets.WEIBO_NER()
print(corpus)
```

We can see that the total 1,890 sentences have already been split into train (1,350), dev (270) and test (270) sets in a 5:1:1 ratio.

## Training and Testing the Model

#### Train the Model

To train the flair `SequenceTagger`, we use the `ModelTrainer` object with the corpus and the tagger to be trained. We use flair's sensible default options in the `.train()` method, while specifying the output folder for the `SequenceTagger` model to be `/content/model/`. We also set the `embeddings_storage_mode` to be `gpu` to utilise the GPU to store the embeddings for more speed. Note that if you run this with a larger dataset you might run out of GPU memory, so be sure to set this option to `cpu` - it will still use the GPU to train but the embeddings will not be stored in the CPU and there will be a transfer to the GPU each epoch.

Be prepared to allow the training to run for about 0.5 to 1 hour. We set the `max_epochs` to 50 so the the training will complete faster, for higher F1-score you can increase this number to 100 or 150.


```
import flair
from typing import List
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, BytePairEmbeddings

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# For an even faster training time, you can comment out the BytePairEmbeddings
# Note: there will be a small drop in performance if you do so.
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('zh-crawl'),
    BytePairEmbeddings('zh'),
    BertEmbeddings('bert-base-chinese'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('/content/model/',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=50,
              embeddings_storage_mode='gpu')
```

We see that the output accuracy (F1-score) for our new model is **67.5%** (F1-score (micro) 0.6748). We use micro F1-score (rather than macro F1-score) as there are multiple entity classes in this setup with [class imbalance](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin).

> We have a new SOTA NER model in mandarin, over 20 percentage points (absolute) better than the previous SOTA for this Weibo dataset!

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `tagger.predict(sentence)`. Do note that for mandarin each character needs to be split with spaces between each character (e.g. `一 节 课 的 时 间`) so that the tokenizer will work properly to split them to tokens (if you're processing them for input into the model when building an app). For more information on this, check out the [flair tutorial on tokenization](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_1_BASICS.md#tokenization).


```
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus

# Load the model that we trained, you can comment this out if you already have 
# the model loaded (e.g. if you just ran the training)
tagger: SequenceTagger = SequenceTagger.load("/content/model/final-model.pt")

# Load the WEIBO corpus and use the first 5 sentences from the test set
corpus = flair.datasets.WEIBO_NER()
for idx in range(0, 5):
  sentence = corpus.test[idx]
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
shutil.move('/content/model/', "/content/drive/My Drive/model/")
```

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## AI Glossary in Mandarin

Visit or star the [eugenesiow/ai-glossary-mandarin](https://github.com/eugenesiow/ai-glossary-mandarin) repository on 
Github if you need an English-to-Mandarin dictionary of AI terminology grouped topically by areas (e.g. NLP) and tasks (e.g. NER):

{{< ghbtns eugenesiow ai-glossary-mandarin "AI Glossary in Mandarin" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
