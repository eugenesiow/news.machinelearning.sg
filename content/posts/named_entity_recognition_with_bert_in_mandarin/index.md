---
title: "Named Entity Recognition with BERT in Mandarin"
date: 2020-12-24T10:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Named Entity Recognition", "命名实体识别", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "ner_msra.png"
    relative: true
    alt: "Named Entity Recognition with BERT in Mandarin"
---

> **tl;dr** A step-by-step tutorial to train a state-of-the-art model with BERT for named entity recognition 
>(NER) in mandarin, 中文命名实体识别. Our model beats the state-of-the-art by 0.7 percentage points.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_Mandarin_MSRA.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Named Entity Recognition in Mandarin on the MSRA/SIGHAN2006 Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_Mandarin_MSRA.ipynb "Open in Colab")

Notebook to train/fine-tune a pre-trained chinese BERT model to perform named entity recognition (NER). 

The [dataset](https://github.com/yzwww2019/Sighan-2006-NER-dataset) used is the SIGHAN 2006, or commonly known as the MSRA NER dataset. It contains 46,364 samples in the training set and 4,365 samples in the test set. The original workshop/paper for the dataset is by [Levow (2006)](https://faculty.washington.edu/levow/papers/sighan06.pdf).

The current state-of-the-art model on this dataset is the Lattice LSTM from [Zhang et al. (2018)](https://arxiv.org/pdf/1805.02023.pdf) with an F1-score of **93.2%**.

Our BERT model (with only 1 epoch training) has an F1-score of **93.9%** which is slightly better than the state-of-the-art!

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

#### Install Dependencies and Restart Runtime


```
!pip install -q transformers
!pip install -q simpletransformers
```

You might see the error `ERROR: google-colab X.X.X has requirement ipykernel~=X.X, but you'll have ipykernel X.X.X which is incompatible` after installing the dependencies. **This is normal** and caused by the `simpletransformers` library.

The **solution** to this will be to **reset the execution environment** now. Go to the menu `Runtime` > `Restart runtime` then continue on from the next section to download and process the data.

## Getting Data

#### Pulling the data from Github

The dataset, includes train and test sets, which we pull from a [Github repository](https://github.com/yzwww2019/Sighan-2006-NER-dataset).


```
import urllib.request
from pathlib import Path

def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

download_file('https://raw.githubusercontent.com/yzwww2019/Sighan-2006-NER-dataset/master/train.txt', '/content/data/train.txt')
download_file('https://raw.githubusercontent.com/yzwww2019/Sighan-2006-NER-dataset/master/test.txt', '/content/data/test.txt')
```

Since the data is formatted in the CoNLL `BIO` type format (you can read more on the tagging format from this [wikipedia article](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))), we need to format it into a `pandas` dataframe with the following function. The 3 columns in the dataframe are a word token (for mandarin this is a single character), a `BIO` label and a sentence_id to differentiate samples/sentences.


```
import pandas as pd
def read_conll(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'labels'], skip_blank_lines = False)
    df['sentence_id'] = (df.words == '').cumsum()
    return df[df.words != '']
```

Now we execute the function on the train and test sets we have downloaded from Github. We also `.head()` the training set dataframe for the first 100 rows to check that the words, labels and sentence_id have been split properly.


```
train_df = read_conll('/content/data/train.txt')
test_df = read_conll('/content/data/test.txt')
train_df.head(100)
```

We now print out the statistics of the train and test set. We can see that we have the right distribution of 46,364 samples in the training set and 4,365 samples in the test set.


```
data = [[train_df['sentence_id'].nunique(), test_df['sentence_id'].nunique()]]

# Prints out the dataset sizes of train and test sets per label.
pd.DataFrame(data, columns=["Train", "Test"])
```

## Training and Testing the Model

#### Set up the Training Arguments

We set up the training arguments. Here we train to 1 epoch to reduce the training time as much as possible (we are impatient). We set a sliding window as NER sequences can be quite long and because we have limited GPU memory we can't increase the `max_seq_length` too long.


```
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 1,
    'train_batch_size': 32,
    'fp16': True,
    'output_dir': '/outputs/',
}
```

#### Train the Model

Once we have setup the `train_args` dictionary, the next step would be to train the model. We use the pre-trained mandarin BERT model, `bert_base_cased` from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library as the base and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the NER (sequence tagging) model with just a few lines of code.


```
from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the bert base cased pre-trained model.
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = NERModel('bert', 'bert-base-chinese', args=train_args)

# Train the model, there is no development or validation set for this dataset 
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, preds_list = model.eval_model(test_df)
```

The F1-score for the model is **93.9%** ('f1_score': 0.939079035179712).

That score is better than the previous state-of-the-art model with **93.2%** by about 0.7 percentage points (absolute).

> We have a new SOTA NER model in mandarin!

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(samples)`. Character level tokenization with spaces: Do note that for mandarin each character needs to be split with spaces between each character (e.g. `一 节 课 的 时 间`) so that the tokenizer will work properly to split them to tokens (if you're processing them for input into the model when building an app).


```
samples = ['我 的 名 字 叫 蕭 文 仁 。 我 是 新 加 坡 人 。']
predictions, _ = model.predict(samples)
for idx, sample in enumerate(samples):
  print('{}: '.format(idx))
  for word in predictions[idx]:
    print('{}'.format(word))
```

    0: 
    {'我': 'O'}
    {'的': 'O'}
    {'名': 'O'}
    {'字': 'O'}
    {'叫': 'O'}
    {'蕭': 'B-PER'}
    {'文': 'I-PER'}
    {'仁': 'I-PER'}
    {'。': 'O'}
    {'我': 'O'}
    {'是': 'O'}
    {'新': 'B-LOC'}
    {'加': 'I-LOC'}
    {'坡': 'I-LOC'}
    {'人': 'O'}
    {'。': 'O'}


We can connect to Google Drive with the following code to save any files you want to persist. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the model checkpount files which are saved in the `/outputs/` directory to your Google Drive.


```
import shutil
shutil.move('/outputs/', "/content/drive/My Drive/outputs/")
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
