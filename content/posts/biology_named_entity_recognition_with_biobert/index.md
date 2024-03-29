---
title: "Biology Named Entity Recognition with BioBERT"
date: 2020-12-30T10:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Named Entity Recognition", "Biology", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "biobert.jpg"
    alt: "Biology NER with BioBERT to Extract Diseases and Chemicals."
    relative: true
---

> **tl;dr** A step-by-step tutorial to train a BioBERT model for named entity recognition (NER), extracting diseases and
>chemical on the BioCreative V CDR task corpus. Our model is #3-ranked and within 0.6 percentage points of the state-of-the-art.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_BC5CDR.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

## Named Entity Recognition on BC5CDR (Chemical + Disease Corpus) with BioBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_BC5CDR.ipynb "Open in Colab")

Notebook to train/fine-tune a BioBERT model to perform named entity recognition (NER). 

The [dataset](https://github.com/shreyashub/BioFLAIR/tree/master/data/ner/bc5cdr) used is a pre-processed version of the BC5CDR (BioCreative V CDR task corpus: a resource for  relation extraction) dataset from [Li et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27161011/).

The current state-of-the-art model on this dataset is the NER+PA+RL model from [Nooralahzadeh et al. (2019)](https://www.aclweb.org/anthology/D19-6125/) has an F1-score of [**89.93%**](https://paperswithcode.com/sota/named-entity-recognition-ner-on-bc5cdr). The authors did not release the source code for the paper.

Our model trained on top of BioBERT has an F1-score of **89.3%** which is slightly worse than the state-of-the-art but almost as good as the #2 [BioFlair](https://github.com/shreyashub/BioFLAIR)!

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

The dataset, includes train, test and dev sets, which we pull from the [Github repository](https://github.com/shreyashub/BioFLAIR/tree/master/data/ner/bc5cdr).


```
import urllib.request
from pathlib import Path

def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/train.txt', '/content/data/train.txt')
download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/test.txt', '/content/data/test.txt')
download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/dev.txt', '/content/data/dev.txt')
```

Since the data is formatted in the CoNLL `BIO` type format (you can read more on the tagging format from this [wikipedia article](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))), we need to format it into a `pandas` dataframe with the following function. The 3 important columns in the dataframe are a word token (for mandarin this is a single character), a `BIO` label and a sentence_id to differentiate samples/sentences.


```
import pandas as pd
def read_conll(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels'],
                    quoting = 3, skip_blank_lines = False)
    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')] # Remove the -DOCSTART- header
    df['sentence_id'] = (df.words == '').cumsum()
    return df[df.words != '']
```

Now we execute the function on the train, test and dev sets we have downloaded from Github. We also `.head()` the training set dataframe for the first 100 rows to check that the words, labels and sentence_id have been split properly.


```
train_df = read_conll('/content/data/train.txt')
test_df = read_conll('/content/data/test.txt')
dev_df = read_conll('/content/data/dev.txt')
train_df.head(100)
```

We now print out the statistics (number of sentences) of the train, dev and test sets.


```
data = [[train_df['sentence_id'].nunique(), test_df['sentence_id'].nunique(), dev_df['sentence_id'].nunique()]]

# Prints out the dataset sizes of train and test sets per label.
pd.DataFrame(data, columns=["Train", "Test", "Dev"])
```


## Training and Testing the Model

#### Set up the Training Arguments

We set up the training arguments. Here we train to 10 epochs to get accuracy close to the SOTA. The train, test and dev sets are relatively small so we don't have to wait too long. We set a sliding window as NER sequences can be quite long and because we have limited GPU memory we can't increase the `max_seq_length` too long.


```
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'fp16': True,
    'output_dir': '/outputs/',
    'best_model_dir': '/outputs/best_model/',
    'evaluate_during_training': True,
}
```

The following line of code saves (to the variable `custom_labels`) a set of all the NER tags/labels in the dataset.


```
custom_labels = list(train_df['labels'].unique())
print(custom_labels)
```

#### Train the Model

Once we have setup the `train_args` dictionary, the next step would be to train the model. We use the pre-trained BioBERT model (by [DMIS Lab, Korea University](https://huggingface.co/dmis-lab)) from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library as the base and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the NER (sequence tagging) model with just a few lines of code.


```
from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the bio BERT pre-trained model.
model = NERModel('bert', 'dmis-lab/biobert-v1.1', labels=custom_labels, args=train_args)

# Train the model
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df, eval_data=dev_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, preds_list = model.eval_model(test_df)
```

The F1-score for the model is **89.3%** ('f1_score': 0.8927974947807933).

> For now thats the #3-ranked SOTA NER model on BC5CDR!

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(samples)`. First we get a sentence from the test set and print it out. Then we run the prediction on the sentence.


```
sample = test_df[test_df.sentence_id == 10].words.str.cat(sep=' ')
print(sample)
```


```
samples = [sample]
predictions, _ = model.predict(samples)
for idx, sample in enumerate(samples):
  print('{}: '.format(idx))
  for word in predictions[idx]:
    print('{}'.format(word))
```

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

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
