---
title: "Sentiment Analysis on Movie Reviews with XLNet"
date: 2020-12-22T16:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Sentiment Analysis", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "sentiment_analysis_movies.png"
    alt: "Sentiment Analysis on Movie Reviews"
---

> **tl;dr** A step-by-step tutorial to train a sentiment analysis model to classify polarity of IMDB movie reviews 
>with XLNet using a free Jupyter Notebook in the cloud. 

## The IMDB Movie Reviews Dataset and XLNet

The Internet Movie Database (IMDb) movie reviews dataset is a very well-established benchmark (since 2011) for sentiment 
analysis performance. It's probably the first large-ish (50,000 train+test), balanced sentiment analysis dataset, 
making it a very nice dataset for benchmarking on. 

The model with state-of-the-art performance on this dataset is XLNet, [Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf), 
which has an accuracy of [96.2%](http://nlpprogress.com/english/sentiment_analysis.html). In this practical, we train a
an XLNet base model to just 1 epoch (training more epochs to minimise loss gives higher performance, but we are impatient
in a practical). We get an accuracy of 92.2% due to these limitations, which is still pretty decent.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sentiment_Analysis_Movie_Reviews.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Sentiment Analysis on IMDB Movie Reviews

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sentiment_Analysis_Movie_Reviews.ipynb "Open in Colab")

Notebook to train an XLNet model to perform sentiment analysis. The [dataset](https://ai.stanford.edu/~amaas/data/sentiment/) used is a balanced collection of (50,000 - 1:1 train-test ratio) IMDB movie reviews with binary labels: **`postive`** or **`negative`** from the paper by [Maas et al. (2011)](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf). The current state-of-the-art model on this dataset is XLNet by [Yang et al. (2019)](https://arxiv.org/pdf/1906.08237.pdf) which has an accuracy of [96.2%](http://nlpprogress.com/english/sentiment_analysis.html). We get an accuracy of 92.2% due to the limitations of GPU memory on Colab (we use XLNet base instead of the large model), train to 1 epoch only for speed and we are unable to replicate all the hyperparameters (sequence length).

The notebook is structured as follows:
* Setting up the GPU Environment
* Getting Data
* Training and Testing the Model
* Using the Model (Running Inference)

#### Task Description

> Sentiment analysis is the task of classifying the polarity of a given text.

## Setting up the GPU Environment

#### Ensure we have a GPU runtime

If you're running this notebook in Google Colab, select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`. This will allow us to use the GPU to train the model subsequently.

#### Install Dependencies and Restart Runtime


```shell script
!pip install -q transformers
!pip install -q simpletransformers
!pip install -q datasets
```

You might see the error `ERROR: google-colab X.X.X has requirement ipykernel~=X.X, but you'll have ipykernel X.X.X which is incompatible` after installing the dependencies. **This is normal** and caused by the `simpletransformers` library.

The **solution** to this will be to **reset the execution environment** now. Go to the menu `Runtime` > `Restart runtime` then continue on from the next section to download and process the data.

## Getting Data

#### Dataset Description

The IMDb dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative (this is the polarity). The dataset contains of an even number of positive and negative reviews (balanced). Only highly polarizing reviews are considered. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. No more than 30 reviews are included per movie. There are 25,000 highly polar movie reviews for training, and 25,000 for testing. 

#### Pulling the data from `huggingface/datasets`

We use Hugging Face's awesome datasets library to get the pre-processed version of the original [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/). 

The code below pulls the train and test datasets from [huggingface/datasets](https://github.com/huggingface/datasets) using `load_dataset('imdb')` and transform them into `pandas` dataframes for use with the `simpletransformers` library to train the model.


```python
import pandas as pd
from datasets import load_dataset

dataset_train = load_dataset('imdb',split='train')
dataset_train.rename_column_('label', 'labels')
train_df=pd.DataFrame(dataset_train)

dataset_test = load_dataset('imdb',split='test')
dataset_test.rename_column_('label', 'labels')
test_df=pd.DataFrame(dataset_test)
```

Once done we can take a look at the `head()` of the training set to check if our data has been retrieved properly.

```python
train_df.head()
```

We also double check the dataset properties are exactly the same as those reported in the papers (25,000 train, 25,000 test size, balanced). **`0`** is the **`negative`** polarity class while **`1`** is the **`positive`** polarity class.

```python
data = [[train_df.labels.value_counts()[0], test_df.labels.value_counts()[0]], 
        [train_df.labels.value_counts()[1], test_df.labels.value_counts()[1]]]
# Prints out the dataset sizes of train test and validate as per the table.
pd.DataFrame(data, columns=["Train", "Test"])
```

## Training and Testing the Model

#### Set the Hyperparmeters

First we setup the hyperparamters, using the hyperparemeters specified in the  Yang et al. (2019) paper whenever possible (we take Yelp hyperparameters as IMDB ones are not specified). The comparison of hyperparameters is shown in the table below. The major difference is due to GPU memory limitations we are unable to use a sequence length of 512, instead we use a sliding window on a sequence length of 64. We also train to 1 epoch only as want the training to complete fast.

|Parameter  	    |Ours  	    |Paper  	|
|-	                |-	        |-	        |
|Epochs  	        |1  	    |?  	    |
|Batch Size  	    |128  	  |128  	    |
|Seq Length  	    |64  	    |512  	    |
|Learning Rate      |1e-5       |1e-5       |
|Weight decay       |1e-2       |1e-2       |


```python
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 1,
    'learning_rate': 0.00001,
    'weight_decay': 0.01,
    'train_batch_size': 128,
    'fp16': True,
    'output_dir': '/outputs/',
}
```

#### Train the Model

Once we have setup the hyperparemeters in the `train_args` dictionary, the next step would be to train the model. We use the [`xlnet-base-cased` model](https://huggingface.co/xlnet-base-cased) from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the classification model with just 2 lines of code.

[XLNet](https://arxiv.org/pdf/1906.08237.pdf) is an auto-regressive language model which outputs the joint probability of a sequence of tokens based on the transformer architecture with recurrence. Although its also bigger than BERT and has a (slightly) different architecture, it's change in training objective is probably the biggest contribution. It's training objective is to predict each word in a sequence using any combination of other words in that sequence which seems to perform better on ambiguous contexts.


```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the XLNet base cased pre-trained model.
model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=2, args=train_args) 

# Train the model, there is no development or validation set for this dataset 
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
```

We see that the output accuracy from the model after training for 1 epoch is **92.2%** ('acc': 0.92156).

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(input_list)`.

```python
samples = ['The script is nice.Though the casting is absolutely non-watchable.No style. the costumes do not look like some from the High Highbury society. Comparing Gwyneth Paltrow with Kate Beckinsale I can only say that Ms. Beckinsale speaks British English better than Ms. Paltrow, though in Ms. Paltrow\'s acting lies the very nature of Emma Woodhouse. Mr. Northam undoubtedly is the best Mr. Knightley of all versions, he is romantic and not at all sharp-looking and unfeeling like Mr. Knightley in the TV-version. P.S.The spectator cannot see at all Mr. Elton-Ms. Smith relationship\'s development as it was in the motion version, so one cannot understand where was all Emma\'s trying of make a Elton-Smith match (besides of the portrait).']
predictions, _ = model.predict(samples)
label_dict = {0: 'negative', 1: 'positive'}
for idx, sample in enumerate(samples):
  print('{} - {}: {}'.format(idx, label_dict[predictions[idx]], sample))
```

We can connect to Google Drive with the following code to save any files you want to persist. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```python
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the model checkpount files which are saved in the `/outputs/` directory to your Google Drive.


```python
import shutil
shutil.move('/outputs/', "/content/drive/My Drive/outputs/")
```

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
