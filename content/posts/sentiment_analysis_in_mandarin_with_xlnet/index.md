---
title: "Sentiment Analysis in Mandarin with XLNet"
date: 2020-12-23T10:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Sentiment Analysis", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "sa_mandarin.png"
    alt: "Sentiment Analysis in Mandarin on Food Delivery Reviews"
---

> **tl;dr** A step-by-step tutorial to train a state-of-the-art model for sentiment analysis on mandarin food delivery 
>reviews using the XLNet architecture. We will use Google Colab's free Jupyter Notebook in the cloud. 

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sentiment_Analysis_Mandarin_Food_Reviews.ipynb "Open in Colab")

Continue on if you prefer reading the code here.


## Sentiment Analysis in Mandarin on Food Delivery Reviews, 情感分析

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sentiment_Analysis_Mandarin_Food_Reviews.ipynb "Open in Colab")

Notebook to train a mandarin XLNet model to perform sentiment analysis. The [dataset](https://github.com/SophonPlus/ChineseNlpCorpus#%E6%83%85%E6%84%9F%E8%A7%82%E7%82%B9%E8%AF%84%E8%AE%BA-%E5%80%BE%E5%90%91%E6%80%A7%E5%88%86%E6%9E%90) used is the unbalanced WAIMAI_10K (10,000 food delivery reviews from a food delivery platform in China). The dataset has binary labels: **`postive`** or **`negative`**. There is no published  state-of-the-art model that we know of on this dataset, however, there have been attempts using [BERT](https://github.com/BruceJust/Sentiment-classification-by-BERT) and sklearn's [SVM-SVC](https://www.programmersought.com/article/48933926195/) which report accuracy of about 89% and 85% respectively. We will train a state-of-the-art model with accuracy of 91.5% and an F1-score of 87.1%. Note that F1-score is a better measure as the dataset is unbalanced, but as the 2 previous attempts use accuracy as the measure of reporting, therefore we also report accuracy score for comparison.

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


```
!pip install -q transformers
!pip install -q simpletransformers
```

You might see the error `ERROR: google-colab X.X.X has requirement ipykernel~=X.X, but you'll have ipykernel X.X.X which is incompatible` after installing the dependencies. **This is normal** and caused by the `simpletransformers` library.

The **solution** to this will be to **reset the execution environment** now. Go to the menu `Runtime` > `Restart runtime` then continue on from the next section to download and process the data.

## Getting Data

#### Pulling the data from Github

We pull the data from the [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus#%E6%83%85%E6%84%9F%E8%A7%82%E7%82%B9%E8%AF%84%E8%AE%BA-%E5%80%BE%E5%90%91%E6%80%A7%E5%88%86%E6%9E%90) github repository to a `pandas` dataframe. We then display the top few rows to check if it has been downloaded correctly with `.head()`.


```
import pandas as pd
data_df = pd.read_csv('https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv', usecols=['label','review'])
data_df = data_df.rename(columns={'review': 'text', 'label': 'labels'})
data_df.head()
```

We split the dataset into a training set (80% of the samples) and a test set (20% of the samples). We also choose a fixed value for `fixed_random_state` so that this split is deterministic (always the same samples). 

We can then check the dataset properties (6,387 train negative, 3,302 train positive, 1,600 test negative and 798 test positive, an unbalanced dataset). The label **`0`** is the **`negative`** polarity class while **`1`** is the **`positive`** polarity class.


```
from sklearn.model_selection import train_test_split
fixed_random_state = 5
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=fixed_random_state)

data = [[train_df.labels.value_counts()[0], test_df.labels.value_counts()[0]], 
        [train_df.labels.value_counts()[1], test_df.labels.value_counts()[1]]]

# Prints out the dataset sizes of train and test sets per label.
pd.DataFrame(data, columns=["Train", "Test"])
```

## Training and Testing the Model

#### Set up the Training Arguments

We set up the training arguments. Here we train to 2 epochs to reduce the training time as much as possible, the BERT article on this dataset trained to 10 epochs but didn't see much gain in overall accuracy. It is also possible to split out a development set and use that to evaluate for a better model, this 10k dataset is quite small though and we are confident we can get good accuracy with just 2 epochs (we are impatient).


```
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 2,
    'train_batch_size': 128,
    'fp16': True,
    'output_dir': '/outputs/',
}
```

#### Train the Model

Once we have setup the `train_args` dictionary, the next step would be to train the model. We use the pre-trained mandarin XLNet model, [`hfl/chinese-xlnet-mid`](https://huggingface.co/hfl/chinese-xlnet-mid) from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library and model repository as the base and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the classification model with just 2 lines of code. The pre-trained mandarin model base we use is by [HFL](https://huggingface.co/hfl) with more details at this [repository](https://github.com/ymcui/Chinese-XLNet).

[XLNet](https://arxiv.org/pdf/1906.08237.pdf) is an auto-regressive language model which outputs the joint probability of a sequence of tokens based on the transformer architecture with recurrence. Although its also bigger than BERT and has a (slightly) different architecture, it's change in training objective is probably the biggest contribution. It's training objective is to predict each word in a sequence using any combination of other words in that sequence which seems to perform better on ambiguous contexts.


```
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the XLNet base cased pre-trained model.
model = ClassificationModel('xlnet', 'hfl/chinese-xlnet-mid', num_labels=2, args=train_args) 

# Train the model, there is no development or validation set for this dataset 
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.f1_score)
```

The F1-score for the model is **87.1%**.

As mentioned earlier, the class distribution (the number of **`positive`** vs **`negative`**) is not balanced (not evenly distributed), so [F1-score is a better accuracy measure](https://sebastianraschka.com/faq/docs/computing-the-f1-score.html).

Previous articles, however, published accuracy on the the test/validation set. Hence, we will also calculate the accuracy score of our model.


```
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
```

We see that the accuracy score from the model after training for 2 epochs is **91.5%** ('acc': 0.914512093411176).

> We've just trained a new state-of-the-art mandarin sentiment analysis model on the WAIMAI_10K dataset of food delivery reviews!

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(input_list)`.


```
samples = ['送错地方了，态度还不好，豆腐脑撒的哪都是，本次用餐体验很不好', # food was sent to the wrong place and the attitude was bad...
           '很不错，服务非常好，很认真'] # really quite good, service was very good, very sincere
predictions, _ = model.predict(samples)
label_dict = {0: 'negative', 1: 'positive'}
for idx, sample in enumerate(samples):
  print('{} - {}: {}'.format(idx, label_dict[predictions[idx]], sample))
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

## AI Glossary in Mandarin

Visit or star the [eugenesiow/ai-glossary-mandarin](https://github.com/eugenesiow/ai-glossary-mandarin) repository on 
Github if you need an English-to-Mandarin dictionary of AI terminology grouped topically by areas (e.g. NLP) and tasks (e.g. NER):

{{< ghbtns eugenesiow ai-glossary-mandarin "AI Glossary in Mandarin" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
