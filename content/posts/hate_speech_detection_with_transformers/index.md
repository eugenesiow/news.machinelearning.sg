---
title: "Hate Speech Detection with Transformers"
date: 2021-01-06T08:00:00+08:00
tags: ["Natural Language Processing", "Machine Learning", "GPU", "Source Code", "PyTorch", "Hate Speech Detection", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "hate_speech.jpg"
    alt: "Hate Speech Detection on Dynabench"
    relative: true
---

> **tl;dr** A step-by-step tutorial to train a hate speech detection model to classify text containing hate speech. The 
>trained model has a BERT-based transformer architecture. 

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Hate_Speech_Detection_Dynabench.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

## Hate Speech Detection on Dynabench

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Hate_Speech_Detection_Dynabench.ipynb "Open in Colab")

Notebook to train an RoBERTa model to perform hate speech detection. The [dataset](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) used is the Dynabench Task - Dynamically Generated Hate Speech Dataset from the paper by [Vidgen et al. (2020)](https://arxiv.org/abs/2012.15761). 

> The dataset provides 40,623 examples with annotations for fine-grained labels, including a large number of challenging contrastive perturbation examples. Unusually for an abusive content dataset, it comprises 54% hateful and 46% not hateful entries.

There is no published state-of-the-art model on this dataset at this point though the task on Dynabench reports mean MER (Model Error Rate) scores. We are able to train the model to a F1 score of **86.6%** after only 1 epoch.

The notebook is structured as follows:
* Setting up the GPU Environment
* Getting Data
* Training and Testing the Model
* Using the Model (Running Inference)

#### Task Description

> Hate Speech Detection is the automated task of detecting if a piece of text contains hate speech.

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

The code below uses `pandas` to pull the dataset as a CSV file from the [official Github repository](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset). The dataset is now stored as a  dataframe, in which we can transform for use with the `simpletransformers` library to train the model. So we pull the CSV from Github and split the CSV into training and test sets by using the column `split` in the CSV file which indicates which example/sample is a training sample and which is a test sample.


```
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')
train_df = df[df['split'] == 'train'] # split the dataset by the column 'split' which labels 'train' and 'test' samples
test_df = df[df['split'] == 'test'] # split the dataset by the column 'split' which labels 'train' and 'test' samples
```

Once done we can take a look at the `head()` of the training set to check if our data has been retrieved properly.


```
train_df.head()
```

We transform the dataframe column `label` so that the labels `hate` and `nothate` are now integers `1` and `0` respectively. This input format of labels is required for our training step with the `transformers` library.


```
train_df = train_df.replace({'label': {'hate': 1, 'nothate': 0}}) # relabel the `label` column, hate is 1 and nothate is 0
test_df = test_df.replace({'label': {'hate': 1, 'nothate': 0}}) # relabel the `label` column, hate is 1 and nothate is 0
train_df.head()
```


We also rename the `label` column to `labels` as this also conforms to the input format required for the `simpletransformers` library.


```
train_df = train_df.rename(columns={'label': 'labels'})
test_df = test_df.rename(columns={'label': 'labels'})
```

We can now take a look at the train and test set sizes. We see that this dataset is quite special as the `hate` and `nothate` class sizes are actually quite close in proportion.


```
data = [[train_df.labels.value_counts()[0], test_df.labels.value_counts()[0]], 
        [train_df.labels.value_counts()[1], test_df.labels.value_counts()[1]]]
# Prints out the dataset sizes of train test and validate as per the table.
pd.DataFrame(data, columns=['Train', 'Test'])
```



# Training and Testing the Model

#### Set the Hyperparmeters

First we setup the hyperparamters. We train to 1 epoch only as want the training to complete fast. The important parameters are the `max_seq_length`, which we set to `64` and `sliding_window` to true. As we don't have a high RAM GPU on Colab we can't set the `max_seq_length` to too large a value and by using `sliding_window` we at least are able to handle longer sequences without truncation. See the [simpletransformers documentation](https://simpletransformers.ai/docs/classification-specifics/#training-with-sliding-window) for a more detailed explanation of a sliding window.


```
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 1,
    'train_batch_size': 128,
    'fp16': True,
    'output_dir': '/outputs/',
}
```

#### Train the Model

Once we have setup the hyperparemeters in the `train_args` dictionary, the next step would be to train the model. We use the RoBERTa model from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the classification model with just 2 lines of code.

RoBERTa is an optimized BERT model by Facebook Research with better performance on the masked language modeling objective that modifies key hyperparameters in BERT, including removing BERT’s next-sentence pretraining objective, and training with much larger mini-batches and learning rates. In short, its a bigger but generally better performing BERT model we can easily plug in here with the transformers library.


```
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the XLNet base cased pre-trained model.
model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=train_args) 

# Train the model, there is no development or validation set for this dataset 
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
```

The model has been trained and evaluating on the test set after training to only 1 epoch gives an accuracy of **85.9%**. We want to also evaluate the F1 score which is a better measure as the dataset is slightly imbalanced.


```
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.f1_score)
```

We see that the output F1 score from the model after training for 1 epoch is **86.6%** ('acc': 0.8661675245671503).

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(input_list)`.


```
samples = [test_df[test_df['labels'] == 0].sample(1).iloc[0]['text']] # get a random sample from the test set which is nothate
predictions, _ = model.predict(samples)
label_dict = {0: 'nothate', 1: 'hate'}
for idx, sample in enumerate(samples):
  print('{} - {}: {}'.format(idx, label_dict[predictions[idx]], sample))
```

We can also generate a `results.txt` file from the test set. The file is stored in our Colab environment. Hit the `folder` icon at the side and you can download the `results.txt` file from the file browser. You can submit this `.txt` file to [Dynabench](https://dynabench.org/tasks/5#overall) for evaluation if you wish to.


```
predictions, _ = model.predict(test_df['text'].tolist())
df = pd.DataFrame(predictions)
df.to_csv('results.txt', index=False, header=False) # saves the prediction results to a file in the colab environment
```

We can connect to Google Drive with the following code to save any files you want to persist. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the model checkpount files which are saved in the `/content/outputs/best_model/` directory to your Google Drive.


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
