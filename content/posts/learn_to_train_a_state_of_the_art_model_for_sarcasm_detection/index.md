---
title: "Learn to Train a State-of-the-Art Model for Sarcasm Detection"
date: 2020-12-22T09:00:00+08:00
tags: ["Natural Language Processing", "Deep Learning", "Machine Learning", "GPU", "Source Code", "PyTorch", "Sarcasm Detection", "Jupyter Notebook", "Colab"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "sarcasm_detection.png"
    alt: "Sarcasm Detection on Tweets"
---

> **tl;dr** A step-by-step tutorial to train a state-of-the-art model to detect sarcasm 🙄 from tweets with a 
> free Jupyter Notebook in the cloud. 

## Sooo Impressive Sarcasm Detection Model on Tweets

Recently Venture Beat published (and [Communications of the ACM](https://cacm.acm.org/careers/248837-ai-researchers-made-a-sarcasm-detection-model-and-its-sooo-impressive/fulltext) referenced)  
a news article titled "[AI researchers made a sarcasm detection model and it’s sooo impressive](https://venturebeat.com/2020/11/18/ai-researchers-made-a-sarcasm-detection-model-and-its-soo-impressive/)"
which detailed how researchers from China had come up with a "sarcasm detection AI" that "achieved state-of-the-art 
performance on a dataset drawn from Twitter".
 
What is interesting about the AI model was that it used multimodal learning to combine text and imagery from the tweets 
to do this sarcasm detection, "since both are often needed to understand whether a person is being sarcastic".

{{< figure src="sarcasm_detection_photos.jpg" title="Heatmap Visualization of Attention Focus on Images of Sarcastic Tweets. Retrieved from the official paper." >}}

The premise and findings like the above figure were interesting and kudos to the team behind the paper. 

In this step-by-step practical, we will learn to train a model that has better performance (F1-score) than this 
state-of-the-art model on the same dataset while using only the text from the tweets.

## Practical Machine Learning - Learn Step-by-Step to Train a Model

A great way to learn is by going step-by-step through the process of training and evaluating the model.

In this step-by-step practical, we will learn to train a model that has better performance (F1-score) than the 
state-of-the-art model.

Hit the **`Open in Colab`** button below to launch a Jupyter Notebook in the cloud with a step-by-step walkthrough.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sarcasm_Detection_Twitter.ipynb "Open in Colab")

Continue on if you prefer reading the code here.

## Sarcasm Detection on Twitter Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sarcasm_Detection_Twitter.ipynb "Open in Colab")

Notebook to train a BERT-based (RoBERTa) model to perform sarcasm detection. The dataset used is a collection of (more than 20,000) tweets with binary labels: **`not sarcastic`** or **`sarcastic`** from the paper by [Cai et al. (2019)](https://www.aclweb.org/anthology/P19-1239/). The trained model beats the state-of-the-art at this time (Dec 2020). The current state-of-the-art model on this dataset by [Pan et al. (2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.124/) uses additional information of features from hash-tags and and the image posted along with the tweet (multi-modal sarcasm detection) whereas this model uses just the textual features.

The notebook is structured as follows:
* Setting up the GPU Environment
* Getting Data
* Training and Testing the Model
* Using the Model (Running Inference)

#### Task Description

> The goal of Sarcasm Detection is to determine whether a sentence is sarcastic or non-sarcastic. Sarcasm is a type of phenomenon with specific perlocutionary effects on the hearer, such as to break their pattern of expectation. Consequently, correct understanding of sarcasm often requires a deep understanding of multiple sources of information, including the utterance, the conversational context, and, frequently some real world facts.

Source: [Attentional Multi-Reading Sarcasm Detection](https://arxiv.org/abs/1809.03051)

## Setting up the GPU Environment

#### Ensure we have a GPU runtime

If you're running this notebook in Google Colab, select `Runtime` > `Change Runtime Type` from the menubar. Ensure that `GPU` is selected as the `Hardware accelerator`. This will allow us to use the GPU to train the model subsequently.

#### Install Dependencies and Restart Runtime


```shell script
!pip install -q transformers
!pip install -q simpletransformers
```

You might see the error `ERROR: google-colab X.X.X has requirement ipykernel~=X.X, but you'll have ipykernel X.X.X which is incompatible` after installing the dependencies. **This is normal** and caused by the `simpletransformers` library.

The **solution** to this will be to **reset the execution environment** now. Go to the menu `Runtime` > `Restart runtime` then continue on from the next section to download and process the data.

## Getting Data

Here are the functions that will allow us to download the dataset from the [Github data repository](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection) of the paper by [Cai et al. (2019)](https://www.aclweb.org/anthology/P19-1239/). The function will also process the dataset so we can read it into [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html#).


```python
import csv
import urllib.request


def filtered(sentence):
  """Filter function that indication if sentence should be filtered.

  Filtering function that is adapted from the original, more verbose, 
  pre-processing script from the Cai et al. (2019) paper:
  https://github.com/headacheboy/data-of-multimodal-sarcasm-detection/blob/master/codes/loadData.py

  Args:
      sentence: A string of the sentence to be filtered.
      
  Returns:
      A boolean value (True or False) that indicates if a sentence should be 
      filtered of based on the criterea by Cai et al. (2019).
  """
  words = sentence.split()
  filter = ['sarcasm', 'sarcastic', 'reposting', '<url>', 'joke', 'humour', 'humor', 'jokes', 'irony', 'ironic', 'exgag']
  for filtered_word in filter:
    if filtered_word in words:
      return True
  return False


def download_and_clean(url, output_file, text_index, labels_index, to_filter=False):
  """Download and pre-process the paper's tweet dataset.

  Downloads the dataset from a url (github repository) of the Cai et al. (2019)
  and processes it so that it is a properly formatted CSV file that can be read
  by pandas and follows exactly the Cai et al. (2019) and Pan et al. (2020) 
  papers.

  Args:
      url: the url location of the dataset to download as a string.
      output_file: the output path of the CSV file to write to as a string.
      text_index: the index of the text column (the tweet text) as an int.
      labels_index: the index of the label column (the sarcasm label) as an int.
      to_filter: a boolean to indicate if this dataset should be filtered as
        per the papers preprocessing rules.
  """
  with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['text', 'labels'])
    file = urllib.request.urlopen(url)
    for line in file:
      decoded_line = line.decode('utf-8')
      row = eval(decoded_line)
      if not to_filter or not filtered(row[text_index]):
        csv_writer.writerow([row[text_index], row[labels_index]])
```

Now we use the above functions to download and pre-process the train, test and validation datasets from the paper's Github data repository. The output file are written to the local storage of the notebook as `train.csv`, `test.csv` and `validate.csv`.


```python
download_and_clean('https://raw.githubusercontent.com/headacheboy/data-of-multimodal-sarcasm-detection/master/text/train.txt', 'train.csv', 1, 2, to_filter=True)
download_and_clean('https://raw.githubusercontent.com/headacheboy/data-of-multimodal-sarcasm-detection/master/text/test2.txt', 'test.csv', 1, 3)
download_and_clean('https://raw.githubusercontent.com/headacheboy/data-of-multimodal-sarcasm-detection/master/text/valid2.txt', 'validate.csv', 1, 3)
```

Now we use pandas to read in the well-formatted `train.csv`, `test.csv` and `validate.csv` files into dataframes. We also take a look at the first few rows of the training set with the `.head()` function to check if our CSV files are loaded properly.


```python
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
validate_df = pd.read_csv('validate.csv')
train_df.head()
```

Next, we compare if our dataset size, after the pre-processing, is exactly the same as those reported in both the papers. **`0`** is the **`not sarcastic`** class while **`1`** is the **`sarcastic`** class.

The paper reports the following dataset class sizes for train, test and validate (used as the dev) sets.

|Label  	|Train  	|Test  	|Validate  	|
|-	|-	|-	|-	|
|0  	|11174  	|1450  	|1451  	|
|1  	|8642  	|959  	|959  	|



```python
data = [[train_df.labels.value_counts()[0], test_df.labels.value_counts()[0], validate_df.labels.value_counts()[0]], 
        [train_df.labels.value_counts()[1], test_df.labels.value_counts()[1], validate_df.labels.value_counts()[1]]]
# Prints out the dataset sizes of train test and validate as per the table.
pd.DataFrame(data, columns=["Train", "Test", "Validate"])
```

We are now confident that we have the exact dataset as reported in both the papers, we can go on to train our model to do sarcasm detection.

## Training and Testing the Model

#### Set the Hyperparmeters

First we setup the hyperparamters, using the hyperparemeters specified in the  Pan et al. (2020) paper whenever possible. The comparison of hyperparameters are shown in the table below. The major difference is we only train 1 epoch instead of 8 as we want the training to be fast.

|Parameter  	    |Ours  	    |Paper  	|
|-	                |-	        |-	        |
|Epochs  	        |1  	    |8  	    |
|Batch Size  	    |32  	    |32  	    |
|Seq Length  	    |75  	    |75  	    |
|Learning Rate      |5e-5       |5e-5       |
|Weight decay       |1e-2       |1e-2       |
|Warmup rate        |0.2        |0.2        |
|Gradient Clipping  |1.0        |1.0        |


```python
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': False,
    'max_seq_length': 75,
    'learning_rate': 0.00005,
    'weight_decay': 0.01,
    'warmup_ratio': 0.2,
    'max_grad_norm': 1.0,
    'num_train_epochs': 1,
    'train_batch_size': 32,
    'save_model_every_epoch': False,
    'save_steps': 4000,
    'fp16': True,
    'output_dir': '/outputs/',
    'evaluate_during_training': True,
}
```

#### Train the Model

Once we have setup the hyperparemeters in the `train_args` dictionary, the next step would be to train the model. We use the [`roberta-base` model](https://huggingface.co/roberta-base) from the awesome [Hugging Face Transformers](https://github.com/huggingface/transformers) library and use the [Simple Transformers library](https://simpletransformers.ai/docs/classification-models/) on top of it to make it so we can train the classification model with just 2 lines of code.

[RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) is an optimized BERT model by Facebook Research with better performance on the masked language modeling objective that modifies key hyperparameters in BERT, including removing BERT's next-sentence pretraining objective, and training with much larger mini-batches and learning rates. In short, its a bigger but generally better performing BERT model we can easily plug in here with the transformers library.


```python
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn

logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the RoBERTa base pre-trained model.
model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=train_args) 

# Train the model, use the validation set as the development set as per the paper.
# When training to 1 epoch this is not that essential, however, if you decide to 
# train more and configure early stopping, do check out the simple transformers
# documentation: https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
model.train_model(train_df, eval_df=validate_df)

# Evaluate the model in terms of accuracy score
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
```

We see that the output accuracy from the model after training for 1 epoch is **93.7%** ('acc': 0.9369032793690328).

#### Evaluate the Model (F1-score)

Now we want to calculate the F1-score for the model. 

Since the class distribution (the number of **`sacarstic`** vs **`not sarcastic`**) is not balanced, [F1-score is a better accuracy measure](https://sebastianraschka.com/faq/docs/computing-the-f1-score.html). We calculate the F1-score of the model on the test set below.


```python
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.f1_score)
```

The F1-score is **92.2%** ('acc': 0.9224489795918368) is **9.4 points** better than the state-of-the-art results reported in the Pan et al. (2020) paper at **82.9%** using just the textual features with RoBERTa instead of BERT. 

> We've just trained a new state-of-the-art sarcasm detection model from tweet text!

## Using the Model (Running Inference)

Running the model to do some predictions/inference is as simple as calling `model.predict(input_list)`.


```python
samples = ['hell yeah !  # funny # sleepwell # dreamon # fail',
           'i could enter the olympics ! ;) rt <user> : ',
           'we ’ re excited to hold a q & a session with <user> tomorrow courtesy of <user> ! submit your questions by using # askabluejay ! # wt2017']
predictions, _ = model.predict(samples)
label_dict = {0: 'not sarcastic', 1: 'sarcastic'}
for idx, sample in enumerate(samples):
  print('{}: {}, {}'.format(idx, sample, label_dict[predictions[idx]]))
```

We can connect to Google Drive with the following code to save any files you want to persist. You can also click the `Files` icon on the left panel and click `Mount Drive` to mount your Google Drive.

The root of your Google Drive will be mounted to `/content/drive/My Drive/`. If you have problems mounting the drive, you can check out this [tutorial](https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166).


```
from google.colab import drive
drive.mount('/content/drive/')
```

You can move the `train.csv` file from our local directory to your Google Drive. You can do the same for the model checkpount files which are saved in the `/content/outputs/best_model/` directory.


```
import shutil
shutil.move('/content/train.csv', "/content/drive/My Drive/train.csv")
```

#### Discussion

With an accuracy of >92%, have we solved sarcasm detection? Probably not. We know we have trained a classification model that is great on this dataset with only text content as input, however, if we go back to the task definition, we think that a correct understanding of sarcasm _often requires a deep understanding of multiple sources of information, including the utterance, the conversational context, and, frequently some real world facts_. 

It's certainly possible (and quite trivial) to pick out counterexamples of tweets with little context that could be classified as sarcastic. It is also possible to study these results and this dataset in greater detail (confusion matrix, eyeballing), but it will probably lead to limited insights. Check out more notebooks and check back as we update the repo with more practical ML in NLP and sarcasm detection as things develop on the SOTA frontier.

## More Such Notebooks

Visit or star the [eugenesiow/practical-ml](https://github.com/eugenesiow/practical-ml) repository on Github for more such notebooks:

{{< ghbtns eugenesiow practical-ml "Practical Machine Learning" >}}

## Alternatives to Colab

Here are some alternatives to Google Colab to train models or run Jupyter Notebooks in the cloud:

- [Google Colab vs Paperspace Gradient](https://news.machinelearning.sg/posts/google_colab_vs_paperspace_gradient/)
