---
title: "Top 10 Natural Language Processing Code Repositories from Singapore"
date: 2020-12-12T08:00:00+08:00
tags: ["Github", "Deep Learning", "Singapore", "Machine Learning", "Source Code", "PyTorch", "TensorFlow", "Natural Language Processing"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: ""
    alt: ""
    caption: ""
---

> **tl;dr** We feature 10 of the top Natural Language Processing (NLP) code repositories on Github from Singapore. 
> The ranking is decided based on the total stars (stargazer count) of the repositories.

## 10. A Multilayer Convolutional Encoder-Decoder Neural Network for Grammatical Error Correction

{{< figure src="nlp_repo_grammar.png" title="Architecture of the multilayer convolutional model with seven encoder and seven decoder layers. Retrieved from the official paper." >}}

Code and model files for the paper: "A Multilayer Convolutional Encoder-Decoder Neural Network for Grammatical Error 
Correction" (Published at AAAI-18).

The authors improve the automatic correction of grammatical, orthographic, and collocation errors in text using a 
multilayer convolutional encoder-decoder neural network.


|          |                                                                                      |
|----------|--------------------------------------------------------------------------------------|
|Repository|[nusnlp/mlconvgec2018](https://github.com/nusnlp/mlconvgec2018)                       |
|License   |[GNU General Public License v3.0](https://api.github.com/licenses/gpl-3.0)            |
|Author    |[NUS NLP Group](http://www.comp.nus.edu.sg/~nlp) ([nusnlp](https://github.com/nusnlp))|
|Vocation  |[NLP Group](http://www.comp.nus.edu.sg/~nlp) at National University of Singapore      |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  160|   73|          1|


## 9. Project Insight: NLP as a Service

{{< figure src="nlp_repo_insight.png" title="Project Insight: NLP as a Service. Retrieved from Github." >}}

Project Insight is an NLP as a service project with a frontend UI (**`streamlit`**) and backend server (**`FastApi`**) 
serving transformers models on various downstream NLP task.

The downstream NLP tasks covered are:

* News Classification
* Entity Recognition
* Sentiment Analysis
* Summarization

|          |                                                                                                                                                            |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Repository|[abhimishra91/insight](https://github.com/abhimishra91/insight)                                                                                             |
|License   |[GNU General Public License v3.0](https://api.github.com/licenses/gpl-3.0)                                                                                  |
|Author    |[Abhishek Kumar Mishra](https://www.linkedin.com/in/abhishek-kumar-mishra-116b2554/) ([abhimishra91](https://github.com/abhimishra91))                      |
|Vocation  |Operations Innovation Lead at IHS Markit|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  225|   29|          4|


## 8. An Unsupervised Neural Attention Model for Aspect Extraction

{{< figure src="nlp_repo_abae.png" title="An example of an Attention-based Aspect Extraction (ABAE) structure. Retrieved from the official paper." >}}

Code for the ACL2017 paper "An Unsupervised Neural Attention Model for Aspect Extraction".

Aspect extraction is one of the key tasks in sentiment analysis. It aims to extract entity aspects on which opinions 
have been expressed. For example, in the sentence "The beef was tender and melted in my mouth", the aspect term is "beef".
Experimental results on real-life datasets demonstrate that our approach discovers more meaningful and coherent aspects, 
and substantially outperforms baseline methods on several evaluation tasks. 
Aspect-Based Sentiment analysis (ABSA) can then be performed on the set of aspects in the downstream task.

|          |                                                                                                 |
|----------|-------------------------------------------------------------------------------------------------|
|Repository|[ruidan/Unsupervised-Aspect-Extraction](https://github.com/ruidan/Unsupervised-Aspect-Extraction)|
|License   |[Apache License 2.0](https://api.github.com/licenses/apache-2.0)                                 |
|Author    |[Ruidan He](https://sites.google.com/view/ruidan) ([ruidan](https://github.com/ruidan))          |
|Vocation  |NLP scientist at Alibaba DAMO Academy. Ph.D. from NUS.                                     |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  265|  107|         16|


## 7. ✍🏻 gpt2-client: Easy-to-use TensorFlow Wrapper for GPT-2 🤖 📝 

{{< figure src="nlp_repo_gpt2_client.png" title="Exploring GPT-2 models in less than five lines of code. Retrieved from Github." >}}

GPT-2 is a Natural Language Processing model developed by OpenAI for text generation. The model has 4 versions - 
117M, 345M, 774M, and 1558M - that differ in terms of the amount of training data fed to it and the number of parameters 
they contain.

gpt2-client is a wrapper around the original gpt-2 repository that features the same functionality but with more 
accessiblity, comprehensibility, and utilty. You can play around with all four GPT-2 models in less than five lines of code.

|          |                                                                                      |
|----------|--------------------------------------------------------------------------------------|
|Repository|[rish-16/gpt2client](https://github.com/rish-16/gpt2client)                           |
|License   |[MIT License](https://api.github.com/licenses/mit)                                    |
|Author    |[Rishabh Anand](https://rish-16.github.io/) ([rish-16](https://github.com/rish-16))   |
|Vocation  |CS + ML Research at Advanced Robotics Center @ NUS|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  291|   59|          5|


## 6. A hierarchical CNN based model to detect Big Five personality traits

{{< figure src="nlp_repo_big5.png" title="Architecture of the implemented network from the paper. Retrieved from the official paper." >}}

This code implements the model discussed in [Deep Learning-Based Document Modeling for Personality Detection from Text](http://sentic.net/deep-learning-based-personality-detection.pdf) 
for detection of Big-Five personality traits, namely:

- Extroversion
- Neuroticism
- Agreeableness
- Conscientiousness
- Openness

|          |                                                                                     |
|----------|-------------------------------------------------------------------------------------|
|Repository|[SenticNet/personality-detection](https://github.com/SenticNet/personality-detection)|
|License   |[MIT License](https://api.github.com/licenses/mit)                                   |
|Author    |[SenticNet](http://sentic.net) ([SenticNet](https://github.com/SenticNet))           |
|Vocation  |Computational Intelligence Lab (CIL) in the School of Computer Science and Engineering (SCSE) of Nanyang Technological University (NTU).|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  315|  126|         22|


## 5. SymSpell: Very Fast Spell Checking in Python

{{< figure src="nlp_repo_symspell.png" title="Performance of SymSpell (C# version) vs other edit distance/spell check algorithms. Retrieved from Github." >}}

A Python port of SymSpell, a [1 million times faster](https://medium.com/@wolfgarbe/fast-approximate-string-matching-with-large-edit-distances-in-big-data-2015-9174a0968c0b) 
spelling correction & fuzzy search through Symmetric Delete spelling correction algorithm.

The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and dictionary 
lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster (than the standard approach with 
deletes + transposes + replaces + inserts) and language independent.

|          |                                                             |
|----------|-------------------------------------------------------------|
|Repository|[mammothb/symspellpy](https://github.com/mammothb/symspellpy)|
|License   |[MIT License](https://api.github.com/licenses/mit)           |
|Author    |[mmb L](None) ([mammothb](https://github.com/mammothb))      |
|Vocation  |Don't Know                                                |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  391|   75|         10|


## 4. Textstat: Python Package to Calculate Readability Statistics of Text

{{< figure src="nlp_repo_textstat.png" title="Example code to use the Textstat library." >}}

Textstat is an easy to use library to calculate statistics from text. It helps determine readability, complexity, and grade level.

It supports various statistics including: Flesch Reading Ease Score, Flesch-Kincaid Grade Level, Fog Scale (Gunning FOG Formula),
SMOG Index, Automated Readability Index, Coleman-Liau Index, Linsear Write Formula and the Dale-Chall Readability Score.

|          |                                                                                                                                                 |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------|
|Repository|[shivam5992/textstat](https://github.com/shivam5992/textstat)                                                                                    |
|License   |[MIT License](https://api.github.com/licenses/mit)                                                                                               |
|Author    |[Shivam Bansal](http://www.shivambansal.com) ([shivam5992](https://github.com/shivam5992))                                                              |
|Vocation  |Data Scientist \| Natural Language Processing + Machine Learning Enthusiast \| Data Stories + Visuals \| Programmer + Coder \| at H2O.ai |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  554|  107|         18|


## 3. Sentiment analysis on tweets using Naive Bayes, SVM, CNN, LSTM, etc.

{{< figure src="nlp_repo_twitter_sentiment.png" title="Flowchart of the majority voting ensemble used. Retrieved from the official report." >}}

Sentiment classification on twitter dataset. The authors use a number of machine learning and deep learning methods to 
perform sentiment analysis (Naive Bayes, SVM, CNN, LSTM, etc.). The authors finally use a majority vote ensemble method 
with 5 of our best models to achieve the classification accuracy of 83.58% on kaggle public leaderboard.

|          |                                                                                                 |
|----------|-------------------------------------------------------------------------------------------------|
|Repository|[abdulfatir/twitter-sentiment-analysis](https://github.com/abdulfatir/twitter-sentiment-analysis)|
|License   |[MIT License](https://api.github.com/licenses/mit)                                               |
|Author    |[Abdul Fatir](http://abdulfatir.com) ([abdulfatir](https://github.com/abdulfatir))               |
|Vocation  |CS PhD Student at National University of Singapore                                       |


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  |  918|  442|         22|


## 2. RASA NLU for Chinese

{{< figure src="nlp_repo_rasa_nlu_chinese.png" title="Natural Language Understanding outputs from the RASA NLU Chinese model. Retrieved from Github." >}}

A fork from the RASA (contextual AI assistant/chatbot) NLU repository. Focused on the Natural Language Understanding (NLU) 
task (中文自然语言理解). 
Turn a chinese natural language sentence/utterance into structured data.

|          |                                                                       |
|----------|-----------------------------------------------------------------------|
|Repository|[crownpku/Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)      |
|License   |[Apache License 2.0](https://api.github.com/licenses/apache-2.0)       |
|Author    |[Guan Wang](www.crownpku.com) ([crownpku](https://github.com/crownpku))|
|Vocation  	        |[Senior Data Scientist, VP](https://www.linkedin.com/in/crownpku/) at [Swiss Re](https://www.linkedin.com/company/swiss-re/)  	|


|Language|Stars|Forks|Open Issues|
|--------|----:|----:|----------:|
|Python  | 1,103|  358|         74|


## 1. Chinese Named Entity Recognition and Relation Extraction

{{< figure src="nlp_repo_information_extraction_chinese.jfif" title="Visualization of 3x3, 7x7 and 15x15 receptive fields produced by 1, 2 and 4 dilated convolutions by the IDCNN model." >}}

An __NLP__ repository including state-of-art deep learning methods for various tasks in chinese/mandarin language (中文): 
named entity recognition (__NER__/实体识别), relation extraction (__RE__/关系提取) and word segmentation.

|   |   |
|-	|-	|
|Repository  	    |[crownpku/Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese)  	|
|License            |Not Specified   |
|Author  	        |[Guan Wang](http://www.crownpku.com/) ([crownpku](https://github.com/crownpku))  	|
|Vocation  	        |[Senior Data Scientist, VP](https://www.linkedin.com/in/crownpku/) at [Swiss Re](https://www.linkedin.com/company/swiss-re/)  	|

|Language   |Stars      |Forks      |Open Issues    |
|-	        |-	        |-	        |-              |
|Python     |1,692      |748        |102            |


