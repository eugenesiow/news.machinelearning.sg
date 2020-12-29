---
title: "AI 2020 Roundup: Talent in Demand, GNNs are Hot and a New Macbook M1 for AI"
date: 2020-12-29T16:00:00+08:00
tags: ["Artificial Intelligence", "Roundup", "Computer Vision", "Deep Learning", "Machine Learning", "Graph Neural Networks"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "non_euclidean_data.png"
    alt: "Examples of non-euclidean data where GNNs are being applied on. Retrieved from Flawnson Tong's Article on Medium."
    caption: "Examples of non-euclidean data where GNNs are being applied on. Retrieved from Medium."
---

> Part 2 of our roundup featuring the latest and greatest AI advancements and directions from 2020. We cover the global 
>shortage of AI talent, Graph Neural Networks being the hotest research area and a new Macbook for machine learning.

## AI Talent in Shortage, in High Demand byt not Pandemic Proof

As companies around the world started to embrace AI in 2017 to 2019 and aggressively hire AI talent, [demand outstripped
supply](https://www.reuters.com/article/us-usa-economy-artificialintelligence/as-companies-embrace-ai-its-a-job-seekers-market-idUSKCN1MP10D) 
from universities and institutions of higher learning.

{{< figure src="indeed_ai.png" title="AI-related job postings and search statistics from Indeed.com. Retrieved from Reuters." >}}

However, while the market was still hot in 2020, companies started reigning in spending due to the pandemic and public 
job postings on LinkedIn that mention a deep learning framework, which were initially on a strong early 2020 ramp up, 
took a hit due to COVID-19 since February 2020.

{{< figure src="twitter_drop_jobs.png" title="Public job postings on LinkedIn that mention a deep learning framework. Courtesy of François Chollet. Retrieved from Twitter." >}}

## Are Graph Neural Networks the Hottest Research Area?

The top machine learning and AI conferences this year, NeurIPS, ICML, ICLR were awash with papers on Graph Neural Networks.

{{< figure src="gnns.png" title="Publications and citations on Graph Neural Networks. Retrieved from Microsoft Academic." >}}

To understand this surge in popularity, we should perhaps consider GNNs as a subset or a means for Geometric Deep Learning.
A vast majority of deep learning is performed on Euclidean data. This includes datatypes in 1D and 2D domains. 
We however exist in a 3D world and there is an argument thar our data should reflect that. 
As the community searches for new use cases, new models and new breakthroughs - non-euclidean data and GNNs have gained 
popularity.

This introduction to [Geometric Deep Learning](https://flawnsontong.medium.com/what-is-geometric-deep-learning-b2adb662d91d) 
by Flawnson Tong is a pretty good read.

If you prefer pouring over hundreds of papers in the area (or a few survey papers), this [list of resources](https://github.com/thunlp/GNNPapers#content)
on GNNs is as good as any.

## Benchmarks of Machine Learning Workloads on the new Macbook M1 Chip Are Looking Strong  

Apple's new Macbook "armed" (pun-intended) with their new M1 chips have so much going for them. Great performance on both native (ARM)
and virtualized (x86) workloads, crazy (for laptop) battery life and fanless awesomeness (for the Macbook Air).

{{< figure src="m1.jpg" title="The M1 chips key features. Neural Engine, 8-core GPU and ML accelerators! Retrieved from Hexus.net." >}}

Early machine learning [performance benchmarks](https://blog.roboflow.com/apple-m1-for-machine-learning/) are also showing
excellent performance. 3.6x better vs their Intel CPU and AMD Radeon GPU counterparts on fitting an object detection model
with Apple's CreateML. 

What we really want to (and should soon) see is how it performs against Nvidia's amazing new Ampere cards.  
