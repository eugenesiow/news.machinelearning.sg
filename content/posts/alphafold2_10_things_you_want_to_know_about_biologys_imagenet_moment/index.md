---
title: "AlphaFold2 - 10 Things You Should Know About Biology's ImageNet Moment"
date: 2020-12-07T17:01:35+08:00
tags: ["Biology", "DeepMind", "Attention", "Transformers", "ConvNet", "Deep Learning"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "alphafold2_global_distance_test.png"
    alt: "AlphaFold2 performance on the global distance test. Retrieved from the official AlphaFold2 video."
    caption: "DeepMind's AlphaFold2 prediction performance on a target by the global distance test (GDT)."
    relative: true
---

> **tl;dr** *AlphaFold2*, an AI program developed by [DeepMind](https://deepmind.com/about), completely blew away the 
> competition in a protein-structure prediction competition, CASP, so much so, the [co-founder](http://moult.ibbr.umd.edu/) 
> of the competition has commented that *"in some sense"* the [50-year old](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) 
> problem, is solved.   

### 1. What is protein folding?

A protein's function is determined by its structure, its 3D shape. The process, 
[protein folding](https://en.wikipedia.org/wiki/Protein_folding), is how a protein chain acquires this 
3D shape and structure. The chain can easily contain hundreds of amino acids.

{{< youtube KpedmJdrTpY >}}

### 2. What is CASP and what were the results?

[CASP](https://predictioncenter.org/casp14/index.cgi) (Critical Assessment of Structure Prediction) is a biennial blind 
protein-structure prediction competition where computational biologists try to predict the structure of several proteins 
whose structure has already been determined experimentally but not yet publicly released. 
This year marked the 14th edition of the competition.

{{< figure src="alphafold2_results.png" title="Figure 1. X-axis: Group's number in CASP14. Y-axis: Z-scores of their predictions. AlphaFold2, shows a huge score improvement from the second best group, BAKER. The figure was obtained from the official CASP14 webpage on 7 December 2020." >}}

AlphaFold2 was the huge winner of the competition, beating over 100 other teams. The above figure shows the sum of 
Z-scores representing the different participating groups. The Z-score is the difference of a sample’s value with respect 
to the population mean, divided by the standard deviation. A high value represents a large deviation from the mean. 
Groups with larger Z-scores, like AlphaFold2 (Group 427), whose average Z-score was around 2.5 when considering all 
targets, and rose to 3.8 on the hardest ones, performed much better than the average.

The actual performance (not relative to the other teams) is quite impressive too and has already been covered in great 
detail [elsewhere](https://www.blopig.com/blog/2020/12/casp14-what-google-deepminds-alphafold-2-really-achieved-and-what-it-means-for-protein-folding-biology-and-bioinformatics/)
and debated upon widely based on the little we know without the actual code and paper released. More on this when we 
tackle the debate on whether the problem is really solved.

AlphaFold2 as its name suggests is the second iteration of DeepMind's AlphaFold, which debuted in the previous iteration
of CASP in 2018. It is a huge improvement on that model, which was already state-of-the-art. Again, we don't know the 
exact details till the paper is out, but more on this when we talk about the algorithm below. 

{{< tweet 1333445882759376898 >}}

### 3. What does this have to do with ImageNet?

{{< tweet 1333436710303264772 >}}

DeepMind research scientist Oriol Vinyals termed the breakthrough as "Biology’s ImageNet moment". The AI/ML community 
often recognises the work of the image dataset, ImageNet, and the ImageNet Large Scale Visual Recognition Challenge 
(ILSVRC) as setting the stage for landmark computer vision and ML achievements such as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
in 2012 (which was 41% better than the next best competitor) and 2015’s [ResNet](https://arxiv.org/abs/1512.03385) and
the "Deep Learning" era in ML research. NLP also had its [ImageNet moment](http://jalammar.github.io/illustrated-bert/) 
just a while back.

### 4. What are the implications of this breakthrough?

In short, this speeds up scientists work in understanding diseases, finding treatments, determining which drugs might
be applied. On the other hand, some have brought up the possibility of its misuse, whether such a tool might be [somehow 
weaponized](https://news.ycombinator.com/item?id=25307718). 

In terms of drug discovery, drugs work by attaching a protein in a particular place, thereby altering or disabling its 
function. Knowing a protein’s shape may allow scientists in the future to identify such binding sites and make it easier 
to synthesise new and effective therapeutics.  There is some debate though, on whether 
this is actually a [rate-limiting factor](https://blogs.sciencemag.org/pipeline/archives/2019/09/25/whats-crucial-and-what-isnt) 
in current drug discovery pipelines. What we do know is the [anecdote](https://www.nytimes.com/2020/11/30/technology/deepmind-ai-protein-folding.html) 
from Andrei Lupas, an evolutionary biologist at the Max Planck Institute. 

> He had spent a decade trying to figure out the 
  shape of one protein in a tiny bacteria-like organism called an archaeon, with AlphaFold2, he found the answer in 
  half an hour. Wow!

### 5. Is the problem really solved?

Stephen Curry, a Professor of Structural Biology at Imperial College London seems to think otherwise and posted [an 
article](http://occamstypewriter.org/scurry/2020/12/02/no-deepmind-has-not-solved-protein-folding/) on his blog that 
acknowledges that Alphafold2 "will certainly help to advance biology" but "despite some of the claims being made, 
we are not at the point where this AI tool can be used for drug discovery."

"For delivering reliable insights into protein chemistry or drug design" the root-mean-squared difference (RMSD) in 
atomic positions between the prediction and the actual structure needs to go down from 1.6 Å (0.16 nm) to within a 
margin of around 0.3Å.

Some practitioners seem to [think otherwise](https://news.ycombinator.com/item?id=25306954) though, with some accusations 
of "goal-post shifting". 

> "As a practitioner, what I want to be able to do is pull a sequence that I've retrieved from DNA, drop it into a computer, 
and get the structure out. Intuitive insight can FOLLOW from those results."

We will also expect DeepMind to be working out "corner cases" and improving the model as they decide how they want to 
use or release the model.

{{< tweet 1333383769861054464 >}}

### 6. What was the algorithm used? How was the model trained?

Until the peer-reviewed paper is released and (if) the code is released, we don't know much yet. As always, 
[Yannic Kilcher](https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew)
does a great job of explaining the original AlphaFold paper in the video below.

{{< youtube id="B9PL__gVxLI?start=860" autoplay="false" >}}


Architecture-wise, for AlphaFold2, we don't know. We do know that AlphaFold2 uses some sort of attention mechanism, 
maybe [transformers](https://www.youtube.com/watch?v=B9PL__gVxLI&t=2625s) replacing the ConvNet. It also seems they now
[fit over not just the fragements but the whole shape of the protein](https://explainthispaper.com/ai-solving-protein-folding/).

As mentioned in the [DeepMind blog post](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) 
they trained this system on ~170,000 protein structures from the [protein data bank](https://www.rcsb.org/) together with 
[large databases](https://www.uniprot.org/) containing protein sequences of unknown structure. 
The experimental setup for training was 16 x TPUs (roughly equivalent to 100-200 GPUs) run over a few weeks.

### 7. Will there be source code?

Until DeepMind decides how or if it wants to release whole or part of its code, we can only wait for the paper to be 
released.

DeepMind did release part of the original [AlphaFold](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13) 
code (although in practice, researchers have pointed out "good luck generating the input features for something other than the 
CASP13 proteins") and there have been open source implementations of the proposed model. 
[MiniFold](https://github.com/EricAlcaide/MiniFold) and [AlphaFold Pytorch](https://github.com/Urinx/alphafold_pytorch) 
are some notable efforts.

### 8. How big was DeepMind's funding and team and how does this compare to the Big Pharmas?

People have put the size of DeepMind at around [1000 people](https://www.cnbc.com/2020/06/05/google-deepmind-alphago-buzz-dissipates.html) 
and perhaps 15-30 or so research teams. The video published by DeepMind showed 20 or so researchers on this particular team.
The team was said to be cross-disciplinary and also includes researchers who have done PhDs or postdocs in Biology-related fields. 

Big Pharma companies are said to spend at least hundreds of millions on R&D. They don't usually get to hire the best 
minds in AI/ML though for much more specific, targeted research. It does make sense that this breakthrough comes from a 
pretty well-funded AI research laboratory rather than Big Pharma or academia. 

### 9. What does this mean for AI/ML?

As reported in the [tech media](https://www.vox.com/future-perfect/22045713/ai-artificial-intelligence-deepmind-protein-folding), 
it seems apparent that DeepMind has finally gone from "playing video games to addressing scientific problems with 
real-world significance — problems that can be life-or-death".

The buzz that this breakthrough has created around the scientific and AI/ML communities is also just wonderful. 
It's nice to end a quite terrible 2020 on a high note and it's nice to see an AI/ML lab committed to deep research on 
difficult problems finding more success.  

### 10. What is the link to the ML community in Singapore?

DeepMind co-founder + CEO, [game developer](https://en.wikipedia.org/wiki/Theme_Park_(video_game)) 
and [chess champion](https://www.chessgames.com/perl/chessplayer?pid=57778) 
Demis Hassabis has some [Singaporean](https://en.wikipedia.org/wiki/Demis_Hassabis#Early_life_and_education) heritage.

In a recent [interview with the BBC](https://www.bbc.com/news/technology-55157940), Demis, mentions that he took 
some early inspiration from, *[Foldit](https://fold.it/)*, a crowdsourcing puzzle game. Gamers who played it 
were actually trying to turn the protein into a particular shape, which led to the discovery of a few important 
structures for real proteins. So also a shout out to gamers in the community!
