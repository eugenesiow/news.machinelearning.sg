---
title: "AI 2020 Roundup: AlphaFold, Tennis and Causality"
date: 2020-12-28T16:00:00+08:00
tags: ["Artificial Intelligence", "Roundup", "Computer Vision", "Deep Learning", "Machine Learning"]
author: "Eugene"
showToc: true
TocOpen: false
draft: false
cover:
    image: "tennis.png"
    relative: true
    alt: "Vid2Player: Controllable Video Sprites that Behave and Appear like Professional Tennis Players. Retrieved from Youtube."
---

> Part 1 of our roundup featuring the latest and greatest AI advancements and directions from 2020. Today we cover AlphaFold's 
>biology breakthrough, controllable tennis videos and causality touted as the next step for AI.

## AlphaFold's Winning Protein-structure Prediction Competition CASP By a Mile is AI's Biggest Breakthrough this Year

*AlphaFold2*, an AI program developed by [DeepMind](https://deepmind.com/about), completely blew away the 
competition in a protein-structure prediction competition, CASP, so much so, the [co-founder](http://moult.ibbr.umd.edu/) 
of the competition has commented that *"in some sense"* the [50-year old](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) 
problem, is solved.

See our article on [AlphaFold2 - 10 Things You Should Know About Biology's ImageNet Moment](https://news.machinelearning.sg/posts/alphafold2_10_things_you_want_to_know_about_biologys_imagenet_moment/)
for a quick rundown on what AlphaFold has achieved.

## Researchers from Stanford Created A Controllable Synthetic Video Version of Wimbledon Tennis Matches

Ever wanted to climb into the television and play as Nadal or Federer for the Wimbledon Finals? Do you enjoy how realistic 
the graphics and gameplay from Fifa, Madden or NBA Live is but how far behind Virtua Tennis is?

{{< figure src="virtua_tennis.jpg" title="Virtua Tennis, a series of tennis simulation video games started in 1999 by Sega." >}}

[Researchers from Stanford](cs.stanford.edu/~haotianz/research/vid2player/) combined a model of player and tennis ball 
trajectories, pose estimation, and unpaired image-to-image translation to create a realistic controllable tennis match 
video between any players you wish!

{{< figure src="pose.png" title="Rendering players in their correct pose. Retrieved from the official paper." >}}

Now the next step is to see if that fits in a game... Imagine the possibilities. 
A controllable or gameplay-driven [Red Alert cutscene](https://www.youtube.com/watch?v=Vi98bQTQOUQ) anyone?

## Causality the Fix for Deep Learning?

AI godfather and Turing award recipient Yoshua Bengio has [said that](https://www.wired.com/story/ai-pioneer-algorithms-understand-why/) deep learning needs to be fixed. 
He believes that until AI can go beyond pattern recognition and learn more about cause and effect, we won't be delivering a
true AI revolution. 

His argument is that most ML applications utilise statistical techniques to explore correlations between variables. 
This requires that experimental conditions remain the same and that the trained ML system is applied on the same kind of 
data as the training data (domain specific/dependent).  

For example when a doctor diagnoses a patient and recommends a particular course of treatment. This is not something that 
correlation-based ML systems were designed for. Once a change in policy is made, the relationship between the input 
and output variables will differ from the training data, reducing the accuracy of the system.

Causal inference explicitly addresses this issue and Bengio and other pioneers in the field like Judea Pearl believe 
that this will be a powerful way to allow ML systems to generalize better, be more robust and result in more explainable 
decision-making.

Bengio and his students have published [some initial work](https://arxiv.org/pdf/1901.10912.pdf) in that direction.
