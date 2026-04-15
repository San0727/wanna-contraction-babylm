# Wanna Contraction BabyLM

This repository contains the code and data for the following paper:

> Kangsan Noh and Sanghoun Song. 2026. Do BabyLMs Wanna Learn Wanna Contraction? 
> On the Learnability without Language-Specific Bias. 
> *Findings of the Association for Computational Linguistics: ACL 2026*.

## Overview

This study investigates whether the grammatical constraints on *wanna* contraction 
can be learned by BabyLMs trained under cognitively plausible conditions. 
*Wanna* contraction — the reduction of *want to* into *wanna* — is not uniformly 
permissible: it is blocked in subject-extraction contexts (e.g., *\*Who do you wanna 
go to the station tomorrow?*) but allowed in object-extraction contexts (e.g., 
*Who do you wanna take to the station tomorrow?*). This constraint has traditionally 
been cited as evidence for innate linguistic knowledge.

Using surprisal as an evaluation metric, we tested 24 BabyLMs from the 2024 BabyLM 
Challenge alongside four large-scale pretrained models (BERT and GPT-2 variants). 
Test sentences were constructed from the CHILDES corpus and divided into 
high-frequency and low-frequency datasets based on the frequency of embedded verbs.

Our results show that BabyLMs exhibit limited but non-negligible sensitivity to the 
constraint, particularly encoder-based models tested on high-frequency data. 
These findings suggest that the constraint on *wanna* contraction is learnable, 
at least partially, from linguistic input alone without invoking innate linguistic biases.
