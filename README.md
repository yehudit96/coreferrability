# Paraphrasing vs Coreferring: Two Sides of the Same Coin

## Introduction

This code used in the paper "Paraphrasing vs Coreferring: Two Sides of the Same Coin" by Yehudit Meged, Avi Caciularu, Vered Shwartz, Ido Dagan. EMNLP Finding 2020.
(https://arxiv.org/abs/2004.14979)

A random forest model for classifing and ranking for paraphrases identification taks.

# Instructions

This research is consisit of 4 stages:

## Tweets pair collection

This stage code is in 

## Features creation

  The features are consisint of 5 feture groups:
  _Named Entity Coverage_ is in the NER directory, _cross-document coreference resolution_ is in coreference directory, 
  _connected componenet_ and _clique_ are in _graph_ directory and the chirps features are derived from the chirps resource


## Paraphrases annotation

Tha paraphrases annotation code in MTAnnotation directory

## Model training

The model training code is in classification directory
