## Introduction

# Secure-AI

This repository contains submission for the 2020 Facebook developers circle community challenge. We will be discussing about privacy preserving machine learning tools built on top of pytorch alongside tutorials for technical implementations.

# Overview

The exponential growth of data is overwhelming and data nowadays is basically ubiquitous. Considering the ground-breaking achievements and breakthrough in Artificial intelligence such as deep learning due to the rise and access to compute instances in order to run large training jobs of deep neural networks on the cloud, now is the time to take a look into how AI systems are being developed and how privacy-preserving these systems are. Deep learning is a form of machine  learning which is an exciting domain with a very huge potential, it is fast-growing with wide-array of cutting edge applications in self-driving cars, cancer detection, NLP and speech recognition. The heart of an optimized model is 'data', your model is as good as the data it's being trained on and as ML is hungry for data, we currently do not have a data lake/bank that contains accurate set of data that can utilized for training effective ML models, as such data are often collected from users of a product which often contain sensitive information of these users thereby invading privacy. What if we could train models and perform statistical analysis on distributed data without compromising privacy? 

  Data privacy has been a major concern for tech companies and individuals since the GDPR legislation in may 2018. Therefore, it's imperative for data scientists, machine learning engineers researchers and tech companies to be aware of user's privacy while going about collecting data in the ML pipeline either for analysis, training or some other business oriented reasons. 

This lesson we will tackle lowering some of the potential implications and challenges posed by training Deep neural network on distributed data and AI as well as principles for building responsible AI that is secure to avoid harming others amd protecting privacy of data.

# Pre-requisites
Knowledge of the following are required in order to follow along with the tutorial

- Python
- Pytorch 

# System requirements 
You need to have the following software installed on your machine

- Anaconda/conda
- Pytorch 1.4 or later 
- Syft 0.2.9

# Installation 
Steps
1. Install anaconda 
2. Add conda to your system PATH
3. Check the [INSTALLATION.md here](https://github.com/Boluwatifeh/Secure-AI/blob/master/INSTALLATION.md) on a comprehensive guide on how to install and run the tutorials.

# Quickstart

```bash
$ git clone https://github.com/Boluwatifeh/Secure-AI.git
$ cd tutorials 
$ jupyter notebook
``` 
Run the code cells in the note book 


# Contributing 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. if you're familiar with the PySyft framework and would love to contribute by writing beginner-friendly tutorials you can always send a PR. i'd also highly recommend contributions to the open source Openmined community which is a group of developers working on making the world more privacy preserving, check out the openmined repo here https://github.com/OpenMined. Check out the CONTRIBUTING.md [over here](https://github.com/Boluwatifeh/Secure-AI/blob/master/CONTRIBUTING.md)




## Tutorial: Getting started with federated learning using PyTorch and PySyft

![.](https://preview.redd.it/wh66axb4a2151.png?width=600&format=png&auto=webp&s=276a41a59a5e1c4c0185a0e74623efe6f41d4dbb) 

What is federated learning and why is it important ?
Federated learning is a machine learning technique used in training models on distributed data in a decentralized manner, this tutorial would cover how to leverage pytorch and pysyft for federated learning. At the end, readers would train a federated neural network on the official MNIST dataset. 

## Outline
- Introduction 
- Setup
- Steps
- Conclusion
- Learn more

## Introduction

Federated learning is a machine learning technique that enables training of machine learning models on decentralized data by sending copies of a model to edge devices where data lives in order to perform training and preserve privacy of users data. Brendan Mcmahan of Google pioneered Federated learning and it eliminates moving of data to a central server for training purposes. The idea Federated learning proposes is to leave data at edge devices also known as workers, these workers might be mobile devices, IOT devices etc

## Federated learning workflow 
- Edge devices receives a copy of a global model from a central server.
- The model is being trained locally on the data residing on the edge devices 
- The global model weights are updated during training on each worker 
- A local copy is sent back to the central server 
- The server receives various updated model and aggregate the updates thereby improving the global model and also preserving privacy of data in which it was being trained on.

![.](https://1.bp.blogspot.com/-K65Ed68KGXk/WOa9jaRWC6I/AAAAAAAABsM/gglycD_anuQSp-i67fxER1FOlVTulvV2gCLcB/w1200-h630-p-k-no-nu/FederatedLearning_FinalFiles_Flow%2BChart1.png)

Federated learning workflow

## Use cases 

- **Next word prediction**: Federated learning is used to improve word prediction in mobile devices without uploading users data to the cloud for training. Google Gboard implements FL by using on-device data to train and improve the Gboard next word prediction model for it's users, [watch this video to better understand how google uses federated learning at scale](https://www.youtube.com/watch?v=gbRJPa9d-VU&list=PLDz716gGFpo22O7ePVr4TjNHHWEdhF8Ve&index=71). You can also read this [online comic from Google AI](https://federated.withgoogle.com/) to get a better grasp of federated learning. 

- **Voice recognition**: FL is also used in voice recognition technologies, an example is in apple Siri and recently Google introduced the audio recognition technology on Google assistant to train and better improve users experience with the Google assistant. Watch a [demonstration of FL on audio recording for speech systems here](https://www.youtube.com/watch?v=oqmcvxzbRJs). In every of these use cases, the data doesn't leave the edge devices thereby keeping the data private, safe and secure while still improving these technologies and making products smarter over time.

## Getting started

To run the tutorial and follow along, you need to be familiar with python, basics of pytorch, jupyter notebooks and basics of pysyft for remote execution (this can be found in the part 1 notebook of the tutorials directory on github). Head over to this github repository and make sure your machine meets the following requirements 
- Anaconda/conda 4.8+
- Python 3.6
- Pytorch 1.4 or later 
- Syft 0.2.9

## Quick Start
```bash
$ Conda create -n my_env python=3.6 
```
```bash
$ Conda activate my_env 
```
```bash
$ Pip install syft
```
```bash
$ git clone https://github.com/boluwatifeh/Secure-AI.git 
```
```bash
$ Cd tutorials
```
```bash
$ Jupyter notebook
```
Over this tutorial, we're going to look into one of the features of pysyft by training a neutral network on MNIST for federated learning.

## What is pysyft?

Pysyft is an open source python library for computing on data you do not own or have access to. It is built on top of the popular deep learning framework (pytorch and tensorflow) and allows for computations on decentralized data, it also provides room for privacy preserving tools such as secure multiple party computation and differential privacy. Jump over to the github repository to learn more. 

## Installation 

Pip install syft

## Imports

Import torch
Import syft as sy
