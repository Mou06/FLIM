
# This project is on FLIM inverse modeling.
 ## Autoencoder and CNN for FLI-NET Comparison

### Part 1: Training an Autoencoder

In the first part of this project, we will train an Autoencoder using two different datasets. The Autoencoder will be responsible for learning a compact representation of the input data, which will serve as the bottleneck features for subsequent tasks.

### Part 2: Training a CNN with Autoencoder Bottleneck Features

In the second part of this project, we will build and train a Convolutional Neural Network (CNN) that utilizes the bottleneck features extracted by the previously trained Autoencoder. These bottleneck features capture the essential information from the input data while reducing its dimensionality, making them valuable for various downstream tasks.

### Part 3: Comparing FLI-NET Performance with DL Model

In the final part of this project, we will compare the performance of FLI-NET, a Federated Learning-based model, with the Deep Learning (DL) model built in Part 2. We will evaluate and analyze the results to determine the effectiveness of FLI-NET in comparison to traditional DL approaches.

This repository contains the code and resources necessary to carry out each part of the project and conduct a comprehensive evaluation of the different models.
 * __DL-Model.py__ consists of an autoencoder and CNN training
 * __FLIM-generation.py__ for artificial data generation
 * __FLIMview-fitting__ is for fitting experimental data 
















