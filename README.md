
# This project is on FLIM inverse modeling.
 ## Autoencoder and CNN for FLI-NET Comparison

### Part 1: Training an Autoencoder

In the first part of this project, we will train an Autoencoder using two different datasets. The Autoencoder will be responsible for learning a compact representation of the input data, which will serve as the bottleneck features for subsequent tasks.

### Part 2: Training a CNN with Autoencoder Bottleneck Features

In the second part of this project, we will build and train a Convolutional Neural Network (CNN) that utilizes the bottleneck features extracted by the previously trained Autoencoder. These bottleneck features capture the essential information from the input data while reducing its dimensionality, making them valuable for various downstream tasks.

![Image_for_inverse_modeling_v2](https://github.com/Mou06/FLIM/assets/69230384/d0423cfa-f95e-43e9-a15d-61572509e772)


### Part 3: Comparing FLI-NET Performance with DL Model

In the final part of this project, we will compare the performance of FLI-NET, a Federated Learning-based model, with the Deep Learning (DL) model built in Part 2. We will evaluate and analyze the results to determine the effectiveness of FLI-NET in comparison to traditional DL approaches.

### Part 4: Steps to generate artificial data
$$I(t) = IRF * Σ [ a_i * e^{(-t/τ_i)} ] + P(n)$$
Where:
- 'I(t)' represents decay trace
- 'IRF' is system response function (represents as a Gaussian function)
- 'a_i' represents abundance. Here we take a_1 and a_2. The summation of a_1 and a_2 is always 1.
- 'τ_i' represents as lifetime values. Here we take τ_1 and τ_2 values. The range of τ_1 is (0,1] and τ_2 is (1,5].
-  'a_1', 'a_2', 'τ_1', 'τ_2' - lifetime parameters
-  'P(n)' represents as Poisson noise.


This repository contains the code and resources necessary to carry out each part of the project and conduct a comprehensive evaluation of the different models.
 * __DL-Model.py__ consists of an autoencoder and CNN training
 * __FLIM-generation.py__ for artificial data generation
 * __FLIMview-fitting.py__ is for fitting experimental data
 * __FLIM_convolve_new_realIRFandnoise.csv___ is FLIM decay trace convolution with IRF (system response function)
 * __FLIM_noiseonly_new_realIRFandnoise.csv__ consists of only noise
 * __FLIM_without_convolve_realIRFandnoise.csv__ consists of decay trace without convolution
 * __210310_PosA01_200f.sdt__ experimental data. It can be fiited with __FLIMview-fitting.py__. It consists of 2 channels; Red and Green channel. Red and Green channels should be fitted separately.
 * __Morphotox_lifetimevalue_607and455.xlsx__ lifetime parameter values real data
 * __Mean_decaytrace_R and G channel.xlsx___ decay traces for real value.
















