# Code for [*Optimizer Benchmarking Needs to Account for Hyperparameter Tuning*](https://icml.cc/virtual/2020/poster/6589) at ICML 2020

In this we provide our PyTorch code for our ICML 2020 paper. The pdf can be found [here](https://arxiv.org/abs/1910.11758).
In this repository we provide the background code required for running our experiments in HYPerparameter Aware Optimizer Benchmarking Protocol [(HYPAOBP)](https://TODO). We base our code on a slightly older version of **DeepOBS**. It is quite likely that multiple bug-fixes have been made on the main repository.


The readme of the original DeepOBS is available [here](original_readme.md)

## Installation

We recommend that the `spec-file.txt` in HYPAOBP is used to recreate the environment.  

## Changes

The modify **DeepOBS** in the following ways:

- Added `Early Stopping` to stop after validation loss plateaus
- Added IMDb LSTM problem, and Character RNN (Tolstoi) in PyTorch version. 
- Added wall clock time to the list of saved attributes.
- Added SGD and Adam with poly-learning rate decay schedules.
- Added utility codes for plotting and computing various stats presented in our paper. 
- Modified the `LearningRateScheduleRunner` to provide a signal to the SGD and Adam decay methods to decay the learning rate. 
- 