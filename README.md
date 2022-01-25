# Assignment-1

## Description of folders and files

```tree
   |-- preprocess_phase
   |   |-- Dockerfile 
   |   |-- ECG_time_series.png   
   |   |-- pod.yaml  
   |   |-- requirements2.txt   
   |   |-- spark_preprocess.py   
   |-- train_phase     
   |   |-- snapshots
   |   |   |-- best_weight001.h5
   |   |   |-- best_weight005.h5
   |   |   |-- best_weight006.h5
   |   |-- Dockerfile
   |   |-- app.py
   |   |-- infinite.py
   |   |-- logs.csv
   |   |-- metric.png
   |   |-- model.h5
   |   |-- requirements.txt
   |   |-- train_pod.yaml
   |-- evaluation_phase
   |   |-- Dockerfile
   |   |-- confusion_matrix.png
   |   |-- eval_pod.yaml
   |   |-- evaluation.py
   |   |-- infinite.py
   |   |-- model.h5
   |   |-- requirements.txt
   |-- Notebook.ipynb
   |-- LICENSE
   |-- README.md
```   
## Opensource Dataset details

ECG Heartbeat Categorization Dataset link :- https://www.kaggle.com/shayanfazeli/heartbeat

This dataset is composed of two collections of heartbeat signals derived from two famous datasets in heartbeat classification, the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. The number of samples in both collections is large enough for training a deep neural network.

This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing some of the capabilities of transfer learning on it. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals are preprocessed and segmented, with each segment corresponding to a heartbeat.

Each ECG time series sample is categorized into one of the 5 classes mentioned below:-

- class 1 : Non-Ectopic Beats
- class 2 : Superventrical Ectopic Beats
- class 3 : Ventricular Ectopic Beats
- class 4 : Fusion Beats
- class 5 : Unknown Beats

## Objective

The problem statement is to develop a deep learning model using 1 - Dimensional Convolutional Networks to do heartbeat classification from 5 types of beats using ECG time series data samples.

## Execution of Assignment-1 using Kubernetes and Minikube

1. Clone this private repository.
```shell
git clone https://[Github-Username]:[Personal-Access-Token-associated-with-Github-Username]@github.com/ADITYA964/Assignment-1.git
```
2. Download Minikube installer at https://storage.googleapis.com/minikube/releases/latest/minikube-installer.exe
   or execute below command.
```shell
choco install minikube   # For Windows operating system

brew install minikube    # For Mac or Linux operating systems
```
