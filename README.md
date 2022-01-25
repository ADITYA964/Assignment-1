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
