# Assignment-1

## Description of folders and files

```tree
   Assignment-1
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

### Method 1

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
3. Change directory to enter repositories main folder.
```shell
cd ./Assignment-1/
```
4. Change directory to enter preprocess_phase subfolder.
```shell
cd ./preprocess_phase/
```
5. Initialize the configurations of Minikube cluster.
```shell
minikube config set memory 8192

minikube config set cpus 4
```
6. Activate the Minikube cluster so that we can execute Kubernetes commands.
```shell
minikube start --vm-driver=docker
```
7. Check whether the Minikube cluster's master node is running or not.
```shell
kubectl get all
```
8. Execute the YAML file using kubectl to create two containers within the pod for preprocessing the dataset using Spark.
```shell
kubectl create -f ./pod.yaml
```
9. Check whether the pod is in Running state and it would take some time to reach that state.
```shell
kubectl get pods
```
10. View the contents of the first container after preprocessing the dataset.
```shell
kubectl exec spark-container -c spark-container-one -- ls
```
11. View the contents of the second container after preprocessing the dataset.
```shell
kubectl exec spark-container -c spark-container-two -- ls
```
12. Copy the preprocessed datasets from the first container.
```shell
kubectl cp spark-container:/app/train_dataset_1.csv train_dataset_1.csv -c spark-container-one

kubectl cp spark-container:/app/val_dataset_1.csv val_dataset_1.csv -c spark-container-one
```
13. Copy the preprocessed datasets from the second container.
```shell
kubectl cp spark-container:/app/train_dataset_2.csv train_dataset_2.csv -c spark-container-two

kubectl cp spark-container:/app/val_dataset_2.csv val_dataset_2.csv -c spark-container-two
```
14. After copying the preprocessed files, we stop the Pod from running using below command.
```shell
kubectl delete -f ./pod.yaml
```
15. Now we need to move preprocessed dataset files into train_phase subfolder for the purpose of training the model.
```shell
cd ..

mv ./preprocess_phase/train_dataset_1.csv ./train_phase/

mv ./preprocess_phase/train_dataset_2.csv ./train_phase/

mv ./preprocess_phase/val_dataset_1.csv ./train_phase/

mv ./preprocess_phase/val_dataset_2.csv ./train_phase/

cd ./train_phase/
```
16. Execute below command to start creating container for training deep learning model within the pod.
```shell
kubectl create -f ./train_pod.yaml
```
17. Check whether the pod is in Running state and it would take some time to reach that state.
```shell
kubectl get pods
```
18. View the contents of files present in the pod after training the deep learning model.
```shell
kubectl exec train-pod -- ls
```
19. Copy the results and weights of model from the container running inside the pod.
```shell
kubectl cp train-pod:/app/logs.csv logs.csv -c train-container

kubectl cp train-pod:/app/metric.png metric.png -c train-container

kubectl cp train-pod:/app/model.h5 model.h5 -c train-container

kubectl cp train-pod:/app/snapshots snapshots -c train-container
```
20. The purpose gets served for training the model so now we stop the pod by executing below command.
```shell
kubectl delete -f ./train_pod.yaml
```
21. Change directory and shift the weights of model to evaluation_phase subfolder to vaidate the performance of model.
```shell
cd ..

mv ./train_phase/model.h5 ./evaluation_phase/

cd ./evaluation_phase/
```
22. To evaluate the model , execute below command to run pod for it.
```shell
kubectl create -f ./eval_pod.yaml
```
23. Check whether the pod is in Running state.
```shell
kubectl get pods
```
24. View the contents of files after evaluating the model.
```shell
kubectl exec evaluation-pod -- ls
```
25. Copy the result of evaluated model from the container running inside pod.
```shell
kubectl cp evaluation-pod:/app/confusion_matrix.png confusion_matrix.png -c evaluation-container
```
26. After validating the model , we stop the pod by executing below command.
```shell
kubectl delete -f ./eval_pod.yaml
```
### Method 2

1. Login to the azure portal.
```shell
az login
```
2. Create a resource group in the region eastus.
```shell
az group create -l eastus -n [Resource-Group-Name]
```
3. Create an instance of Azure Container Registry.
```shell
az acr create --name [Registry-Name] `
--resource-group [Resource-Group-Name] `
--sku basic --admin-enabled true
```
4. Authenticate the Registry.
```shell
az acr login -n [Registry-Name] --expose-token
```
5. Create an instance of Kubernetes cluster using Azure Kubernetes Service.
```shell
az aks create --resource-group [Resource-Group-Name] `
--name [Kubernetes-Cluster-Name] `
--node-count 2 `
--generate-ssh-keys `
--attach-acr [Registry-Name] `
--load-balancer-sku basic `
--enable-cluster-autoscaler `
--min-count 1 `
--max-count 5 `
--vm-set-type VirtualMachineScaleSets `
--enable-addons monitoring,http_application_routing
```
6. Authenticate the credentials to use Kubernetes cluster. 
```shell
az aks get-credentials --resource-group [Resource-Group-Name] --name [Kubernetes-Cluster-Name]
```
7. Check whether the Kuberenetes cluster is running or not.
```shell
kubectl get all
```
8. Create a service principal.
```shell
$ACR_NAME="[Registry-Name]"

$SERVICE_PRINCIPAL_NAME="[Service-Principal-Name]"

$ACR_REGISTRY_ID=$(az acr show --name $ACR_NAME --query "id" --output tsv)

$PASSWORD=$(az ad sp create-for-rbac --name $SERVICE_PRINCIPAL_NAME --scopes $ACR_REGISTRY_ID --role acrpull --query "password" --output tsv)

$USER_NAME=$(az ad sp list --display-name $SERVICE_PRINCIPAL_NAME --query "[].appId" --output tsv)

echo "Service principal ID: $USER_NAME" 

echo "Service principal password: $PASSWORD"
```
9. Create a secret to retrieve docker containers from Azure Container Registry.
```shell
kubectl create secret docker-registry [Secret-Name] --docker-server=[Registry-Name].azurecr.io --docker-username=$USER_NAME --docker-password=$PASSWORD
```
10. Build docker images.
```shell 
cd ./Assignment-1/preprocess_phase/

az acr build --registry [Registry-Name] --resource-group [Resource-Group-Name] --image spark_preprocess_container_one:latest -f Dockerfile .

az acr build --registry [Registry-Name] --resource-group [Resource-Group-Name] --image spark_preprocess_container_two:latest -f Dockerfile .

cd ..

cd ./train_phase/

az acr build --registry [Registry-Name] --resource-group [Resource-Group-Name] --image train:latest -f Dockerfile .

cd ..

cd ./evaluation_phase/

az acr build --registry [Registry-Name] --resource-group [Resource-Group-Name] --image evaluation:latest -f Dockerfile .
```

11. Replace YAML file codes with below ones:-
```shell
--------------------------------------------------
./Assignment-1/preprocess_phase/pod.yaml
--------------------------------------------------
apiVersion: v1
kind: Pod
metadata:
  name: spark-container
spec:
  
  containers:
  - name: spark-container-one
    image: [Registry-Name].azurecr.io/spark_preprocess_container_one:latest
    command: ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]
  - name: spark-container-two
    image: [Registry-Name].azurecr.io/spark_preprocess_container_two:latest
    command: ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]

  imagePullSecrets:
    - name: [Secret-Name]  
  restartPolicy: Never  

--------------------------------------------------
./Assignment-1/train_phase/train_pod.yaml
--------------------------------------------------
apiVersion: v1
kind: Pod
metadata:
  name: train-pod
spec:
  containers:
  - name: train-container
    image: [Registry-Name].azurecr.io/train:latest
    command: ["python3", "app.py"]
    command: ["python3", "infinite.py"]
  
  imagePullSecrets:
    - name: [Secret-Name]    
  restartPolicy: Never  
  
--------------------------------------------------
./Assignment-1/evaluation_phase/eval_pod.yaml
--------------------------------------------------
apiVersion: v1
kind: Pod
metadata:
  name: evaluation-pod
spec:
  containers:
  - name: evaluation-container
    image: [Registry-Name].azurecr.io/evaluation:latest
    command: ["python3", "evaluation.py"]
    command: ["python3", "infinite.py"]
  
  imagePullSecrets:
    - name: [Secret-Name]    
  restartPolicy: Never
```  
12. Execute the YAML file using kubectl to create two containers within the pod for preprocessing the dataset using Spark.
```shell
kubectl create -f ./pod.yaml
```
13. Check whether the pod is in Running state and it would take some time to reach that state.
```shell
kubectl get pods
```
14. View the contents of the first container after preprocessing the dataset.
```shell
kubectl exec spark-container -c spark-container-one -- ls
```
15. View the contents of the second container after preprocessing the dataset.
```shell
kubectl exec spark-container -c spark-container-two -- ls
```
16. Copy the preprocessed datasets from the first container.
```shell
kubectl cp spark-container:/app/train_dataset_1.csv train_dataset_1.csv -c spark-container-one

kubectl cp spark-container:/app/val_dataset_1.csv val_dataset_1.csv -c spark-container-one
```
17. Copy the preprocessed datasets from the second container.
```shell
kubectl cp spark-container:/app/train_dataset_2.csv train_dataset_2.csv -c spark-container-two

kubectl cp spark-container:/app/val_dataset_2.csv val_dataset_2.csv -c spark-container-two
```
18. After copying the preprocessed files, we stop the Pod from running using below command.
```shell
kubectl delete -f ./pod.yaml
```
19. Now we need to move preprocessed dataset files into train_phase subfolder for the purpose of training the model.
```shell
cd ..

mv ./preprocess_phase/train_dataset_1.csv ./train_phase/

mv ./preprocess_phase/train_dataset_2.csv ./train_phase/

mv ./preprocess_phase/val_dataset_1.csv ./train_phase/

mv ./preprocess_phase/val_dataset_2.csv ./train_phase/

cd ./train_phase/
```
20. Execute below command to start creating container for training deep learning model within the pod.
```shell
kubectl create -f ./train_pod.yaml
```
21. Check whether the pod is in Running state and it would take some time to reach that state.
```shell
kubectl get pods
```
22. View the contents of files present in the pod after training the deep learning model.
```shell
kubectl exec train-pod -- ls
```
23. Copy the results and weights of model from the container running inside the pod.
```shell
kubectl cp train-pod:/app/logs.csv logs.csv -c train-container

kubectl cp train-pod:/app/metric.png metric.png -c train-container

kubectl cp train-pod:/app/model.h5 model.h5 -c train-container

kubectl cp train-pod:/app/snapshots snapshots -c train-container
```
24. The purpose gets served for training the model so now we stop the pod by executing below command.
```shell
kubectl delete -f ./train_pod.yaml
```
25. Change directory and shift the weights of model to evaluation_phase subfolder to vaidate the performance of model.
```shell
cd ..

mv ./train_phase/model.h5 ./evaluation_phase/

cd ./evaluation_phase/
```
26. To evaluate the model , execute below command to run pod for it.
```shell
kubectl create -f ./eval_pod.yaml
```
27. Check whether the pod is in Running state.
```shell
kubectl get pods
```
28. View the contents of files after evaluating the model.
```shell
kubectl exec evaluation-pod -- ls
```
29. Copy the result of evaluated model from the container running inside pod.
```shell
kubectl cp evaluation-pod:/app/confusion_matrix.png confusion_matrix.png -c evaluation-container
```
30. After validating the model , we stop the pod by executing below command.
```shell
kubectl delete -f ./eval_pod.yaml  
```
