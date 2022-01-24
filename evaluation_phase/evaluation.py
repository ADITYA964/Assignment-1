import os
import numpy as np 
import pandas as pd 
import os, time 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1])
  union = K.sum(y_true,[1])+K.sum(y_pred,[1])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))


test_set_1=pd.read_csv("test_dataset_1.csv")
test_set_2=pd.read_csv("test_dataset_2.csv")

data_test=pd.concat([test_set_1,test_set_2])

if "Unnamed: 0" in data_test.columns:
  data_test.drop({"Unnamed: 0"},axis="columns",inplace=True,)

target_test = data_test['_c187']  

y_test = to_categorical(target_test)

X_test = data_test.iloc[:, :-1].values

X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)    


# making the deep learning function
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(187, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    
    return model

model = model()    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[iou_coef,'accuracy',recall,precision])
model.load_weights("model.h5")

print("Confusion matrix for model on Test set")
y_pred = model.predict(X_test)
y_hat = np.argmax(y_pred, axis = 1)
print(confusion_matrix(np.argmax(y_test, axis = 1), y_hat))

label_dict = ['Non-Ectopic Beats','Superventrical Ectopic Beats','Ventricular Ectopic Beats','Fusion Beats','Unknown Beats']

def print_confusion_matrix(confusion_matrix, class_names, figsize = (15,10), fontsize=14):
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='magma')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    fig.savefig('confusion_matrix.png')
    return fig

plt.figure(figsize=(5,5))
cm = confusion_matrix(np.argmax(y_test, axis = 1), y_hat)
print_confusion_matrix(cm, label_dict)

print("Evaluation is done successfully")
