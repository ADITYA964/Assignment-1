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

  
train_set_1=pd.read_csv("train_dataset_1.csv")
train_set_2=pd.read_csv("train_dataset_2.csv")
val_set_1=pd.read_csv("val_dataset_1.csv")
val_set_2=pd.read_csv("val_dataset_2.csv")


train_dataset = pd.concat([train_set_1,train_set_2])
val_dataset=pd.concat([val_set_1,val_set_2])

if "Unnamed: 0" in train_dataset.columns:
  train_dataset.drop({"Unnamed: 0"},axis="columns",inplace=True,)

if "Unnamed: 0" in val_dataset.columns:
  val_dataset.drop({"Unnamed: 0"},axis="columns",inplace=True,)


target_train = train_dataset['_c187']
target_val = val_dataset['_c187']
y_train = to_categorical(target_train)
y_val = to_categorical(target_val)
X_train = train_dataset.iloc[:, :-1].values
X_val = val_dataset.iloc[:, :-1].values
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_val = X_val.reshape(len(X_val), X_val.shape[1], 1)                          

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
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[iou_coef,'accuracy',recall,precision])
    return model

model = model()
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=1e-3,
                                                 min_lr=1e-7)

target_dir = 'snapshots'

if not os.path.exists(target_dir):
  os.mkdir(target_dir)

file_path = 'snapshots/best_weight{epoch:03d}.h5'
checkpoints = tf.keras.callbacks.ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')

cbks = [ checkpoints, reduce_lr]

logger = CSVLogger('logs.csv', append=True)
with tf.device('/GPU:0'):
    his = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_data=(X_val, y_val), callbacks=[logger,cbks])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(40, 10))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss","recall","precision"]):
    ax[i].plot(model.history.history[metric][:])
    ax[i].plot(model.history.history["val_" + metric][:])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

fig.savefig('metric.png')   


MODEL_FILE =os.environ["MODEL_FILE"]
from joblib import dump, load
model.save(MODEL_FILE)
print("Model Saved successfully")

print(os.listdir())

print("Done")