import os
import numpy as np 
import pandas as pd 
import time 

# Spark Session, Pipeline, Functions, and Metrics
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
# from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("mitbih_train.csv", header=None)
M = df.values
X = M[:, :-1]
y = M[:, -1].astype(int)

C0 = np.argwhere(y == 0).flatten()
C1 = np.argwhere(y == 1).flatten()
C2 = np.argwhere(y == 2).flatten()
C3 = np.argwhere(y == 3).flatten()
C4 = np.argwhere(y == 4).flatten()

x = np.arange(0, 187)*8/1000


fig, ax = plt.subplots(figsize=(15,10))

ax.plot(x,X[C0, :][0], marker='*', markersize=2, color='teal',linestyle='dashed', label="Non-Ectopic Beats")
ax.plot(x,X[C1, :][0], marker='*', markersize=5, color='purple', label="Superventrical Ectopic Beats")
ax.plot(x,X[C2, :][0], marker='*', markersize=2, color='red',linestyle=':', label="Ventricular Ectopic Beats")
ax.plot(x,X[C3, :][0], marker='*', markersize=2, color='blue',linestyle='dashed', label="Fusion Beats")
ax.plot(x,X[C4, :][0], marker='*', markersize=2, color='black',linestyle='-.', label="Unknown Beats")

ax.legend()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('1-beat ECG for every category');

fig.savefig('ECG_time_series.png')

#Create PySpark SparkSession
conf = SparkConf().setAppName('Spark Deep Learning processing phase').setMaster('local[3]')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

# Load Data to Spark Dataframe
df_train = sql_context.read.csv('mitbih_train.csv',
                    header=False,
                    inferSchema=True)

df_val= sql_context.read.csv('mitbih_test.csv',
                    header=False,
                    inferSchema=True)


 
df_val.printSchema()
df_val.show(100)    


data_1 = df_train[df_train[187] == 1]
data_2 = df_train[df_train[187] == 2]
data_3 = df_train[df_train[187] == 3]
data_4 = df_train[df_train[187] == 4]


N_SAMP = os.environ["N_SAMP"]
N_SAMP=int(N_SAMP)
RAND = os.environ["RAND"]
RAND=int(RAND)

from sklearn.utils import resample
data_1_resample = resample(data_1.toPandas(), n_samples=N_SAMP, 
                           random_state=RAND, replace=True)
data_2_resample = resample(data_2.toPandas(), n_samples=N_SAMP, 
                           random_state=RAND, replace=True)
data_3_resample = resample(data_3.toPandas(), n_samples=N_SAMP, 
                           random_state=RAND, replace=True)
data_4_resample = resample(data_4.toPandas(), n_samples=N_SAMP, 
                           random_state=RAND, replace=True)
data_0 = df_train[df_train[187] == 0].toPandas().sample(n=N_SAMP, random_state=RAND)



train_dataset = pd.concat([data_0, data_1_resample, data_2_resample, data_3_resample, 
                          data_4_resample])

val_dataset = df_val.toPandas()

VER=os.environ["VER"]  

train_name="train_dataset_" + str(VER) + ".csv"
val_name="val_dataset_" + str(VER) + ".csv"

train_dataset.to_csv(train_name)
val_dataset.to_csv(val_name)
