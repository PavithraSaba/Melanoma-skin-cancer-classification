from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
import random as python_random
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import EfficientNetB5, EfficientNetB3
from keras.applications.mobilenet import MobileNet
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.vis_utils import plot_model
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import h5py
import torch
from sklearn.metrics import accuracy_score

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def set_data(test):
 
 Image_size = [456,456]
 batchSize = 8
 numClasses = 2

 
                                 
 test_datagen = ImageDataGenerator()
 
 
 test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size, 
              batch_size=batchSize,                
              interpolation='bicubic',
              class_mode='categorical',
              shuffle=False
             )
 
 m = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Models/dermoscopy/big25data9632.hdf5")
 accuracy = m.evaluate(test_set)
 r=m.predict(test_set)
 k=test_set.classes
 
 return accuracy,r,k


test_path = "/content/drive/MyDrive/Colab Notebooks/newdata/test"
start=time.time();
accuracy,r,k = set_data(test_path)
end=time.time();

predIdxs = np.argmax(r,axis=-1)

 
tn, fp, fn, tp = confusion_matrix(k, predIdxs).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
total = (tp+tn+fp+fn)
acc = (tn+tp)/total
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
precision =  tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * tp) / ((2 * tp) + fp + fn)

print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity)) 
print("precision: {:.4f}".format(precision)) 
print("recall: {:.4f}".format(recall)) 
print("f1: {:.4f}".format(f1)) 

plt.figure(1)
cnf_matrix = confusion_matrix(k, predIdxs)
ax= plt.subplot()
sns.heatmap(cnf_matrix, annot=True,cmap='Blues');

print("Evaluation accuracy is:",accuracy); 
print("prediction accuracy is:",acc);
print("Time is :",(end-start)/3561);

  