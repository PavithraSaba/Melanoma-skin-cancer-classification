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
import math
from sklearn.metrics import accuracy_score

def set_data(test,modelpath):
 
 Image_size = [188,188]
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
 
 m = tf.keras.models.load_model(modelpath)
 accuracy = m.evaluate(test_set,batch_size=batchSize)
 r=m.predict(test_set,batch_size=batchSize)
 k=test_set.classes
 
 return accuracy,r,k,test_set


splits = 5
accuracyT = np.zeros(splits)
sensitivityT = np.zeros(splits)
specificityT = np.zeros(splits)
precisionT = np.zeros(splits)
recallT = np.zeros(splits)
f1T = np.zeros(splits)

splits = 5
accuracyT = np.zeros(splits)
sensitivityT = np.zeros(splits)
specificityT = np.zeros(splits)
precisionT = np.zeros(splits)
recallT = np.zeros(splits)
f1T = np.zeros(splits)
tT = np.zeros(splits)

batchSize = 8
Image_size = [188,188]
for fold in range(splits):
  test_path = '/content/drive/MyDrive/Colab Notebooks/kfolddata/fold'+str(fold+1)+'/test'
  modelpath = '/content/drive/MyDrive/Colab Notebooks/Models/clinical/f100/fold'+str(fold+1)+'/best100.hdf5'
  start=time.time();
  accuracy,r,k,test_set = set_data(test_path,modelpath)
  end=time.time();
  tT[fold]=(end-start)/5.0;
  accuracyT[fold] = accuracy[1]
  
  predIdxs = np.argmax(r,axis=-1)
  #actual = np.argmax(test_set,axis=-1)
  
 
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
  
  sensitivityT[fold] = sensitivity
  specificityT[fold] = specificity
  precisionT[fold] = precision
  recallT[fold] = recall
  f1T[fold] = f1

  plt.figure(fold)
  cnf_matrix = confusion_matrix(k, predIdxs)
  ax= plt.subplot()
  sns.heatmap(cnf_matrix, annot=True,cmap='Blues');
  
print(accuracyT)
print("Average accuracy is:", sum(accuracyT)/5.0)
print("Average sensitivity is:",sum(sensitivityT)/5.0)
print("Average specificity is:",sum(specificityT)/5.0)
print("Average precision is:",sum(precisionT)/5.0)
print("Average recall is:",sum(recallT)/5.0)
print("Average f1-score is:",sum(f1T)/5.0)
print("Time is :",sum(tT)/5.0);
