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
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB5, EfficientNetB0
from keras.applications.mobilenet import MobileNet
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.vis_utils import plot_model
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from keras import models
import h5py
from keras import metrics


import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def set_data(train,test, batchSize, image_size):
 np.random.seed(1234)
 python_random.seed(1234)
 tf.random.set_seed(1234)

 
 Image_size = [image_size,image_size]

 train_datagen= ImageDataGenerator(validation_split=0.3,rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=0.5
                                   )

 test_datagen = ImageDataGenerator()

 
 train_set = train_datagen.flow_from_directory(
                train,
                target_size=Image_size,
                batch_size=batchSize, 
                color_mode="rgb",             
                interpolation='bicubic',
                class_mode='categorical'
                )

 test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size,
              color_mode = "rgb", interpolation='bicubic',
              class_mode='categorical'
             )
 validation_set = train_datagen.flow_from_directory(
    train, # same directory as training data
    target_size=Image_size,color_mode = "rgb",interpolation='bicubic',
    batch_size=batchSize)
 return train_set, test_set, validation_set;

 def plot_hist(hist):
    plt.figure(3)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    
    plt.figure(4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def create_model():
  inputs = layers.Input(shape=(188, 188, 3))
  x = inputs
  basemodel = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
  basemodel.trainable = False
  basemodel = unfreeze_model(basemodel, -200)
  x = basemodel.output
  se = layers.GlobalAveragePooling2D(name="ch_pool")(x)
  se = layers.Reshape((1,1,1536))(se)
  se = layers.Dense(64,activation="swish",kernel_initializer='he_normal', use_bias=False)(se)
  se = layers.Dense(1536,activation="sigmoid",kernel_initializer='he_normal', use_bias=False)(se)
  x = layers.Multiply()([se, x])
  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)
  
  outputs = layers.Dense(2, activation="softmax", name="pred")(x)
  model = tf.keras.Model(basemodel.input, outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
  return model

def unfreeze_model(model, num_of_layers):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[num_of_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

splits = 5
accuracy = np.zeros(splits)
accuracyT = np.zeros(splits)
batchSize = 8
epoches =100;
image_size = 188;
for fold in range(splits):
  train_path = '/content/drive/MyDrive/Colab Notebooks/kfolddata/fold'+str(fold+1)+'/train'
  test_path = '/content/drive/MyDrive/Colab Notebooks/kfolddata/fold'+str(fold+1)+'/test'
  train_set, test_set, validation_set = set_data(train_path,test_path, batchSize, image_size)
  model = create_model()
  checkpoint_path = "weights_best.hdf5"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   monitor='val_accuracy',mode='max',
                                                 save_best_only=True,
                                                 verbose=1)
  history=model.fit(train_set, epochs = epoches, validation_data= validation_set, callbacks=[cp_callback], shuffle=True)
  results1 = model.evaluate(test_set,batch_size=8)
  accuracyT[fold] = results1[1]
  model.save('/content/drive/MyDrive/Colab Notebooks/Models/clinical/a4100/fold'+str(fold+1)+'/best100.hdf5')

  model.load_weights("weights_best.hdf5")
  model.save('/content/drive/MyDrive/Colab Notebooks/Models/clinical/f4100/fold'+str(fold+1)+'/best100.hdf5')

  results = model.evaluate(test_set, batch_size=8)
  accuracy[fold] = results[1]
  plot_hist(history)
 
print(accuracy)
print("Average best is:", sum(accuracy)/5.0)
print (accuracyT)
print("Average is:", sum(accuracyT)/5.0)
