# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:59:35 2023
----------
FLINET is 3d CNN model for lifetime parameter prediction.
It is taken from Smith et al. paper
----------
We are going to compare the our DL model performance with FLINET

@author: Mou Adhikari
"""
# Relevant libraries and functions
from __future__ import print_function
import random
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np, h5py
import os, time, sys
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import BatchNormalization, Convolution2D, Input, SpatialDropout2D, UpSampling2D, MaxPooling2D, concatenate
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, Conv1D, Input, Conv2D, add, Conv3D, Reshape
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
from itertools import cycle
from sklearn import metrics
from keras.optimizers import RMSprop
from keras.utils import np_utils
from tensorflow.compat.v1.keras.backend import set_session
from keras.layers.convolutional import Convolution2D, MaxPooling2D, SeparableConv2D, Conv2DTranspose



f_data = r'C:\Users\fe73yap\Downloads\Train_data_FLINET' # Directory with trainging data
stacks = os.listdir(f_data)
numS = int(len(stacks))

nTG = 256 # Number of time-points
xX = 28
yY = 28

tpsfD = np.ndarray(
        (numS, int(nTG), int(xX), int(yY), int(1)), dtype=np.float32
        )
t1 = np.ndarray(
        (numS, int(xX), int(yY), int(1)), dtype=np.float32
        )
t2 = np.ndarray(
        (numS, int(xX), int(yY), int(1)), dtype=np.float32
        )
tR = np.ndarray(
        (numS, int(xX), int(yY), int(1)), dtype=np.float32
        )

i = 0;
for d in stacks:
    #Save values to respective mapping
    f = h5py.File(os.path.join(f_data,d),'r') 
    tpsfD[i,:,:,:,0] = f.get('sigD')
    f = h5py.File(os.path.join(f_data,d),'r') 
    t1[i,:,:,0] = f.get('t1')
    f = h5py.File(os.path.join(f_data,d),'r') 
    t2[i,:,:,0] = f.get('t2')
    f = h5py.File(os.path.join(f_data,d),'r') 
    tR[i,:,:,0] = f.get('rT')
    i = i + 1
    
tpsfD =  np.moveaxis(tpsfD, 1, -2)


#%%

# Relevant resblock functions (Keras API)
def resblock_2D(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

def resblock_2D_BN(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    #output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def resblock_3D_BN(num_filters, size_filter, x):
    Fx = Conv3D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv3D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    #output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def xCeptionblock_2D_BN(num_filters, size_filter, x):
    Fx = SeparableConv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = SeparableConv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

#%%

modelD = None
xX = 28;
yY = 28;

t_data = Input(shape=(xX, yY, 256,1))
tpsf = t_data

# # # # # # # # 3D-Model # # # # # # # #

tpsf = Conv3D(50,kernel_size=(1,1,10),strides=(1,1,5), padding='same', activation=None, data_format="channels_last")(tpsf)
tpsf = BatchNormalization()(tpsf)
tpsf = Activation('relu')(tpsf)
tpsf = resblock_3D_BN(50, (1,1,5), tpsf)
tpsf = Reshape((xX,yY,2600))(tpsf)
tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
tpsf = BatchNormalization()(tpsf)
tpsf = Activation('relu')(tpsf)
tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
tpsf = BatchNormalization()(tpsf)
tpsf = Activation('relu')(tpsf)
tpsf = resblock_2D_BN(256, 1, tpsf)
tpsf = resblock_2D_BN(256, 1, tpsf)

# Short-lifetime branch
imgT1 = Conv2D(64, 1, padding='same', activation=None)(tpsf)
imgT1 = BatchNormalization()(imgT1)
imgT1 = Activation('relu')(imgT1)
imgT1 = Conv2D(32, 1, padding='same', activation=None)(imgT1)
imgT1 = BatchNormalization()(imgT1)
imgT1 = Activation('relu')(imgT1)
imgT1 = Conv2D(1, 1, padding='same', activation=None)(imgT1)
imgT1 = Activation('relu')(imgT1)

# Long-lifetime branch
imgT2 = Conv2D(64, 1, padding='same', activation=None)(tpsf)
imgT2 = BatchNormalization()(imgT2)
imgT2 = Activation('relu')(imgT2)
imgT2 = Conv2D(32, 1, padding='same', activation=None)(imgT2)
imgT2 = BatchNormalization()(imgT2)
imgT2 = Activation('relu')(imgT2)
imgT2 = Conv2D(1, 1, padding='same', activation=None)(imgT2)
imgT2 = Activation('relu')(imgT2)

# Amplitude-Ratio branch
imgTR = Conv2D(64, 1, padding='same', activation=None)(tpsf)
imgTR = BatchNormalization()(imgTR)
imgTR = Activation('relu')(imgTR)
imgTR = Conv2D(32, 1, padding='same', activation=None)(imgTR)
imgTR = BatchNormalization()(imgTR)
imgTR = Activation('relu')(imgTR)
imgTR = Conv2D(1, 1, padding='same', activation=None)(imgTR)
imgTR = Activation('relu')(imgTR)

modelD = Model(inputs=[t_data], outputs=[imgT1,imgT2, imgTR])
rmsprop = RMSprop(lr=1e-5)

modelD.compile(loss='mse',
              optimizer=rmsprop,
              metrics=['mae'])
modelD.summary()

#%%

# Setting patience (patience = 15 recommended)
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience = 15, 
                              verbose = 1,
                              mode = 'auto')

fN = 'testName' # Assign some name for weights and training/validation loss curves here

# Save loss curve (mse) and MAE information over all trained epochs. (monitor = '' can be changed to focus on other tau parameters)
modelCheckPoint = ModelCheckpoint(filepath=fN+'.h5', 
                                  monitor='val_loss', 
                                  save_best_only=True, 
                                  verbose=0)
# Train network (80/20 train/validation split, batch_size=20 recommended, nb_epoch may vary based on application)
history = History()
csv_logger = CSVLogger(fN+'.log')
tic = time.perf_counter()
history = modelD.fit([tpsfD], [t1,t2,tR],
          validation_split=0.2,
          batch_size=20, epochs=500, verbose=1, shuffle=True, callbacks=[earlyStopping,csv_logger,modelCheckPoint])
toc = time.perf_counter()
print(f" Training time {toc - tic:0.4f} seconds")

#%%
#plt.style.use('seaborn-whitegrid') 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#%%
modelD.load_weights(fN+'.h5')

#%%

# Upload test data and use 3D-CNN for inference

t_data = r'C:\Users\fe73yap\Downloads\Train_data_FLINET' # directory with test data
stacksT = os.listdir(f_data)
numT = int(len(stacks))

nTG = 256
xX = 28
yY = 28

tpsfT = np.ndarray(
        (numT, int(nTG), int(xX), int(yY), int(1)), dtype=np.float32
        )
t1T = np.ndarray(
        (numT, int(xX), int(yY), int(1)), dtype=np.float32
        )
t2T = np.ndarray(
        (numT, int(xX), int(yY), int(1)), dtype=np.float32
        )
tRT = np.ndarray(
        (numT, int(xX), int(yY), int(1)), dtype=np.float32
        )

i = 0;
for d in stacksT:
    # Save values to respective mapping
    f = h5py.File(os.path.join(f_data,d),'r') 
    tpsfT[i,:,:,:,0] = f.get('sigD')
    f = h5py.File(os.path.join(f_data,d),'r') 
    t1T[i,:,:,0] = f.get('t1')
    f = h5py.File(os.path.join(f_data,d),'r') 
    t2T[i,:,:,0] = f.get('t2')
    f = h5py.File(os.path.join(f_data,d),'r') 
    tRT[i,:,:,0] = f.get('rT')
    i = i + 1
    
tpsfT =  np.moveaxis(tpsfT, 1, -2)
# tpsfT = np.moveaxis(tpsfT, 1, 2)