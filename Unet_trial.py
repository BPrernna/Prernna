# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:48:12 2020

@author: Prernna
"""



import tensorflow as tf
import os
import cv2
import glob 
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
from keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense
import math
seed = 200
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
data_location="F:\Prernna\Data\Dataset_Creation/"
training_images_loc = data_location + '/Training_T1/'
training_images_label = data_location + '/White Matter/'
training_images_label_2 = data_location + '/Grey Matter/'
TEST_PATH_label_GM =  data_location + '/Ground Truth_test_GM/'
TEST_PATH =  data_location + '/Test_Data/'
train_files = os.listdir(training_images_loc)
test_ids = os.listdir(TEST_PATH)
TEST_PATH_label =  data_location + '/Ground Truth_Test/'
val_path=data_location + 'Validation_T1/'
val_path_label=data_location + 'Validation_GT/'
val_ids = os.listdir(val_path)
X_train = np.zeros((len(train_files), 256, 256, 1), dtype=np.uint8)
Y_train = np.zeros((len(train_files), 256, 256, 1), dtype=np.bool)
#Z_train=  np.zeros((len(train_files),192, IMG_WIDTH,1), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_files), total=len(train_files)):   
    path = training_images_loc+ id_
    img = imread(path)
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img =  np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
    #meta_image = np.stack(img)   
    X_train[n] = img #Fill empty X_train with values from img
mask_ = np.zeros((256, 256, 1), dtype=np.bool)
for n, id2 in tqdm(enumerate(train_files), total=len(train_files)): 
     path_label=training_images_label+id2
     mask_ = imread(path_label)
     #mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
     meta_image_label = np.stack(mask_)  
     meta_image_label = np.expand_dims(resize(meta_image_label, (256, 256), mode='constant',  
                                      preserve_range=True), axis=-1)
        
     Y_train[n] = meta_image_label
     
     mask_GM = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
for n, id4 in tqdm(enumerate(train_files), total=len(train_files)): 
     path_label_GM=training_images_label_2+id4
     mask_GM = imread(path_label_GM)
     mask_GM = np.expand_dims(resize(mask_GM, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
    
     
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
sizes_test = []
print('Resizing test images') 
for n, id2_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path2 = TEST_PATH + id2_
    img2 = imread(path2)
    #sizes_test.append([img.shape[0], img.shape[1]])
    img2=np.expand_dims(resize(img2, (256, 256), mode='constant',  
                                      preserve_range=True), axis=-1)
    #img2 = resize(img2, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img2
    
    mask_test = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
for n, id3 in tqdm(enumerate(test_ids), total=len(test_ids)): 
     path_label_test = TEST_PATH_label + id3
     mask_test = imread(path_label_test)
     mask_test = np.expand_dims(resize(mask_test, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        
     Y_test[n] = mask_test
     
     #mask_test_GM = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
#for n, id4 in tqdm(enumerate(test_ids), total=len(test_ids)): 
    # path_label_test_GM = TEST_PATH_label_GM + id4
    # mask_test_GM = imread(path_label_test_GM)
     #mask_test_GM = np.expand_dims(resize(mask_test_GM, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      #preserve_range=True), axis=-1)
        
    # Z_test[n] = mask_test_GM
#X_val = np.zeros((len(val_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
#Y_val = np.zeros((len(val_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#for n, id4 in tqdm(enumerate(val_ids), total=len(val_ids)):   
    #path3 = val_path+ id4
   # img3 = cv2.imread(path3)
   
    #img3 = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #img3 =  np.expand_dims(resize(img3, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      #preserve_range=True), axis=-1)
    #meta_image = np.stack(img)   
    #X_val[n] = img3 
#mask_val = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#for n, id6 in tqdm(enumerate(val_ids), total=len(val_ids)): 
     #path_label_val = val_path_label+id6
     #mask_val = imread(path_label_val)
     #mask_val = np.expand_dims(resize(mask_val, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      #preserve_range=True), axis=-1)
        
     #Y_val[n] = mask_val    
     
image_x = random.randint(0, len(train_files))
#image_x=300
imshow(np.squeeze(X_train[image_x]))
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#datagen = ImageDataGenerator(
    #rotation_range=15,
    #horizontal_flip=True,
    #width_shift_range=0.1,
    #height_shift_range=0.1
    #zoom_range=0.3
    #)
#datagen.fit(X_train)



#datagen = ImageDataGenerator(rotation_range = 30, horizontal_flip = True, zoom_range = 0.2)

inputs = tf.keras.layers.Input((256, 256, 1))
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

s = tf.keras.layers.Lambda(lambda x: x)(inputs)
#e1=tf.keras.layers.Conv2D(8,(3,3),strides=(1,1),activation='relu', kernel_initializer='he_normal',padding='same')(s)
#c1=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e1)
#c1=tf.keras.layers.ReLU()(c1)


#p1 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c1)


e2=tf.keras.layers.Conv2D(16,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(s)
c2=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e2)
c2=tf.keras.layers.ReLU()(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c2)


e3=tf.keras.layers.Conv2D(32,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e3)
c3=tf.keras.layers.ReLU()(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c3)


e4=tf.keras.layers.Conv2D(64,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e4)
c4=tf.keras.layers.ReLU()(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c4)


e5=tf.keras.layers.Conv2D(128,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e5)
c5=tf.keras.layers.ReLU()(c5)
p5 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c5)

e6=tf.keras.layers.Conv2D(256,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(p5)
c6=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(e6)
c6=tf.keras.layers.ReLU()(c6)
p6 = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2))(c6)

#ExpansivePath 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2),dilation_rate=(1,1), strides=(2,2), padding='same')(p6)
c7=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(u6)
c7=tf.keras.layers.ReLU()(c7)
c7=tf.keras.layers.Conv2D(128,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c7)
c7=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c7) 
c7=tf.keras.layers.ReLU()(c7)
u6 = tf.keras.layers.concatenate([u6, e6])
c7=tf.keras.layers.Conv2D(128,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c7=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c7) 
c7=tf.keras.layers.ReLU()(c7)
c7 = tf.keras.layers.Dropout(0.2)(c7)

 

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2),dilation_rate=(1,1),strides=(2,2), padding='same')(c7)
c8=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(u7)
c8=tf.keras.layers.ReLU()(c8)
c8=tf.keras.layers.Conv2D(64,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c8)
c8=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c8) 
c8=tf.keras.layers.ReLU()(c8)
u7 = tf.keras.layers.concatenate([u7, e5])
c8=tf.keras.layers.Conv2D(64,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c8=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c8) 
c8=tf.keras.layers.ReLU()(c8)
c8 = tf.keras.layers.Dropout(0.2)(c8)




u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2),dilation_rate=(1,1), strides=(2,2), padding='same')(c8)
c9=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(u8)
c9=tf.keras.layers.ReLU()(c9)
c9=tf.keras.layers.Conv2D(32,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu',kernel_initializer='he_normal', padding='same')(c9)
c9=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c9) 
c9=tf.keras.layers.ReLU()(c9)
u8 = tf.keras.layers.concatenate([u8, e4])
c9=tf.keras.layers.Conv2D(32,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c9=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c9) 
c9=tf.keras.layers.ReLU()(c9)
c9 = tf.keras.layers.Dropout(0.3)(c9)



u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2),dilation_rate=(1,1),strides=(2,2), padding='same')(c9)
c10=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(u9)
c10=tf.keras.layers.ReLU()(c10)
c10=tf.keras.layers.Conv2D(16,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c10)
c10=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c10)
c10=tf.keras.layers.ReLU()(c10) 
u9 = tf.keras.layers.concatenate([u9, e3])
c10=tf.keras.layers.Conv2D(16,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c10=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c10) 
c10=tf.keras.layers.ReLU()(c10)
c10 = tf.keras.layers.Dropout(0.3)(c10)

u10 = tf.keras.layers.Conv2DTranspose(8, (2, 2),dilation_rate=(1,1),strides=(2,2), padding='same')(c10)
c11=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(u10)
c11=tf.keras.layers.ReLU()(c11)
c11=tf.keras.layers.Conv2D(8,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c11)
c11=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c11) 
c11=tf.keras.layers.ReLU()(c11)
#u10= tf.keras.layers.concatenate([c11, c2])
#c11=tf.keras.layers.Conv2D(8,(3,3),dilation_rate=(1,1),strides=(1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c11)
#c11=tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001)(c11) 
#c11=tf.keras.layers.ReLU()(c11)
c11 = tf.keras.layers.Dropout(0.4)(c11)

output_WM = tf.keras.layers.Conv2D(1, (3, 3),dilation_rate=(1,1),strides=(1,1), padding='same',activation='sigmoid')(c11)
Y_train=tf.cast(Y_train, tf.float32) 
#Y_test= tf.cast(Y_test, tf.float32) 

def dice_coef(y_true, y_pred,smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model = tf.keras.Model(inputs=[inputs], outputs=[output_WM])
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
model.summary()
checkpointer=tf.keras.callbacks.ModelCheckpoint('test_unet_dice_5.h5', verbose=1,save_best_only=True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results=model.fit(X_train,Y_train,shuffle='true',validation_split=0.1,batch_size=10,epochs=30,callbacks=callbacks)
#results=model.fit(datagen.flow(X_train, Y_train,batch_size=10),steps_per_epoch = len(X_train)/10,epochs=30,validation_data=(X_test, Y_test),callbacks=callbacks)

yp = model.predict(x=X_test, batch_size=10, verbose=1)
yp = np.round(yp,0)

image_x = 90
#image_x=300
plt.subplot(1,3,1)
imshow(np.squeeze(X_test[image_x]))

plt.subplot(1,3,2)
imshow(np.squeeze(Y_test[image_x]))

plt.subplot(1,3,3)
imshow(np.squeeze(yp[image_x]))

