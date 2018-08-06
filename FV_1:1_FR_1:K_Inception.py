#Import Dependencies
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import cv2
import os
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

from keras.layers import ZeroPadding2D, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import AveragePooling2D, Concatenate, Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
K.set_image_data_format("channels_first")

%load_ext autoreload
%autoreload 2

#Load a Pre-Trained Inception Model.
#Inputs: (m, 3, 96, 96) --> Inception --> outputs: (m, 128)
#Two conventions which people use are: (m, n_H, n_W, n_C) and (m, n_C, n_H, n_W)

FRmodel = faceRecoModel(input_shape = (3, 96, 96))
print("Total Params: ", FRmodel.count_params())     # ~3.7m parameters

#Triplet loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
  anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
  pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
  neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
  basic_loss = pos_dist - neg_dist + alpha
  loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
  
  return loss
  
with tf.Session() as sess:
  y_true = (None, None, None)
  y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
            tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
            tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
  loss = triplet_loss(y_true, y_pred)
  print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = "adam", loss = triplet_loss, metrics = ["accuracy"])
load_weights_from_FaceNet(FRmodel)

#Create the Database
database

#Face Verification (1 : 1)
def verify(image_path, identity, database, model):
  encoding = img_to_encoding(image_path, model)
  
  distance = np.linalg.norm(encoding - database[identity])
  
  if distance < 0.7:
    print("It's " + str(identity) + ", welcome home!")
    door_open = True
  else:
    print("It's not " + str(identity) + ", please go away")
    door_open = False
  
  return distance, door_open

verify("image", "name", database, FRmodel)

#Face Recognition (1 : K)
def who_is_it(image_path, database, model):
  encoding = img_to_encoding(image_path, model)
  min_dist = 100
  
  for (name, db_enc) in database.items():
    distance = np.linalg.norm(encoding - db_enc)
    if distance < min_dist:
      min_dist = distance
      identity = name
  
  if min_dist < 0.7:
    print("it's " + str(identity) + ", the distance is " + str(min_dist))
  else:
    print("Not in the database.")
    
  return min_dist, identity

#To Improve the accuracy further add more images of a person into the database
#Crop the Input Image to make the Algorithm perform well on it.
who_is_it("image", database, FRmodel)
