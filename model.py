import os
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import keras
from keras.backend import tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import ntpath
import random
sns.set()

#To specify the directory, so we can read and manipulate the DATA and cvs files
datadir = "./IMG"
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = column_names)

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

df['center'] = df['center'].apply(path_leaf)
df['left'] = df['left'].apply(path_leaf)
df['right'] = df['right'].apply(path_leaf)

print('total data:', len(df))


############## BALANCING THE DATA #############

#To flatten or samples and cut off the steering values which sum exceeds 300 and make it more uniform
num_bins = 25
hist, bins = np.histogram(df['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(x = center, height = hist, width=0.05)
samples_per_bin = 600
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(df['steering'])):
    #If the steering angle falls in between two bins, then it bellongs to the interval j
    if df['steering'][i] >= bins[j] and df['steering'][i] <= bins[j+1]:
      list_.append(i)
      #Eventually, this list will contain all the steering numbers from a specific bin. Because our treshold in this project is max 200 steering numbers per bin, we need to reject the exceding ones, and, because the numbers are stored in an array in order, we need to shuffle first (if we just reject the last ones, we may be rejecting information from the end of our track which is bad for our model to predict how to drive properly on the end of the track)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)

print('removed:', len(remove_list))
df.drop(df.index[remove_list], inplace=True)
print('remaining:', len(df))


############## TRAINING AND VALIDATION SPLIT #############
def load_img_steering(datadir,dataframe):
  image_path = []
  steering = []
  for i in range(len(df)):
    indexed_data = df.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2] 
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

#                                           Track  +  /IMG/  = Track/IMG/        
image_paths, steerings = load_img_steering(datadir, df)

#To split our image_paths and steering arrays into train and test 
X_train, X_valid, y_train, y_valid = train_test_split(image_paths,steerings, test_size=0.2)
#Checking their sizes (test size is 20% of the total and train is 80%)
print('Training Samples: {}\nValidation Samples: {}'.format(len(X_train), len(X_valid)))


############## DATA PREPROCESSING #############

# Defining Nvidia Model image input size
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

def img_preprocess(img):
  #To read the image paths we provided and store the actual image it contains:
  img = mpimg.imread(img)
  #Because we're working with the NVidea Model, it is required to change our image color-space from RGB to YUV
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  #Applying 3x3-Kernel GaussianBlur to smooth the image and reduce noie
  img = cv2.GaussianBlur(img, (3,3), 0)
  return img

#This map functio Make an iterator that computes the function using arguments from each of the iterables. Stops when the shortest iterable is exhausted.
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))


############## DEFINING NVIDIA MODEL #############
def nvidia_model():

  model = Sequential()
     
  model.add(Lambda(lambda x: (x / 255) - 0.5 , input_shape = (160, 320,3)))

  #Cropping the top 60 pixeld and the bottom 25 pixels from the image
  model.add(Cropping2D(cropping=((60,25), (0,0)))) 
    
  model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
  
  model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
  model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  model.add(Dropout(0.5))
 
 
  model.add(Flatten())
    
  model.add(Dense(100, activation='relu'))
  #model.add(Dropout(0.5))
 
  model.add(Dense(50, activation='relu'))
  #model.add(Dropout(0.5))

  model.add(Dense(10, activation ='relu'))
  #model.add(Dropout(0.5))

  model.add(Dense(1))
 
  optimizer= keras.optimizers.Adam(lr=1e-3)
  #Because we'll be working with a regression type example, the error metric is going to be the MSE (Mean Squared Error)
  model.compile(loss='mse', optimizer=optimizer)
 
  return model

model = nvidia_model()
print(model.summary())

################### TRAINING THE MODEL###################
history_object = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid), batch_size=100, verbose=1, shuffle=1)

### print the keys contained in the history object
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss function.png')


model.save('model.h5')

