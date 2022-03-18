# Behavioral Cloning - Udacity Self Driving Car Engineer NanoDegree Project
![After Balancing Data][gif2]
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the [Udacity Self Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior;
* Build a convolution neural network in Keras that predicts steering angles from images;
* Train and validate the model with a training and validation set;
* Test that the model successfully drives around track one without leaving the road;
* Summarize the results with a written report.


[//]: # (Image References)

[image1]: ./writeup_imgs/data_collection.png "Data Collection"
[image2]: ./writeup_imgs/preprocessed_img.png "Preprocessed Image"
[image3]: ./writeup_imgs/data_before_balancing.png "Data Before Balancing"
[image4]: ./writeup_imgs/data_after_balancing.png "Data After Balancing"
[image5]: ./writeup_imgs/train-valid.png "Train / Validation Dataset"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[gif1]: ./writeup_imgs/before_balancing.gif "Before Balancing Data"
[gif2]: ./writeup_imgs/after_balancing.gif "After Balancing Data"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

## Model Architecture and Training Strategy

### 1. Data Collection

Once I started recording my driving data using the [Udacity Self Driving Car Simulator](https://github.com/udacity/self-driving-car-sim), in order to get a good amount of data, at first I decided to perform about 3 complete laps on the racing track.

However, for me to get more balanced data, I realized that I must then perform 3 laps on the oposite direction, because on the first lap we'll be basically driving streering left, which can cause some problems later on when defining the biases - so the model may predict only left turns (another way to prevent this is by flipping the images inside the model.py training script). This is a very important step because basically the main parameter we need to predict in this project is the steering angle, so the car can know where to turn on each curve and will be able to drive by itslef!

![Data Collection][image1]

The repository containing all the collected data (including all the images and the *driving_log.csv* file can be downloaded my accessing this [Google Drive Link](https://drive.google.com/file/d/1c9_e8ltijkSe7ngAkyMNX_2R0HeWGL4u/view?usp=sharing).

In order to visualize the distribution, I plotted the histogram of steering angles for every set of images so I could see the distribution and know which steering angles are more frequent in my recording:

![Data Before Balancing ][image3]

Plotting this histogram shows us that the data I got when driving the car in the training track, there are more "0-angle" than negative or positive steering angles. This means that our track has much more straight areas then curves (which makes sense due to the track configuration!). But we can also see the left and right steering angles look balanced (because we drove in both ways).

So, with this data, I can see that I may have a big problem: because the majority of the steering angle values are on 0 (which means the car is going straight) this means that, when training the model with the data as it is now, the bias will get the model to predict the zero angle, and the car will be biased to be driving straight all the time. We need to balance the data by removing a bunch of data with "0-angle", so it will become more uniform, and the model will not have the tendency of predicting a specific steering angle. At first, I limited it to a max of 200 images per steering angle.

```
#To flatten or samples and cut off the steering values which sum exceeds 200 and make it more uniform
num_bins = 25
hist, bins = np.histogram(df['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(x = center, height = hist, width=0.05)
samples_per_bin = 200
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
```

Output:

![Data After Balancing ][image4]


### 2. Image Preprocessing

#### 2.1 Image Cropping

The first image of the generated dataset is taken from a camera mounted in the center of the windshield and we can see the hood of the car peaking out from the bottom of the image. The image is actually 160 pixels high and 320 pixels wide and the top portion of the image seems to capture hills, trees and the sky which might distract the model more than help it - it makes sense to crop it out of the image. The model might train faster if you crop each image to focus on only the portion of the image that is useful for predicting a steering angle.

To crop the image, we can use:
```
from keras.models import Sequential, Model
from keras.layers import Cropping2D
import cv2

model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3))) 
...
```

The Cropping2D code above crops:

    * 60 rows pixels from the top of the image
    * 25 rows pixels from the bottom of the image
    * 0 columns of pixels from the left of the image
    * 0 columns of pixels from the right of the image

#### 2.2 Image normalization

For normalization, I added a Lambda layer to the model. Within this Lambda layer, I'll normalize the image by dividing each element by 255 which is the maximum value of an image pixel. 

```
from keras.models import Sequential, Model
from keras.layers import Lambda

# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```

#### 2.3 Changing Image color space from RGB to YUV

Because we'll be working with the NVidea Model, it is required to change our image color-space from RGB to YUV. Basically, according to [Poynton, Charles. "YUV and luminance considered harmful: A plea for precise terminology in video"](https://en.wikipedia.org/wiki/YUV) YUV is a color encoding system typically used as part of a color image pipeline. It encodes a color image or video taking human perception into account, allowing reduced bandwidth for chrominance components, compared to a "direct" RGB-representation. For this change of the Color Space, I used opencv:

```
img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
```

#### 2.4 Applying Gaussian Blur

Applyed 3x3-Kernel GaussianBlur to smooth the image and reduce noise:

```
img = cv2.GaussianBlur(img, (3,3), 0)
```

So, the final image after the preprocessing became:

![Preprocessed Image ][image2]

### 3. Model Architecture and Training Strategy

As the goal of this model is to predict the steering angle based on images, we can say that it is a regression type model and a popular model used for behavioural cloning is called the **Nvidia Model** (more information [here](https://drive.google.com/file/d/1g5OCEUGjSYAHu-wbq6muDk8fAyIpVRfG/view?usp=sharing)). For this specific project, I found that this Nvidia Model has proven to be effective for behavioural cloning and is even used on real life self driving cars, turning to be a very nice opportunity for me to first implement it.

The Nvidia Model's architecture is like:
<center><img src="https://miro.medium.com/max/2504/1*2Z_8DB1ybUmRaHUsyi6bSA.png" width="575" height="500"></center>

<center><a href='https://towardsdatascience.com/deep-learning-for-self-driving-cars-7f198ef4cfa2'>Source</a></center>

The model includes RELU layers to introduce nonlinearity and a dropout layer in order to reduce overfitting. Also, I used an ADAM optimizer so that manually training the learning rate wasn't necessary.

Having said that, the final Nvidia model code I used was:

```
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
```

For the training itself, I considered the following inputs (chosen after literally dozens of attempts :P ) :

* Optimizer: Adam
* Learning rate: 1e-3
* Loss: MSE (Mean Squared Error)
* Number of Epochs: 15
* Batch size: 100
* Shuffle: YES

The model was trained and validated on different shuffled data sets to ensure that the model was not overfitting. Also, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

To split the data into training and validation (80% train and 20% validation), I used the code:

```
#To split our image_paths and steering arrays into train and test 
X_train, X_valid, y_train, y_valid = train_test_split(image_paths,steerings, test_size=0.2)
```

![Train / Validation Dataset][image5]

---

## Self-Driving Simulation Results

At first, I could notice that the car wasn't really driving straight. It was actually  steering even on places where it was supposed to not steer (and eventually fell of the track).

![Before Balancing Data][gif1]

The main reason of this error was that, when balancing the data, I used a relatively small number when determining the maximum number of samples per bin. Once the track contains much more straight areas than curves, the model was predicting the occasions where the car was supposed to perform '0-degrees' steer wrongly. After changing the bin from 200 to 600 samples, I got this final result:

![After Balancing Data][gif2]

So, as it is possible to check in full by opening the run1.mp4 file present in this repository, I could finally achieve the goal of programming my first self-driving car using Python and Keras!!