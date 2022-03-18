




If model predictions are poor on both the training and validation set (for example, mean squared error is high on both), then this is evidence of underfitting. Possible solutions could be to

    * increase the number of epochs
	* add more convolutions to the network.

When the model predicts well on the training set but poorly on the validation set (for example, low mean squared error for training set, high mean squared error for validation set), this is evidence of overfitting. If the model is overfitting, a few ideas could be to

    * use dropout or pooling layers
    * use fewer convolution or fewer fully connected layers
    * collect more data or further augment the data set



# Mean Squared Error
Since this model outputs a single continuous numeric value (which is the predicted steering angle), one appropriate error metric would be the **MSE - Mean Squared Error**. The parameters I took in consideration when analyzing the model's performance were:

* If the mean squared error is high on both a training and validation set, the model is underfitting. 
* If the mean squared error is low on a training set but high on a validation set, the model is overfitting.


# Behavioral Cloning
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
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
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

The model includes RELU layers to introduce nonlinearity and one dropout layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).


#### 1. Solution Design Approach







I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Before Balancing Data][gif1]

![After Balancing Data][gif2]