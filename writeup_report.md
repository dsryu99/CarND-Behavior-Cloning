# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my-examples/center_lane.jpg "Center Lane Driving"
[image2]: ./my-examples/normal.jpg "Normal Image"
[image3]: ./my-examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 for the video recoding of the car driving autonomously
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the neural network developed by NVIDIA for driving a real car autonomously. It consists of a normalization layer, five convolution neural network layers  and four fully connected layers. (model.py lines 63-75)
The data is normalized in the model using a Keras lambda layer (code line 64). The model includes RELU layers to introduce nonlinearity (code line 66-70).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and its learning rate is manually set 0.0005 (model.py line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used three camera images (i.e., left, center and right camera images) to train the model. In addition, Flipped images of all three camera images are also used as training data.
For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to refine the model from a simple sequential model to more powerful convolutional neural network model.

My first step was to use a sequential neural network model and next a convolutional neural network similar to the LeNet model. Finally, I used a NVIDIA model which is proven powerful for driving a real car autonomously. I thought this model might be appropriate because multiple convolutional network connections make it easy to extract the features of images correctly.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In the first sequential model, a mean squared error (MSE) on the training set and the validation set was high. After adopting a convolutional network model, I could get the reasonable MSE on both data sets.

Next, I fine-tuned the number of epochs, augmented the data set by applying three camera images (i.e., a center, left and right camera images).

The final step was to run the simulator to see how well the car was driving around track one. There was no problem on the straight road. However, the vehicle fell off the track when the road is sharply curvy. To improve the driving behavior in these cases, I collected more data and augmented the data set by flipping.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-75) consisted of a convolution neural network with the following layers.

| Layer           		|     Description	        					|
|:-------------  ----:|:---------------------------------:|
| Input            		| 160x320x3 RGB image   							|
| Lamda               | Normalization |
| Cropping            | (70,25),(0,0) |
| Convolution 5x5    	| 1x1 stride, valid padding, kernel 24 |
| RELU				       	|												|
| Convolution 5x5     | 1x1 stride, valid padding, kernel 36 |
| RELU                | |
| Convolution 5x5     | 1x1 stride, valid padding, kernel 48 |
| RELU                | |
| Convolution 3x3     | 1x1 stride, valid padding, kernel 64 |
| RELU                | |
| Convolution 3x3     | 1x1 stride, valid padding, kernel 64 |
| RELU                | |
| Flatten             | |
| Fully connected    	| outputs 100|
| Fully connected     | outputs 50 |
| Fully connected     | outputs 10 |
| Fully connected     | outputs 1 |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would reduce the overfitting problem. On the track, a car only turns left. So it has the left turn bias. By applying the flipping technique, this bias can be addressed. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]

After the collection process, I had 6573 number of data points. I then preprocessed this data by applying the normalization. In 160x320 pixel images, there exists data not useful for the autonomous driving. So I cropped the images. It also speeds up the training task.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 5 as evidenced by plotting the progress of training and validation eror based on the epoch. I used an adam optimizer and the learning rate was set 0.0005 so that the optimizer can converge to the optimum.
