# **Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

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

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


## TODO

#### Data generation:

To generate training data for the car I recorded myself manually driving the car following the center of the track with both a forwards lap and a reverse lap.

To gather data where the car may not be in the center of the track, I recorded driving where I would purposefully veer away from center and recover. After I finished recording, I used a python script to help me delete the data where I was veering away from center (maintaining the recovery). For this data set I deleted approximately 30 time slices of data.

#### Model selection:

* Develop code to run existing models:
  * LeNet, AlexNet, GoogLeNet, VGG, Resnet
* Use pretrained models and feature map final layer(s) - center image only
* Train from scratch existing models and compare results
* Pick best performer

#### Data Augmentation:

* Augment data with mirror image of center camera
* Incorporate left and right cameras with fixed delta steering angle
* Vertical and lateral offsets
* Rotations of +/- 30 degrees
* Image masking

* Retry final training data set with different models
## END TODO

### Model Design

First model:

run1
Lambda
Flatten
Dense(50)
Activation(relu)
Dense(1)

Epoch 1/3
4388/4388 [==============================] - 5s - loss: 16.3487 - val_loss: 0.0119
Epoch 2/3
4388/4388 [==============================] - 4s - loss: 0.0169 - val_loss: 0.0119
Epoch 3/3
4388/4388 [==============================] - 4s - loss: 0.0123 - val_loss: 0.0119

No data augmentation, just prior editing of data to remove undesirable steering angles

run2
Same model + Flip image Augmentation

Epoch 1/3
4388/4388 [==============================] - 47s - loss: 10.5942 - val_loss: 0.0133
Epoch 2/3
4388/4388 [==============================] - 4s - loss: 1.3752 - val_loss: 0.0111
Epoch 3/3
4388/4388 [==============================] - 4s - loss: 0.1146 - val_loss: 0.0112

run3



RUN1

model = Sequential()
model.add(Cropping2D(cropping=((35,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(3, 1, 1))  #1x1 for color mapping
model.add(Convolution2D(24, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(24, 3, 3))  #1x1 for color mapping
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(36, 3, 3))  #1x1 for color mapping
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(64))
model.add(Dense(1))



4388/4388 [==============================] - 14s - loss: 33.4489 - val_loss: 0.2285
Epoch 2/5
4388/4388 [==============================] - 10s - loss: 0.0757 - val_loss: 0.0321
Epoch 3/5
4388/4388 [==============================] - 10s - loss: 0.0563 - val_loss: 0.0288
Epoch 4/5
4388/4388 [==============================] - 10s - loss: 0.0430 - val_loss: 0.0200
Epoch 5/5
4388/4388 [==============================] - 10s - loss: 0.0360 - val_loss: 0.0376


RUN2
Same model, doubled sample size by always having flipped image instead of randomly
Changed to 10 epochs

Epoch 1/10
8776/8776 [==============================] - 14s - loss: 16.5868 - val_loss: 0.1380
Epoch 2/10
8776/8776 [==============================] - 12s - loss: 0.0420 - val_loss: 0.0770
Epoch 3/10
8776/8776 [==============================] - 13s - loss: 0.0347 - val_loss: 0.0302
Epoch 4/10
8776/8776 [==============================] - 14s - loss: 0.0296 - val_loss: 0.0230
Epoch 5/10
8776/8776 [==============================] - 13s - loss: 0.0266 - val_loss: 0.0215
Epoch 6/10
8776/8776 [==============================] - 13s - loss: 0.0232 - val_loss: 0.0246
Epoch 7/10
8776/8776 [==============================] - 12s - loss: 0.0217 - val_loss: 0.0182
Epoch 8/10
8776/8776 [==============================] - 13s - loss: 0.0208 - val_loss: 0.0186
Epoch 9/10
8776/8776 [==============================] - 13s - loss: 0.0199 - val_loss: 0.0183
Epoch 10/10
8776/8776 [==============================] - 12s - loss: 0.0189 - val_loss: 0.0197


Epoch 1/7
8776/8776 [==============================] - 14s - loss: 3.3838 - val_loss: 0.0580
Epoch 2/7
8776/8776 [==============================] - 13s - loss: 0.0239 - val_loss: 0.0269
Epoch 3/7
8776/8776 [==============================] - 12s - loss: 0.0201 - val_loss: 0.0213
Epoch 4/7
8776/8776 [==============================] - 13s - loss: 0.0181 - val_loss: 0.0197
Epoch 5/7
8776/8776 [==============================] - 12s - loss: 0.0175 - val_loss: 0.0164
Epoch 6/7
8776/8776 [==============================] - 12s - loss: 0.0164 - val_loss: 0.0177
Epoch 7/7
8776/8776 [==============================] - 13s - loss: 0.0157 - val_loss: 0.0161

RUN3
Lateral & vertical offsets, steering adjust is 0.15deg/27pixels = 0.005 deg/pixel offsets

Epoch 1/7
4388/4388 [==============================] - 8s - loss: 2.6108 - val_loss: 0.0394
Epoch 2/7
4388/4388 [==============================] - 7s - loss: 0.0310 - val_loss: 0.0254
Epoch 3/7
4388/4388 [==============================] - 7s - loss: 0.0251 - val_loss: 0.0257
Epoch 4/7
4388/4388 [==============================] - 7s - loss: 0.0220 - val_loss: 0.0175
Epoch 5/7
4388/4388 [==============================] - 7s - loss: 0.0205 - val_loss: 0.0173
Epoch 6/7
4388/4388 [==============================] - 7s - loss: 0.0193 - val_loss: 0.0176
Epoch 7/7
4388/4388 [==============================] - 7s - loss: 0.0190 - val_loss: 0.0192






Alternate model architecture

model.add(Cropping2D(cropping=((35,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(3, 1, 1))  #1x1 for color mapping
model.add(Convolution2D(24, 5, 5))  #1x1 for color mapping
model.add(Convolution2D(24, 5, 5))  #1x1 for color mapping
#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5))  #1x1 for color mapping
model.add(Convolution2D(36, 5, 5))  #1x1 for color mapping
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Convolution2D(48, 3, 3))  #1x1 for color mapping
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(1))


4388/4388 [==============================] - 65s - loss: 8.5275 - val_loss: 0.0347
Epoch 2/7
4388/4388 [==============================] - 58s - loss: 0.0417 - val_loss: 0.0239
Epoch 3/7
4388/4388 [==============================] - 34s - loss: 0.0306 - val_loss: 0.0222
Epoch 4/7
4388/4388 [==============================] - 28s - loss: 0.0271 - val_loss: 0.0225
Epoch 5/7
4388/4388 [==============================] - 29s - loss: 0.0254 - val_loss: 0.0223
Epoch 6/7
4388/4388 [==============================] - 27s - loss: 0.0241 - val_loss: 0.0223
Epoch 7/7
4388/4388 [==============================] - 28s - loss: 0.0242 - val_loss: 0.0222


DELETED  


RUN4

New model architecture - simplified DENSE layers!!

I had somewhat arbitrarily assembled my model originally before incorporating data augmentation tools

4388/4388 [==============================] - 7s - loss: 0.0825 - val_loss: 0.0214
Epoch 2/7
4388/4388 [==============================] - 6s - loss: 0.0264 - val_loss: 0.0192
Epoch 3/7
4388/4388 [==============================] - 7s - loss: 0.0220 - val_loss: 0.0181
Epoch 4/7
4388/4388 [==============================] - 6s - loss: 0.0200 - val_loss: 0.0155
Epoch 5/7
4388/4388 [==============================] - 6s - loss: 0.0173 - val_loss: 0.0148
Epoch 6/7
4388/4388 [==============================] - 6s - loss: 0.0171 - val_loss: 0.0148
Epoch 7/7
4388/4388 [==============================] - 6s - loss: 0.0157 - val_loss: 0.0133





Add another dense layer:
model.add(Cropping2D(cropping=((35,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(3, 1, 1))  #1x1 for color mapping
model.add(Convolution2D(24, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(24, 3, 3))  #1x1 for color mapping
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(36, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(36, 3, 3))  #1x1 for color mapping
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Convolution2D(36, 3, 3))  #1x1 for color mapping
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(1))

4388/4388 [==============================] - 8s - loss: 0.0426 - val_loss: 0.0171
Epoch 2/7
4388/4388 [==============================] - 6s - loss: 0.0192 - val_loss: 0.0175
Epoch 3/7
4388/4388 [==============================] - 6s - loss: 0.0171 - val_loss: 0.0151
Epoch 4/7
4388/4388 [==============================] - 6s - loss: 0.0152 - val_loss: 0.0130
Epoch 5/7
4388/4388 [==============================] - 6s - loss: 0.0150 - val_loss: 0.0153
Epoch 6/7
4388/4388 [==============================] - 6s - loss: 0.0137 - val_loss: 0.0142
Epoch 7/7
4388/4388 [==============================] - 6s - loss: 0.0132 - val_loss: 0.0129


model.add(Cropping2D(cropping=((35,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64,64))))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(3, 1, 1))  #1x1 for color mapping
model.add(Convolution2D(16, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(16, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(16, 3, 3))  #1x1 for color mapping
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Convolution2D(32, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(32, 3, 3))  #1x1 for color mapping
model.add(Convolution2D(16, 3, 3))  #1x1 for color mapping
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(15))
model.add(Dense(35))
model.add(Dense(1))

Epoch 1/7
4388/4388 [==============================] - 7s - loss: 0.0451 - val_loss: 0.0195
Epoch 2/7
4388/4388 [==============================] - 6s - loss: 0.0204 - val_loss: 0.0168
Epoch 3/7
4388/4388 [==============================] - 7s - loss: 0.0159 - val_loss: 0.0166
Epoch 4/7
4388/4388 [==============================] - 6s - loss: 0.0164 - val_loss: 0.0134
Epoch 5/7
4388/4388 [==============================] - 6s - loss: 0.0148 - val_loss: 0.0125
Epoch 6/7
4388/4388 [==============================] - 6s - loss: 0.0142 - val_loss: 0.0120
Epoch 7/7
4388/4388 [==============================] - 7s - loss: 0.0136 - val_loss: 0.0129

RUN5

Added rotation of images
4388/4388 [==============================] - 10s - loss: 0.0350 - val_loss: 0.0213
Epoch 2/7
4388/4388 [==============================] - 8s - loss: 0.0214 - val_loss: 0.0202
Epoch 3/7
4388/4388 [==============================] - 8s - loss: 0.0194 - val_loss: 0.0216
Epoch 4/7
4388/4388 [==============================] - 9s - loss: 0.0182 - val_loss: 0.0167
Epoch 5/7
4388/4388 [==============================] - 8s - loss: 0.0180 - val_loss: 0.0162
Epoch 6/7
4388/4388 [==============================] - 9s - loss: 0.0179 - val_loss: 0.0142
Epoch 7/7
4388/4388 [==============================] - 9s - loss: 0.0166 - val_loss: 0.0159


RUN6

Updated distribution of training data to include more large steering angles



RUN7
Run with Udacity data set (no distribution adjustments)
6428/6428 [==============================] - 75s - loss: 0.0371 - val_loss: 0.0228
Epoch 2/7
6428/6428 [==============================] - 51s - loss: 0.0219 - val_loss: 0.0202
Epoch 3/7
6428/6428 [==============================] - 37s - loss: 0.0198 - val_loss: 0.0221
Epoch 4/7
6428/6428 [==============================] - 28s - loss: 0.0201 - val_loss: 0.0201
Epoch 5/7
6428/6428 [==============================] - 24s - loss: 0.0200 - val_loss: 0.0199
Epoch 6/7
6428/6428 [==============================] - 19s - loss: 0.0195 - val_loss: 0.0208
Epoch 7/7
6428/6428 [==============================] - 17s - loss: 0.0196 - val_loss: 0.0180

RUN8
Collected more data at the turn that was failing and added to training data


RUN9
Removed offsets and rotations augmentation

4739/4739 [==============================] - 9s - loss: 0.0560 - val_loss: 0.0125
Epoch 2/7
4739/4739 [==============================] - 7s - loss: 0.0133 - val_loss: 0.0094
Epoch 3/7
4739/4739 [==============================] - 6s - loss: 0.0106 - val_loss: 0.0088
Epoch 4/7
4739/4739 [==============================] - 7s - loss: 0.0089 - val_loss: 0.0081
Epoch 5/7
4739/4739 [==============================] - 7s - loss: 0.0089 - val_loss: 0.0084
Epoch 6/7
4739/4739 [==============================] - 6s - loss: 0.0087 - val_loss: 0.0078
Epoch 7/7
4739/4739 [==============================] - 6s - loss: 0.0084 - val_loss: 0.0082



RUN10 Lots more data
10780/10780 [==============================] - 94s - loss: 0.0238 - val_loss: 0.0188
Epoch 2/7
10780/10780 [==============================] - 124s - loss: 0.0187 - val_loss: 0.0157
Epoch 3/7
10780/10780 [==============================] - 122s - loss: 0.0172 - val_loss: 0.0142
Epoch 4/7
10780/10780 [==============================] - 95s - loss: 0.0157 - val_loss: 0.0136
Epoch 5/7
10780/10780 [==============================] - 82s - loss: 0.0156 - val_loss: 0.0142
Epoch 6/7
10780/10780 [==============================] - 57s - loss: 0.0153 - val_loss: 0.0126
Epoch 7/7
10780/10780 [==============================] - 39s - loss: 0.0144 - val_loss: 0.0128



Now with track 2 included:
17007/17007 [==============================] - 68s - loss: 0.0571 - val_loss: 0.0516
Epoch 2/10
17007/17007 [==============================] - 62s - loss: 0.0470 - val_loss: 0.0402
Epoch 3/10
17007/17007 [==============================] - 53s - loss: 0.0426 - val_loss: 0.0400
Epoch 4/10
17007/17007 [==============================] - 49s - loss: 0.0405 - val_loss: 0.0421
Epoch 5/10
17007/17007 [==============================] - 47s - loss: 0.0389 - val_loss: 0.0368
Epoch 6/10
17007/17007 [==============================] - 46s - loss: 0.0380 - val_loss: 0.0378
Epoch 7/10
17007/17007 [==============================] - 44s - loss: 0.0366 - val_loss: 0.0346
Epoch 8/10
17007/17007 [==============================] - 46s - loss: 0.0363 - val_loss: 0.0353
Epoch 9/10
17007/17007 [==============================] - 45s - loss: 0.0362 - val_loss: 0.0351
Epoch 10/10
17007/17007 [==============================] - 45s - loss: 0.0366 - val_loss: 0.0346



My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
