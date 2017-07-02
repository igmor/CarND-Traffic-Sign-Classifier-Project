#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/0.png "30km/h"
[image5]: ./examples/1.png "Keep right"
[image6]: ./examples/2.png "Priority Road"
[image7]: ./examples/3.png "Yield"
[image8]: ./examples/4.png "Stop"
[image12]: ./examples/5.png "Children crossing"
[image9]: ./examples/data_hist.png "Data Histogram"
[image10]: ./examples/test_image.png "Test image"
[image11]: ./examples/processed_image.png "Processed image"
[image13]: ./examples/softmax_prob.png "Top 5 Softmax probabilities"
[image14]: ./examples/conv.png "Convolutional layer 2"
[image15]: .//examples/5_traffic_signs.png "Five german traffic signs from the web"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code](https://github.com/igmor/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (36, 36, 3)
* Number of unique classes in the data set = 43

####2. 
Here is the data set distribution histogram. You can see a slight bias towards
some traffic signs in the beginning of the list.

![alt text][image9]

###Design and Test a Model Architecture

Before doing anything with the data set I have decided to pad the images from 32x32 to 36x36. That was rather
an experimental step that yielded suprisignly good results in the entire pipeline

Then I decided to convert the images to grayscale because it  
simplifies the architecture and makes it more tolerant towards noise in
different color channels. Also if one thinks about the problem of traffic sign classifier
it is really about being able to correctly identify shapes and pictures in traffic signs, color information
is redandant in this case. I also decided to normalize the image by subtracting a mean from all the image pixels
and then deviding them by standard deviation. I also used a perceptually friendly transformation
to a grayscale 0.21*R + 0.72*G + 0.07*B that slightly amplifies a blue channel

Here is an example of a traffic sign image before and after grayscaling and normalizing

![alt text][image10]
![alt text][image11]


####2. The architecture of my classifier CNN network was inspired by CIFAR CNN, more specifically
I used unusually deep filter depths in two convolutional layers: 32 and 64 accordingly, then
it was folded into 3 fully connected layers

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 36x36x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 12x12x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 					|
| Fully connected		| 2304x120     									|
| Fully connected		| 120x84     									|
| Fully connected		| 84x10     									|
|						|												|
 

To train the model, I used an a variant of a gradient descent optimizer (AdamOptimizer) with softmax cross entropy with logits
as a loss function. I used number of epoch  = 15, learning rate = 0.003 and batch size = 128
I've tried to play with learning rates and different batch sizes, batch size > 128 were causing a significant underfitting while batch size = 64 was generally ok but in some cases of hyperparameters was extremely unstable.
Learning rate 0.003 was also a bit higher comparing to the default 0.001 but proved to provide good fast converging results to ~ 0.9 accuracy almost immediately from the first iteration.

####4. Iterative approach

My final model results were:
* training set accuracy of 0.947
* validation set accuracy of 0.934 
* test set accuracy of 0.932

I first started with general LeNet architecture but that was a dead end as it was clear the accuracy of that network peaked at ~85% regardless of what I did with the data. So I just started reading about different CNN architectures and some ideas from CIFAR-10 was something that I decided to try out and almost immediately got very promising results. More specifically the network with two convolutional 5x5 layers of filter depth 32 and 64 depth started converging to ~0.9 from the very first iterations of training.
The next thing I did that got me very good results was padding the training set to 36x36. Apparently the was some information lost in translation if one does not do that. I also decided to keep really simple pre processing stack consisting of steps normalizing images and grayscaling but I believe I could have explored more with shifting and augmenting the training set that would have probably gotten me into a high 0.9 accuracy. 


###Test a Model on New Images

####1. Traffic Signs from the web

Here are six (I picked one more by mistake and decided to keep it in the list) German traffic signs that I found on the web. 
I copied all of them from Berlin Google map just browsing on a street, then made screen shots and converted the images into 32x32 thumbnails.

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image12]


The image 30 km/h might be difficult to classify because it tilted a bit, its shape is not an ideal circle, the same is true for children crossing sign: it's a bit shifted to the left and generally speaking figurines of chidren have lots of fine graned structure that is hard to recognise properly

####2. Predictions.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Yield     			| Yield 										|
| Priority road			| Priority road									|
| 30 km/h	      		| 30 km/h 						 				|
| Keep right			| Keep right      								|
| Children crossing		| Children crossing   							|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of a 93%

####3. 
The model was very certain in predictions with final predictions being in 0.99 percentile group.

Softmax probabilities looks like this:

```
TopKV2(values=array([[  1.00000000e+00,   1.10655152e-10,   6.29429148e-12,
          2.50925962e-13,   9.30440824e-14],
       [  1.00000000e+00,   1.94001016e-22,   4.29774040e-27,
          3.40759407e-28,   2.20673149e-28],
       [  1.00000000e+00,   1.38332583e-13,   1.08967231e-14,
          5.04938532e-15,   1.10998575e-15],
       [  1.00000000e+00,   4.03179851e-19,   2.00988569e-23,
          1.41999026e-24,   2.15754937e-25],
       [  1.00000000e+00,   4.03202934e-19,   4.22562267e-21,
          2.59299332e-21,   1.30059025e-21],
       [  9.99984264e-01,   1.57158902e-05,   8.83159679e-09,
          1.21080190e-09,   4.68334087e-11]], dtype=float32), indices=array([[ 1,  5, 32,  2,  3],
       [38,  8, 20, 31, 40],
       [12, 35, 13, 28,  9],
       [13, 12,  1, 29,  9],
       [14,  4, 13,  5, 38],
       [28, 30, 11, 26, 23]], dtype=int32))
```

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook:

![alt text][image13]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

This is the output of my convolutional layer 2. You can see it's 5x32 feature maps. I believe if it was shaped like the input images 32x32 we would be able to see some spatial patterns which convolutional layer learns. In this case it is pretty hard to make any conclusions:

![alt text][image14]



