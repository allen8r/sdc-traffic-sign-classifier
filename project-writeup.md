# **Traffic Sign Recognition** 

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

[signs_palette]: ./visuals/palette.png "Traffic Signs Sample Palette"
[dataset_histograms]: ./visuals/dataset_histograms.png "Dataset historgrams"
[ahead-only]: ./predict_images/32x32/ahead-only.png "Ahead only"
[children-crossing]: ./predict_images/32x32/children-crossing.png "Children crossing"
[speed-30]: ./predict_images/32x32/speed-30.png "Speed limit (30km/h)"
[turn-right-ahead]: ./predict_images/32x32/turn-right-ahead.png "Turn right ahead"
[yield]: ./predict_images/32x32/yield.png "Yield"


## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

1. The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code cells executed and displaying output:

    [Traffic_Sign_Classifier.ipynb](https://github.com/allen8r/sdc-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)
  
2. An HTML or PDF export of the project notebook with the name report.html or report.pdf:

    [Traffic_Sign_Classifier.html](http://htmlpreview.github.com/?https://github.com/allen8r/sdc-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.html)

3. Any additional datasets or images used for the project that are not from the German Traffic Sign Dataset:

    See [Project repo](https://github.com/allen8r/sdc-traffic-sign-classifier).

    The pickled German Traffic Sign Dataset is externally available [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).
   
4. Your writeup report as a markdown or pdf file:

    You're reading it! Here is a link to my [project notebook](https://github.com/allen8r/sdc-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Dataset Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34,799**.
* The size of the validation set is **4,410**.
* The size of test set is **12,630**.
* The shape of a traffic sign image is **32x32x32**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. On the left is a sample palette of all the different traffic signs from the training set. On the right are histograms depicting the distribution of all the types of traffic signs within each dataset: the training, validation, and test sets. The histograms seem to indicate that the data is a bit heavier on the first 18 or so types of signs from the palette.

| ![alt text][signs_palette] | ![alt_text][dataset_histograms] |
|:--------------------------:|:-------------------------------:|

Here are the string name values assigned to each integer label (ClassId) for the different sign types in the dataset (from the [signnames.csv](https://github.com/allen8r/sdc-traffic-sign-classifier/blob/master/signnames.csv) file):

| ClassId | SignName                                          |
|:--------|:--------------------------------------------------|
|  0      | Speed limit (20km/h)                              |
|  1      | Speed limit (30km/h)                              |
|  2      | Speed limit (50km/h)                              |
|  3      | Speed limit (60km/h)                              |
|  4      | Speed limit (70km/h)                              |
|  5      | Speed limit (80km/h)                              |
|  6      | End of speed limit (80km/h)                       |
|  7      | Speed limit (100km/h)                             |
|  8      | Speed limit (120km/h)                             |
|  9      | No passing                                        |
| 10      | No passing for vehicles over 3.5 metric tons      |
| 11      | Right-of-way at the next intersection             |
| 12      | Priority road                                     |
| 13      | Yield                                             |
| 14      | Stop                                              |
| 15      | No vehicles                                       |
| 16      | Vehicles over 3.5 metric tons prohibited          |
| 17      | No entry                                          |
| 18      | General caution                                   |
| 19      | Dangerous curve to the left                       |
| 20      | Dangerous curve to the right                      |
| 21      | Double curve                                      |
| 22      | Bumpy road                                        |
| 23      | Slippery road                                     | 
| 24      | Road narrows on the right                         |
| 25      | Road work                                         |
| 26      | Traffic signals                                   |
| 27      | Pedestrians                                       |
| 28      | Children crossing                                 |
| 29      | Bicycles crossing                                 |
| 30      | Beware of ice/snow                                |
| 31      | Wild animals crossing                             |
| 32      | End of all speed and passing limits               |
| 33      | Turn right ahead                                  |
| 34      | Turn left ahead                                   |
| 35      | Ahead only                                        |
| 36      | Go straight or right                              |
| 37      | Go straight or left                               |
| 38      | Keep right                                        |
| 39      | Keep left                                         |
| 40      | Roundabout mandatory                              |
| 41      | End of no passing                                 |
| 42      | End of no passing by vehicles over 3.5 metric tons|
  

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

First, I normalized the training data by applying the basic RGB image normalization, (pixel - 128.0) / 128.0. As standard practice, data normalization helps in speeding up training and reduces the chance of getting stuck at local optima.

That's it for preprocessing! Instead of converting the training image data to grayscale, at the front of the training pipeline, I decided to apply a 1x1 convolution projection with a [1, 1, 3, 1] filter to transform the 3-channel image data to a single layer image data which turns out to be equivalent to converting to grayscale. See the convolution layer 0 in the model architecture below.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					                                        | 
|:----------------------|:--------------------------------------------------------------------------------------| 
| Input         		| 32x32x3 RGB image   							                                        | 
| Convolution 0  1x1   	| [1, 1, 3, 1] filter, 1x1 stride, VALID padding, outputs 32x32x1; convert to grayscale	|
| RELU					|												                                        |
| Convolution 1  5x5   	| [5, 5, 1, 20] filter, 1x1 stride, VALID padding, outputs 28x28x20                     |
| RELU					|												                                        |
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				                                        |
| Convolution 2  5x5    | [5, 5, 20, 36] filter, 1x1 stride, VALID padding, outputs 10x10x36                    |
| RELU					|												                                        |
| Max pooling	      	| 2x2 stride,  outputs 5x5x36 				                                            |
| Fully connected 0	    | flatten Input 5x5x36 to Output 900        									        |
| Fully connected 1	    | Input 900 to Output 1024        									                    |
| Dropout				|												                                        |
| RELU					|												                                        |
| Fully connected 2	    | Input 1024 to Output 512     									                        |
| Dropout				|												                                        |
| RELU					|												                                        |
| Fully connected 3	    | Input 512 to Output 128     									                        |
| Dropout				|												                                        |
| RELU					|												                                        |
| Output                | Input 128 to Output 43 (number of classes)        									|
| Softmax               | tf.nn.softmax_cross_entropy_with_logits                                               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a tf.train.AdamOptimizer. The optimizer was set up to minimize the cross entropy loss provided by tf.nn.sofmax_cross_entropy_with_logits. The following hyperparameters were used for the training:

| Hyperparameter          | Value |
| :-----------------------| -----:|
| Learning rate           | 0.001 |
| Epochs                  |    10 |
| Batch size              |   128 |
| Keep prob (for dropout) |  0.45 |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

| Dataset           | Accuracy          |
|:------------------|:------------------|
| Training          | **0.997**         |
| Validation        | **0.965**         | 
| Test              | **0.950**         |

As starting point for my model, I used the LeNet5 architecture (with 5x5 convolutions) from the lab work. Getting everything up and running with this neural network, the initial accuracy numbers were not very good. Validation accuracy was only 0.737.
Playing around with the hyperparameters--the learning rate, the number of epochs, and the batch size--had little or worse effect on the accuracy results. There was an issue with the normalization where using the integer 128 in the normalization
seem to affect the training outcome. Fixing the normalization by using 128.0 floating number upped the accuracy to 0.89. However, the still poor performance led me to start tinkering with the architecture of the model. Based on the LeNet5, 
I modified the convolution kernel to 3x3 (LeNet3), hoping that the finer filter would squeeze out more information from the image data. The accuracy actually went down slightly from before. At this point, I went back to the LeNet5 model and added
an additional fully connected layer and got a better result of 0.917. Seeing that the training accuracy was somewhat higher than the validation accuracy, I added a dropout to the fully connected layers; but result was slighly worse than before.
From the suggested tactics in the notebook, I then explored using grayscale images as input. But being lazy, I decided to add a 1x1 convolution layer at the front of the network to transform the incoming input image from 3 layer input to a
single-layer input, which turns out to be equivalent to preprocessing the image input data to grayscale. I got this idea from a helpful student's comments in the Slack channel. Having added this transform layer, surprisingly, the accuracy took
yet another hit. Turning back to Slack, I got another tip that we should pay attention to how we initialize the biases for the network. So, I modified the biases to be initialized with zeros instead of using random_normal. The zero initialization
for the biases seem to help quite a bit and the accuracy jumped back up 10% from the last iteration. Then a bit more tweaking that finally nudged up the accuracy was when I increased the number of hidden units for the fully
connected layers. Now we were at 0.922. The final adjustment to the model that put the accuracy over the 0.93 threshold was increasing the output depth of the convolution layers. I added more depth to the convolution layers in the hopes of
teasing out a little more information from the image input. Apparently, the deeper convolutional layers did the trick. Below is a table showing detailed notes from the training iterations:


| Iteration | Model (function) | Learn rate | Epochs | Batch size | Validation accuracy     | Total training time | Notes                                                                                                                            |
|----------:|:-----------------|:-----------|:------:|:----------:|:-----------------------:|--------------------:|:---------------------------------------------------------------------------------------------------------------------------------|
|         1 | LeNet5           | 0.001      |   10   |     128    |                   0.737 |           419.998 s |                                                                                                                                  |
|         2 | LeNet5           | 0.001      |   15   |     256    |                   0.725 |                   - | Changing epochs and batch size showed no improvement in accuracy                                                                 |
|         3 | LeNet5           | 0.0001     |   20   |     256    |                   0.561 |           789.800 s | Decreasing the learning rate to 0.0001 slowed down the training too much, resulting in accuracy of only 0.561 after 20 epochs.   |
|         4 | LeNet5           | 0.001      |   10   |     128    |                   0.894 |           409.572 s | Fixed input normalization in preprocessing from (x - 128) / 128 to ((x - 128.0) / 128.0)                                     |
|         5 | LeNet3           | 0.001      |   10   |     128    |                   0.869 |           442.429 s | Updated model to have smaller kernel size of 3x3 for convolution layers. Accuracy reduced a bit to 0.869                         |
|         6 | LeNet3           | 0.001      |   10   |     128    |                   0.864 |           465.583 s | Updated 3x3 kernel model by adding an additional fully connected layer. Accuracy ended at 0.864                                  |
|         7 | LeNet5_plus      | 0.001      |   10   |     128    |                   0.917 |           382.130 s | Revisited LeNet5 with 5x5 kernel for convolution layers. Added additional fully connected layer and resulted in accuracy of 0.917 |
|         8 | LeNet5_plus      | 0.005      |   10   |     128    |                   0.880 |           412.327 s | Added 25% dropout to fully connected layers. Increased learning rate to 0.005                                                        |
|         9 | LeNet5_plus      | 0.001      |   10   |     256    |                   0.866 |          6319.714 s | Removed dropouts to fully connected layers. Set learning rate back to 0.001. The training time looks way too long; I know I didn't sit and wait for 1.75 hours b/c I monitored it to completion. |
|        10 | LeNet5_plus      | 0.001      |   10   |     128    |                   0.797 |           491.281 s | Added initial convolution layer with filter [1, 1, 3, 1] to let the network learn a 1-output grayscale layer before training continues. This should be equivalent to preprocessing the images by first converting them to grayscale. |
|        11 | LeNet5_plus      | 0.001      |   10   |     128    |                   0.895 |           476.808 s | **Initialized biases with zeros instead of random_normal.** |
|        12 | LeNet5_plus      | 0.001      |   10   |     128    |                   0.745 |           503.033 s | Add 50% dropout to first fully connected layer. |
|        13 | LeNet5_plus      | 0.001      |   10   |     128    |                   0.922 |           501.152 s | Increased hidden units for the fully connected layers to 1024, 512, 256 |
|        14 | LeNet5_plus      | 0.001      |   10   |     128    |               **0.938** |           655.973 s | Increased depth of convolution layers. Added Dropout to 2nd fully connected layer fc2. Dropout rate set to 60% (keep_prob=0.4). Fully connected layers' hidden units changed to 1024, 512, 128 |
|        15 | LeNet5_plus      | 0.00075    |   10   |     128    |               **0.958** |           852.887 s | Changed learn rate to 0.00075 to slow down trainig a bit. Changed dropout to 55% (keep_prob=0.45). Added dropout to the 3rd fully connected layer. |
|        16 | LeNet5_plus      | 0.001      |   10   |     128    |               **0.965** |           778.259 s | Changed learn rate to 0.001. |




### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Images were obtained using the Street View feature in [Google Earth](https://www.google.com/earth/ "Google Earth"). Navigating through random streets of Berlin, Germany traffic signs were encountered. Once traffic signs were identified, screenshots of the signs were made. Finally, using [an online image converter](http://convert-my-image.com/ImageConverter "convert-my-image.com"), the screenshots were scaled down to 32 x 32 png images. Here are the five German traffic signs:

![alt text][ahead-only] ![alt text][children-crossing] ![alt text][speed-30] 
![alt text][turn-right-ahead] ![alt text][yield]

1. The first image might be difficult to classify because there is an extra white block in the right-half middle of the sign that might confuse the image of the arrow. This white block might be from discoloration or chipped paint from the background blue color of the sign.
2. The second image might be challenging to classify because the finer details of the figures depicting children may be a little blurry.
3. The third image might have issues due to the slight angle of the view of the image, causing the circular sign to look slightly distorted into an oval.
4. The fourth image might have some trouble because of its similarity to other signs with arrows or mutilple arrows.  
5. The fifth image might have a hard time getting classified because of the poor lighting and shade that is present in the image.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| Ahead only      		| Ahead Only   									| 
| Children crossing		| Children crossing								|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Turn right ahead      | Turn right ahead                              |
| Yield     			| Yield                                         |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.950.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

* The model predicted that the first image is an ahead only sign with a probability of 99.9%. The image contains an ahead only sign. The top five softmax probabilities were:

| Probability         	|     Prediction   	        					| 
|:----------------------|:----------------------------------------------| 
| *9.999e-01*          	| *Ahead only*      			            	| 
| 6.368e-07    			| Speed limit (60km/h)			    			|
| 2.198e-08				| Turn left ahead		        				|
| 1.592e-08    			| Go straight or right    		 				|
| 9.133e-10			    | No vehicles        					        |


* The model is almost certain that the image contains a children crossing sign with a probability of 99.9%. The image does, in fact, contain a children crossing sign. The top five softmax probabilities were:

| Probability         	|     Prediction   	        					| 
|:----------------------|:----------------------------------------------| 
| *9.999e-01*      		| *Children crossing*			            	| 
| 1.425e-06    			| Road narrows on the right		    			|
| 1.257e-06				| Dangerous curve to the right     				|
| 4.062e-07    			| Bicycles crossing		    	 				|
| 2.020e-07			    | Speed limit (60km/h)     				        |

* For the third image, the model gives a probability of 99.9% that it is a speed limit 30km/h sign. The image does contain a speed limit 30km/h sign. The top five softmax probabilities were:

| Probability         	|     Prediction   	        					| 
|:----------------------|:----------------------------------------------| 
| *9.999e-01*      		| *Speed limit (30km/h)*		            	| 
| 5.147e-07    			| Speed limit (50km/h)			       			|
| 2.842e-09				| Speed limit (70km/h)			    			|
| 2.312e-11    			| Speed limit (20km/h)		    				|
| 3.332e-13			    | Speed limit (80km/h)        			        |

* The model is 100.0% sure that the fourth image is a turn-right sign; and the image does contain a turn-right sign. The top five softmax probabilities were:

| Probability         	|     Prediction               					| 
|:----------------------|:----------------------------------------------| 
| *1.000e+00*      		| *Turn right ahead*			            	| 
| 5.897e-10    			| Keep left			    		    			|
| 2.398e-12				| Go straight or left	        				|
| 7.220e-13    			| Ahead only		    		 				|
| 4.185e-13			    | Yield                					        |

* For the fifth image, the model is very sure that this is a yield sign (probability of 100.0%). Indeed the image does contain a yield sign. The top five softmax probabilities were:

| Probability         	|     Prediction   	        					| 
|:----------------------|:----------------------------------------------| 
| *1.000e+00*     		| *Yield*						            	| 
| 8.034e-23    			| No vehicles					    			|
| 6.507e-27				| Priority road			        				|
| 1.308e-29    			| Ahead only		    		 				|
| 1.386e-30			    | No passing        					        |


