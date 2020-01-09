# CIFAR 10 Image Classification

In this project, we will look at several approaches to predict labels based on image classification

We will be working with the CIFAR-10 dataset which can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html)

For best results please see [NBViewer](https://nbviewer.jupyter.org/github/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Cifar%20Image%20Classification.ipynb?flush_cache=true) to see the jupyter notebook. 



## Smallest delta model:

We will first look at a perceptual model which takes the smallest delta between a typical image for each label and typical image, where the delta is based on the human vision color index difference for each pixel.


1.We will first create a random image and then create a typical image for each class using the smallest perpetual delta

<p>  <img src="https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/Random_Image.jpg"  align="left">-------------------------------------></p>


2.The second step will be to make predictions on the test set based on the closest "typical" image trained in the first step.

### Results

* Smallest Delta Model reaches an accuracy of ~ 29%
* Large number of misclassifications between all image classes

