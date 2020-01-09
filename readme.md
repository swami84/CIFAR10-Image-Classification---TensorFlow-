# CIFAR 10 Image Classification

In this project, we will look at several approaches to predict labels based on image classification

We will be working with the CIFAR-10 dataset which can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html)

For best results please see [NBViewer](https://nbviewer.jupyter.org/github/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Cifar%20Image%20Classification.ipynb?flush_cache=true) to see the jupyter notebook. 



## Smallest delta model:

We will first look at a perceptual model which takes the smallest delta between a typical image for each label and typical image, where the delta is based on the human vision color index difference for each pixel.


1.We will first create a random image and then create a typical image for each class using the smallest perpetual delta
<div>
<p align="left">  <img src="https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/Random_Image.jpg"  >-------------------------------------><img src="https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/Typical_Image_Frog.jpg" align="right" ></p>
</div>

2.The second step will be to make predictions on the test set based on the closest "typical" image trained in the first step.

### Typical Images for All Classes

While some typical images such as horse, deer have some resemblence most of the other typical images have little to no resemblence to actual class. Also one can see similarity between related classes such as automobile and truck or cat and dog typical images.

![Image 1](https://github.com/swami84/Let-s-Pool-That-/blob/master/Images/Method%201_Pickups%20and%20Dropoff.png)

### Smallest Delta Model Summary

* **Smallest Delta Model reaches an accuracy of ~ 29%**
* **Large % of misclassifications between all image classes**

![Image 2](https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/Heatmap_Delta_Model_Norm.jpg)


## Fully-connected model (FC-NN)

In  this model we used a multi-layer fully-connected neural network that takes the pixel values as input and yields a class prediction as output. This NN model consists of 3 layers where first 2 layers have 300 and 150 neurons and use elu activation and the last layer has 10 (# of image classes) neurons with softmax activation.

After training the model we achieved a validation accuracy ~ 50%. 

![Image 3](https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/History_FC_Model.jpg)

### Fully-connected model Summary

* **After training with 20 epochs, we get an accuracy of ~50%. Obtained accuracy with NN approach much higher than delta model (which was ~ 28%)**
* **We still see many animals misclassified as dogs,birds. Model is also failing at dog-cat distinction**
* **Truck and automobile are also misclassified as each other**

![Image 4](https://github.com/swami84/CIFAR10-Image-Classification---TensorFlow-/blob/master/Data/Images/Norm_Heatmap_CNN_Model.jpg)

