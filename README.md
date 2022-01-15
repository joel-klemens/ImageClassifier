# ImageClassifier

 TensorFlow Image Classifier Application using the Keras Fashion MNIST dataset. 

## About Fashion MNIST

This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST.

The classes are:

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

## How it works 

1. Define a neural network layers
2. Input layer is 784 nodes for each pixel in the image 
3. Hidden layer used tensorflow relu activation function (output the input directly if positive or 0 if negative)
4. Output layer is 10 nodes resembling the classes of images 


#### Sources used 
- https://keras.io/api/datasets/fashion_mnist/ 
- https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
- https://www.youtube.com/watch?v=XIrOM9oP3pA&ab_channel=CleverProgrammer
- https://www.tensorflow.org/api_docs/python/tf/nn/relu

