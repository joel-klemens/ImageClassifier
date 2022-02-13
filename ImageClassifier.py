import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# load dataset (https://keras.io/api/datasets/fashion_mnist/)
# This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories to train a model, along with a test set of 10,000 images.
# This dataset can be used as a drop-in replacement for MNIST.
fashion_mnist = keras.datasets.fashion_mnist

# Get data from dataset
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# show one image from the set using pyplot
# first image is of an ankle boot
print(train_labels[0])
print(train_labels[1])
print(train_labels[2])
print(train_labels[3])
print(train_labels[4])

plt.imshow(train_images[3], cmap='gray', vmin=0, vmax=255)
plt.show()

# define neural net structure
model = keras.Sequential([
    # input layer
    # take in a 28x28 image, flatten into a single layer (784x1) which is the input layer
    # each pixel from an image forms one node in the input layer
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer
    # start with just one hidden layer then evaluate performance
    # dense means each node in layer is connected to all nodes in other layers
    # hidden layer is 128 nodes, relu will retuns the value or 0
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output layer
    # the output will be a number 0-10 to match corresponding piece of clothing
    # dense means each node in layer is connected to all nodes in other layers
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
# Optimizer makes adjustments to node connection weights
# sparse_categorical_crossentropy (scce) produces a category index of the most likely matching category.
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy', metrics='accuracy')

# Train the model using the training data
# start with 5 epochs (this is pretty optimal as outlined in project)
model.fit(train_images, train_labels, epochs=5)

# Test model using testing data
test_loss = model.evaluate(test_images, test_labels)

# Predictions
predictions = model.predict(test_images)

# This is going to show a list of 10 probabilities for each category of clothing
# Test image 0 is of a ankle boot label 9
# example list: 1.2109062e-27 8.6840383e-38 0.0000000e+00 6.5932940e-25 0.0000000e+00
# 1.4666915e-02 0.0000000e+00 1.9368279e-01 8.4207612e-19 7.9165024e-01
print(predictions[0])
# show the index of the biggest number in the list, should be the label
print(list(predictions[0]).index(max(predictions[0])))
