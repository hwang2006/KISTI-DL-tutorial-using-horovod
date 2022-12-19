# Import Necessary Libraries

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

# Let's Import the Dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Print array size of training dataset
print("Size of Training Images: " + str(train_images.shape))
# Print array size of labels
print("Size of Training Labels: " + str(train_labels.shape))

# Print array size of test dataset
print("Size of Test Images: " + str(test_images.shape))
# Print array size of labels
print("Size of Test Labels: " + str(test_labels.shape))

# Let's see how our outputs look
print("Training Set Labels: " + str(train_labels))
# Data in the test dataset
print("Test Set Labels: " + str(test_labels))


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

from tensorflow.keras import backend as K
K.clear_session()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, steps_per_epoch=2000, epochs=20)

# Evaluating the model using the test dataset

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

