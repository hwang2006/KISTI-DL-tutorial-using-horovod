# Import Necessary Libraries

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Horovod: module load
import horovod.tensorflow.keras as hvd

#print(tf.__version__)

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Let's Import the Dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Print array size of training dataset
#print("Size of Training Images: " + str(train_images.shape))
# Print array size of labels
#print("Size of Training Labels: " + str(train_labels.shape))

# Print array size of test dataset
#print("Size of Test Images: " + str(test_images.shape))
# Print array size of labels
#print("Size of Test Labels: " + str(test_labels.shape))

# Let's see how our outputs look
#print("Training Set Labels: " + str(train_labels))
# Data in the test dataset
#print("Test Set Labels: " + str(test_labels))


#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

from tensorflow.keras import backend as K
K.clear_session()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)


# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
			  experimental_run_tf_function=False) 


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

#model.fit(train_images, train_labels, steps_per_epoch=2000, epochs=10)
model.fit(train_images, train_labels, steps_per_epoch=2000 // hvd.size(), callbacks=callbacks, epochs=20, verbose=verbose)


# Evaluating the model using the test dataset

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=verbose)

if verbose:
  print('\nTest loss:', test_loss)
  print('\nTest accuracy:', test_acc)


# loading and evaluating the checkpointed model using the test dataset
ckpt_model = tf.keras.models.load_model("checkpoint-20.h5")
test_loss, test_acc = ckpt_model.evaluate(test_images,  test_labels, verbose=verbose)

if verbose:
  print('\nTest loss (ckpt):', test_loss)
  print('\nTest accuracy (ckpt):', test_acc)
