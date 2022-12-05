# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import socket

(mnist_images, mnist_labels), _ =     tf.keras.datasets.mnist.load_data(path='mnist.npz')

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

opt = tf.optimizers.Adam(0.001)

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


#checkpoint_dir = './checkpoints'
checkpoint_dir = './checkpoint_dir/checkpoint'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    train_loss(loss_value)
    train_accuracy(labels,probs)

    return loss_value, grads

for batch, (images, labels) in enumerate(dataset.take(10000)):
    loss_value, gradient = training_step(images, labels, batch == 0)

    if batch % 500 == 0:
        print('Step #%d\tLoss: %.6f\tAvg Loss: %.6f\tAccuracy: %.6f' % (batch, loss_value, \
						train_loss.result(), train_accuracy.result()))

#save checkpoint 
checkpoint.save(checkpoint_dir)
