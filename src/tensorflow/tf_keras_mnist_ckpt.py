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

import argparse
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import os
import horovod


parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/apps/applications/singularity_images/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/apps/applications/singularity_images/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=25,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()


# Horovod: initialize Horovod.
hvd.init()

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#(mnist_images, mnist_labels), _ = \
#    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())
(mnist_images, mnist_labels), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

valdata = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(y_test, tf.int64))
)
valset = valdata.batch(128)

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
if hvd.rank() == 0:
  for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break
  
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

print(f'************************* resume_from_epoch: {resume_from_epoch}')

#if resume_from_epoch > 0 and hvd.rank() == 0:
#      mnist_model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
if resume_from_epoch > 0: 
	mnist_model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
else: 
  mnist_model = tf.keras.Sequential([
    #inputs = tf.keras.layers.Input(shape=(28,28,1)),
    #tf.keras.layers.Conv2D(32, [3, 3], activation='relu')(inputs),
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  # Horovod: adjust learning rate based on number of GPUs.
  #scaled_lr = 0.001 * hvd.size()
  opt = tf.optimizers.Adam(scaled_lr)

  # Horovod: add Horovod DistributedOptimizer.
  opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

  # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
  # uses hvd.DistributedOptimizer() to compute gradients.
  mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #hvd.callbacks.BroadcastGlobalVariablesCallback(0),

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
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5',period=5,verbose=1))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0
#verbose = 1

#Horovod: broadcast initial variable states from rank 0 to all other processes.
#horovod.tensorflow.broadcast_variables(mnist_model.variables, root_rank=0)
#horovod.tensorflow.broadcast_variables(mnist_model.optimizer.variables(), root_rank=0)
#horovod.tensorflow.broadcast_variables(opt.variables())
if resume_from_epoch == 0:
    horovod.tensorflow.broadcast_variables(mnist_model.variables, 0)
    horovod.tensorflow.broadcast_variables(mnist_model.optimizer.variables(), 0) 
#else: 
    #horovod.tensorflow.broadcast_variables(mnist_model.variables, 0)
    #horovod.tensorflow.broadcast_variables(mnist_model.optimizer.variables(), 0) # missing rank & broadcast error 

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
mnist_model.fit(dataset, validation_data = valset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=args.epochs, initial_epoch=resume_from_epoch, verbose=verbose)


# Evaluate the model
loss, acc = mnist_model.evaluate(valset)
print(f'before checkpoint: loss: {loss:.4f}, acc: {acc:.4f}')

ckpt_model = tf.keras.models.load_model("checkpoint-10.h5")
valset = valdata.shuffle(2000).batch(128)
#score = hvd.allreduce(ckpt_model.evaluate(valset, verbose=verbose))
#print("********************* the saved model loaded from a checkpoint")
score = hvd.allreduce(ckpt_model.evaluate(valset, verbose=verbose))
#score = ckpt_model.evaluate(valset, verbose=verbose)
print(f'after checkpoint: loss: {score[0]:.4f}, acc: {score[1]:.4f}')
