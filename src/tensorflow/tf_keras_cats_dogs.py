import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pylab as plt

from tensorflow import keras
import horovod.tensorflow.keras as hvd
import socket
import os

parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/scratch/qualis/dataset/training_set/training_set'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/scratch/qualis/dataset/test_set/test_set'),
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
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=3,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()


# Horovod: initialize Horovod.
hvd.init()
print('************* hvd.size:', hvd.size(),'hvd.rank:', hvd.rank(),\
        'hvd.local_rank:', hvd.local_rank(), 'hostname:', socket.gethostname())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print('************* gpus:', gpus, 'on', socket.gethostname())

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# 훈련 셋, 검증 셋 저장위치 지정
train_dir = "/scratch/qualis/dataset/training_set/training_set"
valid_dir = "/scratch/qualis/dataset/test_set/test_set"

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# 이미지 데이터 제너레이터 정의 (Augmentation 미적용)
image_gen = ImageDataGenerator(rescale=(1/255.))

# flow_from_directory 함수로 폴더에서 이미지 가져와서 제너레이터 객체로 정리 
train_gen = image_gen.flow_from_directory(train_dir, 
                                          batch_size=args.batch_size, 
                                          target_size=(224, 224),   
                                          classes=['cats','dogs'], 
                                          class_mode = 'binary',
                                          seed=2022)

valid_gen = image_gen.flow_from_directory(valid_dir,                                          
                                          batch_size=args.val_batch_size, 
                                          target_size=(224, 224),   
                                          classes=['cats','dogs'], 
                                          class_mode = 'binary',
                                          seed=2022)

# Sequential API를 사용하여 샘플 모델 생성

def build_model():

    model = tf.keras.Sequential([

        # Convolution 층 
        tf.keras.layers.BatchNormalization(), 
        #tf.keras.layers.BatchNormalization(input_shape=(224,224,3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classifier 출력층 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'), 
        tf.keras.layers.Dropout(0.5),              
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Dense(32, activation='relu'),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return model

model = build_model()

# Horovod: (optional) compression algorithm.
#compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
# Horovod: adjust learning rate based on number of GPUs.
#initial_lr = args.base_lr * hvd.size()
#opt = keras.optimizers.SGD(lr=initial_lr, momentum=args.momentum)
# Horovod: add Horovod Distributed Optimizer.
#opt = hvd.DistributedOptimizer(opt, compression=compression)

# Horovod: adjust learning rate based on number of GPUs.
#opt = tf.optimizers.Adam(lr=0.001 * hvd.size())

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.0015 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)


# Horovod: add Horovod Distributed Optimizer.
#opt = hvd.DistributedOptimizer(opt)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# 모델 컴파일
#model.compile(optimizer=opt, 
#              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
#              metrics=['accuracy'])


# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=opt,
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
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr,
                                             warmup_epochs=args.warmup_epochs,
                                             verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 20th, 30th and 80th epochs.
    # hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr,
    #                                            multiplier=1.,
    #                                           start_epoch=args.warmup_epochs,
    #                                           end_epoch=10),
    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-1, start_epoch=10, end_epoch=30),
    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-2, start_epoch=30, end_epoch=40),
]
# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#if hvd.rank() == 0:
#    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))

# 모델 훈련
history = model.fit(train_gen, steps_per_epoch=len(train_gen) // hvd.size(), validation_data=valid_gen, callbacks=callbacks, epochs=20, verbose=verbose)

# 손실함수, 정확도 그래프 그리기 
def plot_loss_acc(history, epoch, plot_filename="loss-acc-plot.png"):

    loss, val_loss = history.history['loss'], history.history['val_loss']
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, epoch + 1), loss, label='Training')
    axes[0].plot(range(1, epoch + 1), val_loss, label='Validation')
    axes[0].legend(loc='best')
    axes[0].set_title('Loss')

    axes[1].plot(range(1, epoch + 1), acc, label='Training')
    axes[1].plot(range(1, epoch + 1), val_acc, label='Validation')
    axes[1].legend(loc='best')
    axes[1].set_title('Accuracy')

    #plt.show()
    plt.savefig(plot_filename)

plot_loss_acc(history, 20, "plot1-hvd-%d.png" % hvd.size() )

# 이미지 데이터 제너레이터 정의 (Augmentation 적용)
image_gen_aug = ImageDataGenerator(rescale=1/255., 
                                   horizontal_flip=True,
                                   rotation_range=15,                                
                                   shear_range=0.15,
                                   zoom_range=0.15)

# flow_from_directory 함수로 폴더에서 이미지 가져와서 제너레이터 객체로 정리 
train_gen_aug = image_gen_aug.flow_from_directory(train_dir, 
                                                  batch_size=args.batch_size, 
                                                  target_size=(224,224),   
                                                  classes=['cats','dogs'], 
                                                  class_mode = 'binary', 
                                                  seed=2022)

valid_gen_aug = image_gen_aug.flow_from_directory(valid_dir,  
                                                  batch_size=args.batch_size, 
                                                  target_size=(224,224),   
                                                  classes=['cats','dogs'], 
                                                  class_mode = 'binary', 
                                                  seed=2022)

# 모델 생성
model_aug = build_model()

# 모델 컴파일
#model_aug.compile(optimizer=opt,
#              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
#              metrics=['accuracy'])

model_aug.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)


# 모델 훈련
history_aug = model_aug.fit(train_gen_aug, steps_per_epoch=len(train_gen_aug) // hvd.size(), validation_data=valid_gen_aug, callbacks=callbacks, epochs=args.epochs, verbose=verbose)

plot_loss_acc(history_aug, args.epochs,"plot2-hvd-%d.png" % hvd.size())
