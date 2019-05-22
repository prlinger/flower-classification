from __future__ import absolute_import, division, print_function

import os
import time

import tensorflow as tf
from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

import numpy as np

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg") # This prevents program failing on mac
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

save_dir = os.path.join('./saved_models')

train_dir = os.path.join('./data_train')
validation_dir = os.path.join('./data_validate')
test_dir = os.path.join('./data_test')

image_size = 160
batch_size = 32
IMG_SHAPE = (image_size, image_size, 3)

# Rescale all images by 1./255
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=True)

validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=True)

# Print shapes of raw images (this will be changed by the generator)
for image_batch,label_batch in train_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


def plotHistory(history):
    # Plot the learning curves:
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.show()

################################
#                              #
#  Define the benchmark models #
#                              #
################################

# Simple as in not as many parameters.  ResNet50 has depth of 168
# and 25,636,712 parameters.
def train_simple():
    model_name = "simple_bench_ResNet50_trial2"
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE,
        include_top=False
    )
    # model_name = "simple_bench_InceptionV3Model"
    # base_model = tf.keras.applications.InceptionV3(
    #     input_shape=IMG_SHAPE,
    #     include_top=False
    # )

    # freeze the base model so that it does not get trained.
    base_model.trainable = False
    for layer in base_model.layers[:-4]: # every layer except the last 4
        layer.trainable = False
    # base_model.summary()
    # exit()
    # Add new classifier layers
    # model = tf.keras.Sequential([
    #     base_model,
    #     # keras.layers.Flatten(),
    #     keras.layers.GlobalAveragePooling2D(),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(label_batch.shape[1], activation='softmax') # output in 24 categories
    # ])
    # model.summary()
    # exit()

    model = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(), # use average pooling to lose less information
        keras.layers.Dense(label_batch.shape[1], activation='softmax') # output in 24 categories
    ])

    # initialize RMSprop optimizer
    opt = keras.optimizers.RMSprop(lr=0.0001)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.summary()

    # Tensorboard for evaluation
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))

    # Training:
    epochs = 100
    steps_per_epoch = train_data.n # batch size
    validation_steps = validation_data.n # batch size

    history = model.fit_generator(
        train_data,
        steps_per_epoch = steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[tensorboard]
    )

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(validation_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    plotHistory(history)


# High end as in many parameters.  InceptionResNetV2 has depth of 572
# and 55,873,763 parameters.
def train_highend():
    model_name = "highend_bench_InceptionResNetV2"
    base_model = tf.keras.applications.InceptionResNetV2(
        input_shape=IMG_SHAPE,
        include_top=False
    )

    # freeze the base model so that it does not get trained.
    base_model.trainable = False

    # Add new classifier layers
    model = tf.keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(), # use average pooling to lose less information
        keras.layers.Dense(label_batch.shape[1], activation='softmax') # output in 24 categories
    ])

    # initialize RMSprop optimizer
    opt = keras.optimizers.RMSprop(lr=0.0001)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.summary()

    # Tensorboard for evaluation
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))

    # Training:
    epochs = 16
    steps_per_epoch = train_data.n # batch size
    validation_steps = validation_data.n # batch size

    history = model.fit_generator(
        train_data,
        steps_per_epoch = steps_per_epoch,
        epochs=epochs,
        workers=4,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[tensorboard]
    )

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(validation_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    plotHistory(history)


def main():
    # print("In main.")
    train_simple()
    # train_highend()

main()


# EOF
