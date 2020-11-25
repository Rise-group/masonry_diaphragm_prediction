import configparser
import os
import pathlib
import random
from glob import glob

import imgaug as ia
import keras_metrics as km
import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters as iaa
from keras import backend as K
from keras.applications import VGG16, VGG19, InceptionV3, ResNet50, Xception, MobileNetV2

from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.applications.inception_v3 import preprocess_input as inceptionV3_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout
from keras.layers.merge import add, average, concatenate, multiply
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adam
from keras.preprocessing import image as krs_image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from sklearn.utils import class_weight
from skimage import exposure


# Numpy seed
np.random.seed(73)


def setup_dirs(models_dir, logs_dir, networks_list):
    """Creates output directories to save models and logs for each network used.

    Arguments:
        models_dir {string} -- Path to the output directory for models
        logs_dir {string} -- Path to the output directory for logs
        networks_list {list} -- List of strings for each network
    """
    for net in networks_list:
        os.makedirs(f'{models_dir}/{net}', exist_ok=True)

    os.makedirs(f'{logs_dir}', exist_ok=True)


def sometimes(aug): return iaa.Sometimes(0.5, aug)


# Create preprocessing pipeline
preprocessing = iaa.Sequential(
    [
        # Apply the following to most images
        iaa.Fliplr(0.6),
        iaa.GaussianBlur(0.7),
        # Crop images by -5% and 10% of height/width
        sometimes(iaa.Crop(percent=(0, 0.3))),

        iaa.ChannelShuffle(0.4, [1, 1, 1]),
        iaa.OneOf([
            iaa.GammaContrast(gamma=(0.5, 1.75), per_channel=True),
            iaa.LogContrast(gain=(0.5, 1.0), per_channel=True),
            iaa.LinearContrast(alpha=(0.3, 1.75), per_channel=True)
        ]),
        iaa.SomeOf((1, 2),
                   [
            iaa.Affine(rotate=(-30, 30)),
            iaa.Affine(shear=(-30, 30))

        ],
            random_order=True),
    ])


def get_image_generator(network, images_dir, split, *args):
    """Creates a custom image generator and applies the preprocessing pipeline.

    Arguments:
        images_dir {string} -- Path to images directory where splits are saved in different subdirs.
        split {string} -- Data split used from the images directory

    Returns:
        [tuple] -- Number of images, number of classes and a custom generator for Keras
    """
    img_width, img_height, batch_size = args

    image_file_list = glob(f'{images_dir}/{split}/**/*.JPG', recursive=True)
    dirs = sorted(os.listdir(f'{images_dir}/{split}'))
    num_classes = len(dirs)
    num_images = len(image_file_list)
    classes = {v: i for i, v in enumerate(dirs)}
    random.shuffle(image_file_list)

    datagen = ImageDataGenerator()

    def image_generator(images_list, batch_size):
        i = 0
        while True:
            batch = {'images_path': [], 'images': [], 'labels': []}
            for b in range(batch_size):
                if i == len(images_list):
                    i = 0
                    random.shuffle(images_list)

                # Load image from path
                image_path = images_list[i]
                image = krs_image.load_img(
                    image_path, target_size=(img_height, img_width))
                image = krs_image.img_to_array(image)
                image = exposure.rescale_intensity(image, in_range=(0, 255))

                # Get label from path
                label = classes[image_path.split('/')[-2]]

                batch['images_path'].append(image_path)
                batch['images'].append(image)
                batch['labels'].append(label)
                i += 1

            batch['images'] = np.array(batch['images'], dtype=np.float)
            if split == 'train':
                batch['images'] = preprocessing.augment_images(batch['images'])
            batch['labels'] = np.eye(len(dirs))[batch['labels']]

            yield batch['images'], batch['labels']

    generator = image_generator(image_file_list, batch_size)

    return num_images, num_classes, generator


def get_cnn_model(network, input_shape, main_input, *args):
    """
    Returns a convolutional neural network model with imagenet weights.

    Arguments:

    network : string
              Name of a predefined network must be implemented.

    input_shape : tuple
                  Three values with image width, height, and channels
                  (img_width, img_height, channels)

    main_input : Input
                 Input object using the input shape defined, 
                 redundancy for a bug in keras.

    """
    args = {'input_shape': input_shape, 'weights': 'imagenet',
            'include_top': False, 'input_tensor': main_input}

    models = {
        'inceptionV3': InceptionV3(**args),
        'vgg16': VGG16(**args),
        'vgg19': VGG19(**args),
        'xception': Xception(**args),
        'resnet50': ResNet50(**args)
    }

    base_model = models[network]
    if network == 'inceptionV3':
        return base_model, 249
    else:
        return base_model, len(base_model.layers)


def get_image_model(network, num_classes, img_width, img_height):
    """Returns a model that uses only images as input.

    Arguments:
        network {string} -- Name of the network to be used pretrained on imagenet
        num_classes {int} -- Number of classes in the dataset
        img_width {int} -- Image width
        img_height {int} -- Image height

    Returns:
        Keras Model -- Keras Model ready to compile.
    """
    input_shape = (img_width, img_height, 3)
    image_input = Input(shape=input_shape)
    base_model, last_layer_number = get_cnn_model(
        network, input_shape, image_input)

    #base_model.trainable = True

    # adding regularization
    #regularizer = l2(0.01)

    #for layer in base_model.layers:
    #    for attr in ['kernel_regularizer']:
    #        if hasattr(layer, attr):
    #            setattr(layer, attr, regularizer)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(base_model.input, predictions), last_layer_number


def get_callback_list(network, path, models_dir, logs_dir, patience=40):
    """
    Returns a list of parameters for training in keras.

    Arguments
        network : string
            Name of an implemented network
        path : string
            Filename to store the logs and models while training
        models_dir : string
            Path to folder where models are saved.
        logs_dir : string
            Path to folder where logs are saved.
    """
    callback_list = [
        ModelCheckpoint(
            f'{models_dir}/{network}/{path}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1, min_delta=0.001),
        #ReduceLROnPlateau(monitor='val_loss', patience=patience//2, verbose=1),
        TensorBoard(log_dir=f'{logs_dir}/{network}/{path}')
    ]
    return callback_list


def train_on_images(network, images_dir, *args):
    """
    Trains a convolutional neural network on images from images_dir.

    Arguments:
        network : string
            Name of an implemented CNN on current keras version.
        images_dir : string
            Path to a directory with subdirs for each image class.
    """
    # Extract parameters from args
    img_width, img_height, batch_size, lr_rate, epochs, models_dir, logs_dir, gpu_number = args

    # Get image generators
    num_images_train, num_classes_train, train_gen = get_image_generator(
        network, images_dir, 'train', img_width, img_height, batch_size)

    num_images_val, num_classes_val, val_gen = get_image_generator(
        network, images_dir, 'val', img_width, img_height, batch_size)

    assert num_classes_train == num_classes_val

    # Create class weights, useful for imbalanced datasets
    class_weights = {0: 1, 1: 2}
    # Get image model
    model, last_layer_number = get_image_model(
        network, num_classes_train, img_width, img_height)

    # Create path to save training models and logs
    top_weights_path = f'A_{network}'

    # Use a multi-gpu model if available and configured
    if gpu_number > 1:
        model = multi_gpu_model(model, gpus=gpu_number)

    # Compile model and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_precision(),
            km.categorical_recall(),
        ])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network,
        top_weights_path,
        models_dir,
        logs_dir,
    )

    # Train the model on train split, for half the epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)

    # Load the best model from previous training phase
    model.load_weights(f'{models_dir}/{network}/{top_weights_path}.h5')

    # After training for a few epochs, freeze the bottom layers, and train only the last ones.
    if last_layer_number > 0:
        for layer in model.layers[:last_layer_number]:
            layer.trainable = False
        for layer in model.layers[last_layer_number:]:
            layer.trainable = True

    # Compile model with frozen layers, and set learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adadelta(lr=lr_rate),
        metrics=[
            'accuracy',
            km.categorical_precision(),
            km.categorical_recall(),
        ])

    # Get list of training parameters in keras
    callback_list = get_callback_list(
        network,
        top_weights_path,
        models_dir,
        logs_dir,
        patience=30,
    )

    # Train the model on train split, for the second half epochs
    model.fit_generator(
        train_gen,
        steps_per_epoch=num_images_train // batch_size,
        epochs=epochs // 2,
        validation_data=val_gen,
        validation_steps=num_images_val // batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        use_multiprocessing=True)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Read image parameters
    images_dir = config.get('IMAGES', 'images_dir')
    img_width = config.getint('IMAGES', 'width')
    img_height = config.getint('IMAGES', 'height')

    # Read training parameters
    lr_rate = config.getfloat('TRAINING', 'lr_rate')
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    networks_list = config.get('TRAINING', 'cnn_network_list').split(',')
    gpu_number = config.getint('TRAINING', 'gpu_number')
    batch_size = batch_size * gpu_number

    # Read data paths
    models_dir = config.get('OUTPUT', 'models_dir')
    logs_dir = config.get('OUTPUT', 'logs_dir')

    setup_dirs(models_dir, logs_dir, networks_list)

    for network in networks_list:

        args = [
            img_width, img_height, batch_size, lr_rate, epochs, models_dir,
            logs_dir, gpu_number
        ]
        K.clear_session()
        train_on_images(network, images_dir, *args)
