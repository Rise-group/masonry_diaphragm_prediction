
import configparser
import os
from glob import glob

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import metrics
from keras.applications import VGG16, VGG19, InceptionV3, ResNet50, Xception
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import GlobalAveragePooling2D,Input
from keras.layers.core import Dense, Dropout
from keras.layers.merge import concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm

from train import (get_cnn_model, get_image_generator, get_image_model)

# Random seed
np.random.seed(73)


def get_report_new(images_dir, model, split, network, figures_dir):
    generator = ImageDataGenerator(rescale=1/255)

    data_gen = generator.flow_from_directory(
        f'{images_dir}/{split}',
        target_size=(224,224),
        batch_size=12,
        class_mode='categorical',
        shuffle=False)
    
    data_gen.reset()

    y_probs = model.predict_generator(data_gen, steps=len(data_gen), verbose=1)
    y_probs = np.round(y_probs, 2)
    y_pred = np.argmax(y_probs, axis=-1)
    y_true = data_gen.classes

    filenames = list(map(os.path.basename, data_gen.filenames))

    results = pd.DataFrame({
        'filename' : filenames,
        'true' : y_true,
        'pred' : y_pred,
        'flexible_prob' : y_probs[:, 0],
        'rigido_prob' : y_probs[:, 1]  
    })
    results.to_csv(f'{figures_dir}/{network}_{split}.csv')
    classes = ['Flexible', 'Rígido']

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm, cmap='Blues', annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.title(f'{network}_{split}'.capitalize(), fontsize=14)
    plt.savefig(f'{figures_dir}/{network}_{split}.png')


def test_on_images(network, images_dir, models_dir, *args):
    """"""
    print('Testing on images...')
    # Extract parameters from args
    img_width, img_height, batch_size, models_dir, figures_dir = args

    # Get image generators
    num_images_val, num_classes_val, val_gen = get_image_generator(
        network, images_dir, 'val', img_width, img_height, batch_size)

    num_images_test, num_classes_test, test_gen = get_image_generator(
        network, images_dir, 'test', img_width, img_height, batch_size)


    # Get image model
    model, last_layer_number = get_image_model(
        network, num_classes_val, img_width, img_height)

    path = f'{models_dir}/{network}'
    print(path)
    models_list = glob(f'{path}/A*', )

    net_id = f'{os.path.basename(models_dir)}A_'
    print('Loading model...')
    print(models_list)
    for h5_model in models_list:
        model = load_model(h5_model)
        print('Validation')
        get_report_new(images_dir, model, 'val', network, figures_dir, )
        print('Test')
        get_report_new(images_dir, model, 'test', network, figures_dir)


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

    # batch_size = batch_size * gpu_number

    # Read data paths
    models_dir = config.get('OUTPUT', 'models_dir')
    logs_dir = config.get('OUTPUT', 'logs_dir')
    figures_dir = config.get('OUTPUT', 'figures_dir')
    os.makedirs(f'{figures_dir}', exist_ok=True)

    for network in networks_list:
        K.clear_session()
        args = [img_width, img_height, batch_size, models_dir, figures_dir]

        test_on_images(network, images_dir, models_dir, *args)