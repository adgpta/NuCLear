''''MIT License

Copyright (c) 2020 Constantin Pape

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import json
import os
from glob import glob

import imageio
import numpy as np

from csbdeep.utils import normalize
from stardist import (calculate_extents, fill_label_holes,
                      gputools_available, Rays_GoldenSpiral)
from stardist.models import Config3D, StarDist3D

import platform
import tensorflow as tf
import subprocess as sp


def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]

    COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

    return memory_used_values, memory_total_values


def use_selected_card(gpu, memory_used_values, memory_total_values):
    # Use the first free GPU
    if gpu == 0:
        array_gpu = []  # [0 for n_gpu in gpus]

        # Print the list of available GPUs
        for i in range(len(memory_total_values)):

            free_memory = memory_total_values[i] - memory_used_values[i]
            print('>> GPU' + str(i), 'available memory:', memory_total_values[i], '- Used memory:',
                  memory_used_values[i],
                  '- Free memory:', free_memory)

            if free_memory > (memory_total_values[i] * 0.02):
                array_gpu.append(i)

                # Set first available GPU in the list
                default_gpu = array_gpu[-len(array_gpu)]

            else:

                # If more GPUs are available use the last one in the list
                if len(memory_total_values) > 1:
                    array_gpu.append(i)

                    # Set last available GPU in the list
                    default_gpu = array_gpu[len(array_gpu) - 1]

                else:
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    print('>> A second Nvidia GPU is NOT available in your system!')
                    print('>> Number of GPU found', len(memory_total_values), gpus)
                    exit()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[default_gpu], True)
        # tf.config.LogicalDeviceConfiguration(memory_limit=8048)

    else:

        # Use the selected GPU
        default_gpu = gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[default_gpu], True)
        # tf.config.LogicalDeviceConfiguration(memory_limit=8048)


    # Set visible GPU device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(default_gpu)  # "0, 1, 2, 3"
    print('>> Running on GPU', str(gpus[default_gpu]))


def check_training_data(train_images, train_labels):
    train_names = [os.path.split(train_im)[1] for train_im in train_images]
    label_names = [os.path.split(label_im)[1] for label_im in train_labels]
    assert len(train_names) == len(label_names), "Number of training images and label masks does not match"
    assert len(set(train_names) - set(label_names)) == 0, "Image names and label mask names do not match"


def check_training_images(train_images, train_labels):
    assert all(im.ndim == 3 for im in train_images), "Inconsistent image dimensions"
    assert all(im.ndim == 3 for im in train_labels), "Inconsistent label dimensions"
    assert all(label.shape == im.shape
               for label, im in zip(train_labels, train_images)), "Incosistent shapes of images and labels"


def load_training_data(input_dir, image_folder, labels_folder, ext):

    # get the image and label mask paths and validate them
    image_pattern = os.path.join(input_dir, image_folder, f'*{ext}')
    print("Looking for images with the pattern", image_pattern)
    train_images = glob(image_pattern)
    assert len(train_images) > 0, "Did not find any images"
    train_images.sort()

    label_pattern = os.path.join(input_dir, labels_folder, f'*{ext}')
    print("Looking for labels with the pattern", image_pattern)
    train_labels = glob(label_pattern)
    assert len(train_labels) > 0, "Did not find any labels"
    train_labels.sort()

    check_training_data(train_images, train_labels)

    # normalization parameters: lower and upper percentile used for image normalization
    # maybe these should be exposed
    lower_percentile = 1
    upper_percentile = 99.8
    ax_norm = (0, 1, 2)

    train_images = [imageio.volread(im) for im in train_images]
    train_labels = [imageio.volread(im) for im in train_labels]
    check_training_images(train_images, train_labels)
    train_images = [normalize(im, lower_percentile, upper_percentile, axis=ax_norm)
                    for im in train_images]
    train_labels = [fill_label_holes(im) for im in train_labels]

    return train_images, train_labels


def make_train_val_split(train_images, train_labels, validation_fraction):
    n_samples = len(train_images)

    # we do train/val split with a fixed seed in order to be reproducible
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    #n_val = max(1, int(validation_fraction * n_samples))
    n_val = max(1, int(float(validation_fraction) * n_samples))
    train_indices, val_indices = indices[:-n_val], indices[-n_val:]
    x_train, y_train = [train_images[i] for i in train_indices], [train_labels[i] for i in train_indices]
    x_val, y_val = [train_images[i] for i in val_indices], [train_labels[i] for i in val_indices]

    return x_train, y_train, x_val, y_val


# TODO add more augmentations and refactor this so it can be used elsewhere
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MORE AUGMENTATIONS IS A VERY GOOD IDEA
# We could think to use somethink like this: https://github.com/mdbloice/Augmentor

###################________New Augmenter________###################### 
# The augmenter applies 3 augmentation methods on data. 
# Flip random rotations, flips, and intensity changes, which are typically sensible for (3D) microscopy images.


def random_flips_and_rotations(x, y, axis=None): 
    if axis is None:
        axis = tuple(range(y.ndim))
    axis = tuple(axis)
    assert x.ndim>=y.ndim
    #1. Rotation
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(y.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    x = x.transpose(transpose_axis + tuple(range(y.ndim, x.ndim))) 
    y = y.transpose(transpose_axis) 
    #2. Flips
    for ax in axis: 
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=ax)
            y = np.flip(y, axis=ax)
    return x, y 

#(random_intensity_change)
def random_uniform_noise(x):
    #3. Intensity                   
    x = x*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return x

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_flips_and_rotations(x, y, axis=(1,2))
    x = random_uniform_noise(x)
    return x, y

#--------------------------------------------------------

# we leave n_rays at the default of 32, but may want to expose this as well
def train_model(x_train, y_train,
                x_val, y_val,
                save_path,
                patch_size,
                anisotropy,
                n_rays=32,
                no_epochs=100,
                steps_per_epoch=400):

    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
    # make the model config
    # copied from the stardist training notebook, this is a very weird line ...
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        # GPU COULD BE EXPOSED IN THE GUI
        # MAYBE USEFULL IN CASE OF MULTIPLE GPUS SYSTEMs
        gpu = 0

        memory_used_values, memory_total_values = get_gpu_memory()
        print('>> Running on:', platform.system(), 'OS - GPU is enabled!')

        # make the model config
        # copied from the stardist training notebook, this is a very weird line ...
        # use_gpu = False and gputools_available()
        # predict on subsampled image for increased efficiency
        grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
        config = Config3D(
            rays=rays,
            grid=grid,
            use_gpu=use_selected_card(gpu, memory_used_values, memory_total_values),
            n_channel_in=1,
            train_patch_size=patch_size,
            train_epochs=no_epochs,
            train_steps_per_epoch=steps_per_epoch,
            anisotropy=anisotropy
        )

    # Mac OS M1
    elif platform.system() == 'Darwin':
        print('>> Found tensorflow-macos version', tf.__version__)

        # GPU COULD BE EXPOSED IN THE GUI
        # MAYBE USEFULL IN CASE OF MULTIPLE GPUS SYSTEMs
        gpu = False or gputools_available()

        #memory_used_values, memory_total_values = get_gpu_memory()
        #print('>> Running on:', platform.system(), 'OS - GPU is enabled!')

        grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
        config = Config3D(
            rays=rays,
            grid=grid,
            use_gpu=gpu,
            n_channel_in=1,
            train_patch_size=patch_size,
            train_epochs=no_epochs,
            train_steps_per_epoch=steps_per_epoch,
            anisotropy=anisotropy
        )

    else:
        print('>> OS Not Supported!')
        exit()

    save_root, save_name = os.path.split(save_path)
    os.makedirs(save_root, exist_ok=True)
    model = StarDist3D(config, name=save_name, basedir=save_root)

    model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augmenter, epochs=no_epochs)
    optimal_parameters = model.optimize_thresholds(x_val, y_val)
    return optimal_parameters


def compute_anisotropy_from_data(data):
    extents = calculate_extents(data)
    anisotropy = tuple(np.max(extents) / extents)
    return anisotropy


def train_stardist_model(input_dir, model_save_path, image_folder, labels_folder, ext,
                         validation_fraction, patch_size, anisotropy, n_rays, no_epochs, steps_per_epoch):
    print("Loading training data")
    train_images, train_labels = load_training_data(input_dir, image_folder, labels_folder, ext)
    print("Found", len(train_images), "images and label masks for training")

    x_train, y_train, x_val, y_val = make_train_val_split(train_images, train_labels,
                                                          validation_fraction)
    if anisotropy is None:
        anisotropy = compute_anisotropy_from_data(y_train)
        print("Anisotropy factor was computed from data:", anisotropy)

    print("Made train validation split with validation fraction", validation_fraction, "resulting in")
    print(len(x_train), "training images")
    print(len(y_train), "validation images")

    print("Start model training ...")
    print("You can connect to the tensorboard by typing 'tensorboaed --logdir=.' in the folder where the training runs")
    optimal_parameters = train_model(x_train, y_train, x_val, y_val, model_save_path, patch_size, anisotropy, n_rays, no_epochs,
                steps_per_epoch)
    print("The model has been trained and was saved to", model_save_path)
    print("The following optimal parameters were found:", optimal_parameters)


# use configarparse?
# TODO set batch size
# TODO enable fine-tuning on pre-trained
# TODO enable excluding images by name
def main():
    parser = argparse.ArgumentParser(description="Train a 3d stardist model")
    parser.add_argument('-i','--input_dir', type=str, help="input folder with folders for the training images and labels.")
    parser.add_argument('-m', '--model-name', type=str,
                        help='models name in the models directory')
    parser.add_argument('-n', '--models-dir', type=str,
                        help='directory where models are loaded')
    parser.add_argument('--image-folder', type=str, default='images',
                        help="Name of the folder with the training images, default: images.")
    parser.add_argument('--labels-folder', type=str, default='labels',
                        help="Name of the folder with the training labels, default: labels.")
    parser.add_argument('--ext', type=str, default='.tif', help="Image file extension, default: .tif")
    parser.add_argument('--validation-fraction', type=float, default=.1,
                        help="The fraction of available data that is used for validation, default: .1")
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                        help="Size of the image patches used to train the network, default: 128, 128, 128")
    aniso_help = """Anisotropy factor, needs to be passed as json encoded list, e.g. \"[.05,0.5,0.5]\".
                    If not given, will be computed from the dimensions of the input data, default: None"""
    parser.add_argument('--anisotropy', type=str, default=None,
                        help=aniso_help)

    args = parser.parse_args()
    anisotropy = args.anisotropy
    if anisotropy is not None:
        anisotropy = json.loads(anisotropy)
    model_path = os.path.join(args.models_dir, args.model_name)

    train_stardist_model(args.input_dir, model_path,
                         args.image_folder, args.labels_folder,
                         args.ext, args.validation_fraction,
                         tuple(args.patch_size), anisotropy)


if __name__ == '__main__':
    main()
