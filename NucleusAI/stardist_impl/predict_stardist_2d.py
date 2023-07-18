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
import os
from glob import glob

import imageio
from tqdm import tqdm

from csbdeep.utils import normalize
from csbdeep.data import PercentileNormalizer
import numpy as np
from stardist.models import StarDist2D
import metrics
from mask_overlay_func import overlay
import platform

def get_image_files(input_dir, ext):
    # get the image paths and validate them
    image_pattern = os.path.join(input_dir, 'images', f'*{ext}')
    print("Looking for images with the pattern", image_pattern)
    images = glob(image_pattern)
    assert len(images) > 0, "Did not find any images"
    images.sort()

    return images

def get_mask_files(input_dir, ext):
    # get the mask paths and validate them
    label_pattern = os.path.join(input_dir, 'masks', f'*{ext}')
    print("Looking for labels with the pattern", label_pattern)
    labels = glob(label_pattern)
    assert len(labels) > 0, "Did not find any labels"
    labels.sort()

    return labels

# could be done more efficiently, see
# https://github.com/hci-unihd/batchlib/blob/master/batchlib/segmentation/stardist_prediction.py
def run_prediction(image_files, model_path, output_dir, multichannel, memory_usage):


    # load the model
    model_root, model_name = os.path.split(model_path.rstrip('/'))
    model = StarDist2D(None, name=model_name, basedir=model_root)

    os.makedirs(output_dir, exist_ok=True)

    # normalization parameters: lower and upper percentile used for image normalization
    # maybe these should be exposed
    lower_percentile = 1
    upper_percentile = 99.8
    ax_norm = (0, 1)  # independent normalization for multichannel images

    for im_file in tqdm(image_files, desc="run stardist prediction"):
        if multichannel:
            im = imageio.volread(im_file).transpose((1, 2, 0))
        else:
            im = imageio.imread(im_file)

        if memory_usage == 100:
            print('Prediction on 100% Memory. No memory reduction')
            im = normalize(im, lower_percentile, upper_percentile, axis=ax_norm)
            pred, _ = model.predict_instances(im, n_tiles=model._guess_n_tiles(im))

        else:
            print('Memory usage of ', memory_usage, '%')
            memory_reduction = memory_usage/100
            if len(model.config.axes)>3:
                print(f'Warning: Model {model.config.axes} axes configutation doesn\'t match image dimensions {im.shape}. Using ZYX from ax_norm={ax_norm}')
                axes ='ZYX'
            else:
                axes = model.config.axes
            normalizer = PercentileNormalizer(lower_percentile, upper_percentile)
            memory_reduction = memory_reduction/100
            block_size = [int(memory_reduction*s) for s in im.shape]
            min_overlap= [int(0.1*b) for b in block_size]
            context= [int(0.3*b) for b in block_size]

            for size, bz, mo, c in zip(im.shape, block_size, min_overlap, context):
                assert 0 <= mo + 2 * c < bz <= size, '0 <= min_overlap + 2 * context < block_size <= size'
                assert bz > 0, 'block_size > 1'
            print(f'min_overlap = {min_overlap} context = {context}, block_size = {block_size}, size ={im.shape}')
            from tempfile import mkdtemp
            import os.path as path
            filename = path.join(mkdtemp(), 'temp_labels.dat')
            print("created temporal label prediction file at: ", filename)
            labels_out = np.memmap(filename, dtype='int32', mode='w+', shape=im.shape)
            pred, _ = model.predict_instances_big(im,
                                                  axes=axes,
                                                  block_size=block_size,
                                                  min_overlap=min_overlap,
                                                  context=context,
                                                  n_tiles=model._guess_n_tiles(im),
                                                  normalizer=normalizer,
                                                  labels_out=labels_out
                                                  )
            size = im.shape
            del im
            pred = np.fromfile(filename, np.int32).reshape(size)
            os.remove(filename)
            if os.path.exists(filename):
                print('Warning: file still exists delete it manually.: ', filename)
            labels_out.flush()


        im_name = os.path.split(im_file)[1]
        save_path = os.path.join(output_dir, im_name)
        imageio.imsave(save_path, pred)

#######------------VALIDATION PART TEST PHASE-------------###
def run_validation(image_files, mask_files, model_path, output_dir, multichannel, memory_usage):

    # load the model
    model_root, model_name = os.path.split(model_path.rstrip('/'))
    model = StarDist2D(None, name=model_name, basedir=model_root)
    #model = StarDist2D.from_pretrained('2D_paper_dsb2018')

    os.makedirs(output_dir, exist_ok=True)

    # normalization parameters: lower and upper percentile used for image normalization
    # maybe these should be exposed
    lower_percentile = 1
    upper_percentile = 99.8
    ax_norm = (0, 1)  # independent normalization for multichannel images

    scores_dict = {'image_name': [],'iou': [], 'precion': [], 'recall': [], 'accuracy': [], 'dice':[], 'auc': []}

    for im_file in tqdm(image_files, desc="run stardist prediction"):
        if multichannel:
            im = imageio.volread(im_file).transpose((1, 2, 0))
        else:
            im = imageio.imread(im_file)

        if memory_usage == 100:
            print('Prediction on 100% Memory. No memory reduction')
            im = normalize(im, lower_percentile, upper_percentile, axis=ax_norm)
            pred, _ = model.predict_instances(im, n_tiles=model._guess_n_tiles(im))

        else:
            print('Memory usage of ', memory_usage, '%')
            memory_reduction = memory_usage/100
            if len(model.config.axes)>3:
                print(f'Warning: Model {model.config.axes} axes configutation doesn\'t match image dimensions {im.shape}. Using ZYX from ax_norm={ax_norm}')
                axes ='ZYX'
            else:
                axes = model.config.axes
            normalizer = PercentileNormalizer(lower_percentile, upper_percentile)
            memory_reduction = memory_reduction/100
            block_size = [int(memory_reduction*s) for s in im.shape]
            min_overlap= [int(0.1*b) for b in block_size]
            context= [int(0.3*b) for b in block_size]

            for size, bz, mo, c in zip(im.shape, block_size, min_overlap, context):
                assert 0 <= mo + 2 * c < bz <= size, '0 <= min_overlap + 2 * context < block_size <= size'
                assert bz > 0, 'block_size > 1'
            print(f'min_overlap = {min_overlap} context = {context}, block_size = {block_size}, size ={im.shape}')
            from tempfile import mkdtemp
            import os.path as path
            filename = path.join(mkdtemp(), 'temp_labels.dat')
            print("created temporal label prediction file at: ", filename)
            labels_out = np.memmap(filename, dtype='int32', mode='w+', shape=im.shape)
            pred, _ = model.predict_instances_big(im,
                                                  axes=axes,
                                                  block_size=block_size,
                                                  min_overlap=min_overlap,
                                                  context=context,
                                                  n_tiles=model._guess_n_tiles(im),
                                                  normalizer=normalizer,
                                                  labels_out=labels_out
                                                  )
            size = im.shape
            #del im
            pred = np.fromfile(filename, np.int32).reshape(size)
            os.remove(filename)
            if os.path.exists(filename):
                print('Warning: file still exists delete it manually.: ', filename)
            labels_out.flush()

        im_name = os.path.split(im_file)[1]
        save_path = os.path.join(output_dir, im_name)
        imageio.imsave(save_path, pred)
        maskfile = ' '.join([str(f) for f in mask_files if im_name in f])
        mask_1 = overlay.imread(maskfile)

        iou_1 = np.round(metrics.cal_iou(mask_1, pred), 2)
        prec, recall, acc, dice, auc = metrics.get_accuracy(mask_1, pred)

        scores_dict['image_name'].append(im_name)
        scores_dict['iou'].append(iou_1)
        scores_dict['precion'].append(prec)
        scores_dict['recall'].append(recall)
        scores_dict['accuracy'].append(acc)
        scores_dict['dice'].append(dice)
        scores_dict['auc'].append(auc)
        
    return scores_dict

#######------------VALIDATION PART TEST PHASE-------------###
def validate_stardist(model_path, input_dir, output_dir, ext, multichannel, memory_usage):
    print("Loading images")
    if os.path.isdir(input_dir):
        val_image_files = get_image_files(input_dir, ext)
        val_labels = get_mask_files(input_dir, ext)
    else:
        val_image_files = [input_dir]
        val_labels = [input_dir]

    print("Found", len(val_image_files), "images for validation")
    print("Found", len(val_labels), "masks for validation")
    print("Start validation ...")
    scores_dict = run_validation(val_image_files, val_labels, model_path, output_dir, multichannel, memory_usage)
    print("Finished validation")
    #return list of ious for each true_mask and pred_mask.
    return scores_dict
#######------------END VALIDATION PART----------------########

def predict_stardist(model_path, input_dir, output_dir, ext, multichannel, memory_usage):
    print("Loading images")
    if os.path.isdir(input_dir):
        image_files = get_image_files(input_dir, ext)
    else:
        image_files = [input_dir]
    print("Found", len(image_files), "images for prediction")
    print("Start prediction ...")
    run_prediction(image_files, model_path, output_dir, multichannel, memory_usage)
    print("Finished prediction")


def main():
    parser = argparse.ArgumentParser(description="Predict new images with a stardist model")
    parser.add_argument('-i', '--input-dir', type=str, help="input directory contains input images.")
    parser.add_argument('-m', '--model-name', type=str,
                        help='models name in the models directory')
    parser.add_argument('-n', '--models-dir', type=str,
                        help='directory where models are loaded')
    parser.add_argument('-o', '--output-dir', type=str, default='Null',
                        help='output directory where the predicted images are saved')
    parser.add_argument('--ext', type=str, default='.tif', help="Image file extension, default: .tif")
    parser.add_argument('--multichannel', action='store_true', help="Do we have multichannel images? Default: 0")
    parser.add_argument('-r', '--memory-memory_usage', type=int, default=100,
                        help="Memory memory_usage percentage. Defaults 100 %")

    args = parser.parse_args()
    model_path = os.path.join(args.models_dir,args.model_name)
    predict_stardist(model_path, args.input_dir, args.output_dir, args.ext, args.multichannel, args.memory_usage)

if __name__ == '__main__':
    main()
