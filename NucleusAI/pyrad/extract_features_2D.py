#!/usr/bin/env python

from __future__ import print_function

import collections
import csv
import logging
import os
import platform
import glob
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor


def load_image_and_masks(inputfile):
  _filenames = []
  _masks = []
  supported_extensions = ('*.tif', '*.tiff')
  if platform.system() == 'Windows':
      inputfile = os.path.normpath(inputfile)
      path_to_image = os.path.join(inputfile, 'images')
      path_to_mask  = os.path.join(inputfile, 'masks')

      for ext in supported_extensions:
          _filenames.extend(glob.glob(path_to_image +'\\' + ext)) 

      for ext in supported_extensions:
          _masks.extend(glob.glob(path_to_mask +'\\' + ext))
      #self._filenames = glob.glob(self.path_to_image +'\\' +"*.tif")
  else:
      path_to_image = os.path.join(inputfile, 'images')
      path_to_mask  = os.path.join(inputfile, 'masks')
      print('self.path_to_image',path_to_image)
      for ext in supported_extensions:
          _filenames.extend(glob.glob(path_to_image +'/' + ext))
      
      for ext in supported_extensions:
          _masks.extend(glob.glob(path_to_mask +'/' + ext))
        
  return _filenames, _masks

def check_data(imagefiles, maskfiles):
  train_names = [os.path.split(train_im)[1] for train_im in imagefiles]
  label_names = [os.path.split(label_im)[1] for label_im in maskfiles]
  assert len(imagefiles) > 0, "Did not find any images"
  assert len(maskfiles) > 0, "Did not find any masks"
  assert len(train_names) == len(label_names), "Number of training images and label masks does not match"
  assert len(set(train_names) - set(label_names)) == 0, "Image names and label mask names do not match"
  print('Data checked')

def feature_extractor(inputPath, outPath):
  outputFilepath = os.path.join(outPath, 'radiomics_features.csv') #output csv with extracted features
  progress_file = os.path.join(outPath,'pyrad_log.txt') # output pyrad logs
  params = os.path.join(inputPath, 'Params.yaml') #look for parameters file for pyradiomics

  rLogger = logging.getLogger('radiomics') # Configure logging
  handler = logging.FileHandler(filename=progress_file, mode='w') # Create handler for writing to log file
  handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
  rLogger.addHandler(handler)
  logger = rLogger.getChild('batch') # Initialize logging for batch log messages
  radiomics.setVerbosity(logging.INFO) # Set verbosity level for output to stderr (default level = WARNING)
  logger.info('pyradiomics version: %s', radiomics.__version__)

  try:
    imagefiles, maskfiles = load_image_and_masks(inputPath)
    check_data(imagefiles, maskfiles)
  except Exception:
    logger.error('IMAGES AND MASKS READ FAILED', exc_info=True)

  imagefiles.sort()
  maskfiles.sort()
  
  logger.info('Loading Done')
  logger.info('Number of Images: %d', len(imagefiles))
  logger.info('Number of Masks: %d', len(maskfiles))


  if os.path.isfile(params):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
  else:  # Parameter file not found, use hardcoded settings instead
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3]
    settings['interpolator'] = sitk.sitkBSpline
    settings['enableCExtensions'] = True
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

  logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
  logger.info('Enabled features: %s', extractor.enabledFeatures)
  logger.info('Current settings: %s', extractor.settings)

  headers = None
  for idx, (imageFilepath, maskFilepath) in enumerate(zip(imagefiles, maskfiles)):
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, 
                      len(imagefiles), os.path.basename(imageFilepath), os.path.basename(maskFilepath))

    label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = collections.OrderedDict()
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

        with open(outputFilepath, 'a') as outputFile:
          writer = csv.writer(outputFile, lineterminator='\n')
          if headers is None:
            headers = list(featureVector.keys())
            print('100 ::', headers)
            writer.writerow(headers)

          row = []
          for h in headers:
            row.append(featureVector.get(h, "N/A"))
          writer.writerow(row)
      except Exception:
        logger.error('FEATURE EXTRACTION FAILED', exc_info=True)



def main(inputPath, outPath):
    feature_extractor(inputPath, outPath)