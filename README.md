# _NuCLear_
#### Nucleus-instructed tissue composition using deep learning - A neural network based cell type classification using nuclei features from fluorescent imaging in 3D.
#### Version: 4.1
#### Author: Amrita Das Gupta
#### Publications:
[Comprehensive monitoring of tissue composition using in vivo imaging of cell nuclei and deep learning](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1)  
Amrita Das Gupta, Livia Asan, Jennifer John, Carlo A. Beretta, Thomas Kuner, Johannes Knabbe

## Outline

This version of NuCLear can be used to train and classify cell types and their subtypes based on nuclei features obtained from fluorescent imaging as described in the paper.

Our application of the NuCLear algorithm is performed on in-vivo two-photon 3D images of histone 2B-eGFP-labeled cell nuclei. Using StarDist (https://github.com/stardist/stardist) to segment the nuclei and PyRadiomics (https://github.com/AIM-Harvard/pyradiomics) to extract various features, NuCLear trains sevaral classifiers depending on the data provided by the user to classify all cells per imaging volume. Beyond in-vivo imaging, With good training data, NuCLear should work with any fluorescence-based microscopy images and perform in any organ or model system. 
![1](https://github.com/adgpta/NuCLear/assets/77382748/fc7d4557-1a13-4346-950a-4753b178c36a)
#
### How to run: 
Add folder containing all scripts, workspaces and models to MATLAB path. Run NuCLearMAIN.m. 

NuCLearMAIN.m is divided into training and the Classification module. 
#
### Training Module:
Designed to train any number of major classes and sub classes of cells. The training is performed on all major classes for classes defined as "Maj" in the variable "ClassDef". For classes defined as "Sub" in "ClassDef", the training is done only on data for all subclasses belonging to the same major class. For eg. to train the classifiers for excitatory neurons, the training will be performed on all neuronal subclass data, i.e. excitatory and inhibitory. The "ClassDef" variable MUST contain ALL the training data available, with correct denotion of "Sub" or "Maj" with the subclass names containing parts of the major class. For eg. excitatory and inhibitory neurons are labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which defines they belong to the "Neuron" class. "toTrain" contains all the ids for the classes that needs to be trained. 

Change directory to folder with NuCLearTrainingWorkspace.mat for default ground truth dataset. NuCLearTrainingWorkspace.mat contains 1 variable for each cell type with feature extraction data from pyradiomics. The classification model may be trained with real dataset (tableOrig) or combined with augmented / Synthetic datasets (refer to python script SynthGen.py) created from the real dataset (tableSynth). Training multiple classifiers using datasets provided as tables in NuCLearTrainingWorkspace. Combines all data (including synthetically generated data in some case) to create dataset for training. The dataset is divided into training, validation and test sets with a ratio of 70:15:15 respectively. This module saves the training models as a structure and the validation accuracies for each model as an excel to the export path.
#
### Classification Module:
Change directory to folder with NuCLearModels.mat to extract the models for classification. Add input file directory containing the feature extraction csvs from pyradiomics feature extraction script. Batch processes all csvs from pyradiomics feature extraction in inputFileDir. SynthVer specifies the model to be used for classification which depends on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))
