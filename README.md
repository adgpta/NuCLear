# _NuCLear_
#### Nucleus-instructed tissue composition using deep learning - A neural network based cell type classification using nuclei features from fluorescent imaging in 3D.
#### Version: 4.1
#### Author: Amrita Das Gupta
#### Publications:
[Comprehensive monitoring of tissue composition using in vivo imaging of cell nuclei and deep learning](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1)  
Amrita Das Gupta, Livia Asan, Jennifer John, Carlo A. Beretta, Thomas Kuner, Johannes Knabbe

## Outline

NuCLear is a MLP neural network based cell type classification which can be used to train and classify cell types and their subtypes based on nuclei features obtained from fluorescent imaging as described in the [here](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1).


![1111](https://github.com/adgpta/NuCLear/assets/77382748/94f90f01-bd83-421b-849c-1970c171756c)

Our application of the NuCLear algorithm is performed on in-vivo two-photon 3D images of histone 2B-eGFP-labeled cell nuclei. Using StarDist (https://github.com/stardist/stardist) to segment the nuclei and PyRadiomics (https://github.com/AIM-Harvard/pyradiomics) to extract various features, NuCLear trains sevaral classifiers depending on the data provided by the user to classify all cells per imaging volume. Beyond in-vivo imaging, With good training data, NuCLear would be able to work with any fluorescence-based microscopy images and perform in any organ or model system. 


![3](https://github.com/adgpta/NuCLear/assets/77382748/8a7ec983-fb8e-40a8-897f-1aeec0b46bc3)


## Methodology
Using segmentation editor in Fiji, ground truth were generated by manually labelling and segmenting nuclei in 3D as below.

https://github.com/adgpta/NuCLear/assets/77382748/3a002c61-110e-45cc-9141-8a5234d33911 

https://github.com/adgpta/NuCLear/assets/77382748/9ec39476-c839-4b51-9360-e7883dacca9e


These ground truth sets (labelled and raw images) were used for training a segmentation model using [StarDist](https://github.com/stardist/stardist), in 3D via an in-house developed Jupyter Notebook.
![2](https://github.com/adgpta/NuCLear/assets/77382748/c1e4a16a-5574-4b60-adfb-3daabe800033)

Using the [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) python package, in total 107 radiomics features were extracted for each segmented nucleus after segmentation using StarDist, including, but not limited, to morphological features, intensity features and texture features. 

The features are further used to train classifier models for each cell type provided in the ground truth data. This includes major cell types and sub types. This version can classify between the major cell types of Neurons, Glia and endothelial cells, along with the neuronal subtypes: Excitatory and inhibitory neurons and glial subtypes: Astroglia, microglia and oligodendroglia.
A GUI based version for segmentation and extraction of radiomics feature can be found [here](https://github.com/SFB1158RDM/NucleusAI).

## Guide:

This repository contains the MATLAB and python implementations of NuCLear. The following describes step-by-step guide for ground truth extraction, training and classification for each case.


#### Generate ground truth data for training
To generate classifier models for each cell type, supervised training was performed using ground truth datasets from two-photon volumetric fluorescence imaging. GFP and RFP stacks were acquired (with the RFP indicating different cell types). 
![4](https://github.com/adgpta/NuCLear/assets/77382748/337adb0c-5600-4fc1-b81b-723637f049f0)

The GFP images were segmented with each nuclei being assigned an unique value and their radiomics features were extracted in a .csv file. Positive cells for each type were identified by overlaying the RFP and GFP iamges as described in the methods [here](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1). For each positive cell, the corresponding label was identified from the segemented images and all the radiomics features of the said label was extracted and saved.

![5](https://github.com/adgpta/NuCLear/assets/77382748/374a30a0-15b3-4b57-a65b-3febc61fc130)

#### Generate synthetic / augmented data
To increase the number of nuclei for training and include possible variations, synthetic datasets were created from nuclei features using the synthetic data vault (SDV) package (Patki, Wedge et al. 2016), which utilizes correlations between features of the original dataset as well as mean, minimum or maximum values and standard deviations. CSV files containing pre-generated synthetic data from the provided ground truths are available in the (folder). To create your own synthetic data follow the instructions below to run the python script:

The following scripts has been created with Python 3.9.16.

1. Clone the repository.
2. Create a virtual environment and install "requirements.txt" file using
   ```
   pip install -r requirements.txt 
   ```
   




Add folder containing all scripts, workspaces and models to MATLAB path. Run NuCLearMAIN.m. 

NuCLearMAIN.m is divided into training and the Classification module. 

### Training Module:
Designed to train any number of major classes and sub classes of cells. The training is performed on all major classes for classes defined as "Maj" in the variable "ClassDef". For classes defined as "Sub" in "ClassDef", the training is done only on data for all subclasses belonging to the same major class. For eg. to train the classifiers for excitatory neurons, the training will be performed on all neuronal subclass data, i.e. excitatory and inhibitory. The "ClassDef" variable MUST contain ALL the training data available, with correct denotion of "Sub" or "Maj" with the subclass names containing parts of the major class. For eg. excitatory and inhibitory neurons are labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which defines they belong to the "Neuron" class. "toTrain" contains all the ids for the classes that needs to be trained. 

Change directory to folder with NuCLearTrainingWorkspace.mat for default ground truth dataset. NuCLearTrainingWorkspace.mat contains 1 variable for each cell type with feature extraction data from pyradiomics. The classification model may be trained with real dataset (tableOrig) or combined with augmented / Synthetic datasets (refer to python script SynthGen.py) created from the real dataset (tableSynth). Training multiple classifiers using datasets provided as tables in NuCLearTrainingWorkspace. Combines all data (including synthetically generated data in some case) to create dataset for training. The dataset is divided into training, validation and test sets with a ratio of 70:15:15 respectively. This module saves the training models as a structure and the validation accuracies for each model as an excel to the export path.

### Classification Module:
Change directory to folder with NuCLearModels.mat to extract the models for classification. Add input file directory containing the feature extraction csvs from pyradiomics feature extraction script. Batch processes all csvs from pyradiomics feature extraction in inputFileDir. SynthVer specifies the model to be used for classification which depends on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))
