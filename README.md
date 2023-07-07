%% NuCLear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NuCLear neural network cell type classification algorithm
% Version: 4.1
% Author: Amrita Das Gupta
% Reference: https://doi.org/10.1101/2022.10.03.510670
% This script is divided into training and the Classification module. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to run: 
% % Add folder containing all scripts, workspaces and models to MATLAB
% % path!
%
% Training Module:
% % Designed to train any number of major classes and sub classes of cells. The training is performed on all major classes for classes defined as "Maj" in the variable "ClassDef".
% % For classes defined as "Sub" in "ClassDef", the training is done only on data for all subclasses belonging to the same major class. For eg. to train the classifiers for excitatory neurons, the training will be performed on 
% % all neuronal subclass data, i.e. excitatory and inhibitory. The "ClassDef" variable MUST contain ALL the training data available, with correct denotion of "Sub" or "Maj" with the subclass names containing parts of the major
% % class. For eg. excitatory and inhibitory neurons are labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which defines they belong to the "Neuron" class. "toTrain" contains all the ids for the classes that needs to be trained. 
% 
% % Change directory to folder with NuCLearTrainingWorkspace.mat for default ground truth dataset.
% % NuCLearTrainingWorkspace.mat contains 1 variable for each cell type with
% % feature extraction data from pyradiomics. The classification model may be
% % trained with real dataset (tableOrig) or combined with augmented / Synthetic
% % datasets (refer to python script SynthGen.py) created from the real dataset (tableSynth).
% 
% Classification Module:
% % Change directory to folder with NuCLearModels.mat to extract the models
% % for classification. Add input file directory containing the feature 
% % extraction csvs from pyradiomics feature extraction script.
% % Run AD_classify.
% 
% 
% % Defined ids for each class. Default ids: 
% % Neuron = "0"
% % Astroglia = "1"
% % Microglia = "2"
% % Oligodendroglia = "3"
% % Endothelial = "4"
% % Excitatory Neuron = "99"
% % Inhibitory Neuron = "100"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
