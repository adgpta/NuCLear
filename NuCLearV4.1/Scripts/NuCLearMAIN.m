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

% % Training module
% Training multiple classifiers using datasets provided as tables in
% NuCLearTrainingWorkspace. Combines all data (including synthetically
% generated data in some case) to create dataset for training. The dataset
% is divided into training, validation and test sets with a ratio of
% 70:15:15 respectively. This module saves the training models as a 
% structure and the validation accuracies for each model as an excel to the
% export path.


clearvars

% Change directory to folder containing the workspace: NuCLearTrainingWorkspace.mat. 
% Loads training datasets as variables for each class and subclass.

cd ("W:\UniHeidelberg\ProjectData\Main\Classification\Matlab\Workspace\")
  
% Load training data
load NuCLearTrainingWorkspace_Revision_allCombinations.mat

% Set export path
export_path = "W:\UniHeidelberg\ProjectData\Main\Classification\Matlab\";
cd (export_path)


% Define Name, id and type of classes provided in the training data in "ClassDef". "toTrain" defines ids for the cell types to be trained. Only the
% cell types provided in "toTrain" will be used for training models.

ClassDef = ["Neuron" "0" "Maj"; "Astroglia" "1" "Maj";"Microglia" "2" "Maj";"Oligodendroglia" "3" "Maj";"Endothelial" "4" "Maj";"ExciNeuron" "99" "Sub";"InhibNeuron" "100" "Sub"];
toTrain = ["0";"1";"2";"3";"4";"99"];

% Create ground truth table with original dataset (label : tableOrig).
% Default structure is below.

tableOrig = cat(1,neurons,astroglia,microglia,oligodendroglia,endothelial,exciNeuron,inhibNeuron);
tableOrig = array2table(tableOrig);
tableOrig.Properties.VariableNames = header;

% Create ground truth table with original dataset and synthesized data (label : tableSynth). 
% Default structure is below.

tableSynth = cat(1,neurons,astroglia,microglia,oligodendroglia,endothelial,exciNeuron,inhibNeuron,NeuroSynth9,AstroSynth9,MgliaSynth9,OligSynth9,EndoSynth9,exciNeuronSynth9,inhibNeuronSynth9);
tableSynth = array2table(tableSynth);
tableSynth.Properties.VariableNames = header;

% Train Classifiers with Orig or Synth9 datasets

[ModelInfo.Orig.model, ModelInfo.Orig.accuracy,ModelInfo.Orig.YPred, ModelInfo.Orig.YTest] = AD_trainModel(tableOrig, ClassDef,toTrain);
[ModelInfo.Synth9.model, ModelInfo.Synth9.accuracy,ModelInfo.Synth9.YPred, ModelInfo.Synth9.YTest] = AD_trainModel(tableSynth, ClassDef,toTrain);

% Extract accuracy for trained models

accuracy_all = array2table(cat(1,ModelInfo.Orig.accuracy, ModelInfo.Synth9.accuracy));
accuracy_all.Properties.VariableNames = {'Accuracy','Class','SynthVar'};
writetable(accuracy_all,strcat(export_path,'Accuracy_Comparison.xlsx'));

ModelInfo.ClassDef = ClassDef;

% save classifier models
ModelName = 'NuCLearModels_neurons_exi_autoFeatures_final_2_COMBINED.mat';
save(ModelName,'ModelInfo')
fprintf("\nModel saved: %s\n",ModelName)

%% Classification Module
% Batch process all csvs from pyradiomics feature extraction in inputFileDir.
% SynthVer specifies the model to be used for classification which depends
% on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))

clearvars

% Change directory to export folder with classifier models (ModelInfo)
cd ("W:\UniHeidelberg\ProjectData\Main\Classification\Matlab\") 
load NuCLearModels_neurons_exi_autoFeatures_final_2_COMBINED.mat

% Change directory to folder with pyradiomics csv for all files
%inputFileDir =  "Z:\Aamrita_Jennifer\Amrita\Raw Data\2P data\CL-REV\PyradiomicsFeatures";
inputFileDir = "W:\UniHeidelberg\ProjectData\Main\PainM2\datastore\CSV_set_with_realigned\NuCLearFiles";    
cd (inputFileDir)

% SynthVer: "Orig" for using the model without synthesized data (tableOrig).
% "Synth9" for using the model with synthesized data (tableSynth).

SynthVer = "Synth9";

% Classify data: 
NuCLassify(inputFileDir,SynthVer,ModelInfo)

