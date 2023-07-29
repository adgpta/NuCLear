%% NuCLear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NuCLear neural network cell type classification algorithm
% Version: 4.1
% Author: Amrita Das Gupta
% Reference: https://doi.org/10.1101/2022.10.03.510670
% GitHub :  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to run: 
% % Add folder containing all scripts, workspaces and models to MATLAB
% % path!
%
% Classification Module:
% % Change directory to folder with NuCLearModels.mat to extract the models
% % for classification. Add input file directory containing the feature 
% % extraction csvs from pyradiomics feature extraction script.
% % Run NuCLearClassificationModule.m.
% 
% % Define ids for each class. Default ids: 
% % Neuron = "0"
% % Astroglia = "1"
% % Microglia = "2"
% % Oligodendroglia = "3"
% % Endothelial = "4"
% % Excitatory Neuron = "99"
% % Inhibitory Neuron = "100"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Classification Module
% Batch process all csvs from pyradiomics feature extraction in inputFileDir.
% SynthVer specifies the model to be used for classification which depends
% on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))


% Open trained model
[filename,path] = uigetfile('*.mat','Select trained model');
load(fullfile(path,filename))

clearvars filename path

% Change directory to folder with pyradiomics csv for all files
inputFileDir = uigetdir(pwd,'Select the folder with csvs containing radiomics features');
cd (inputFileDir)

% SynthVer: "Orig" for using the model without synthesized data (tableOrig).
% "Synth9" for using the model with synthesized data (tableSynth).

SynthVer = selModel(ModelInfo);

% Classify data: 
NuCLassify(inputFileDir,SynthVer,ModelInfo)

