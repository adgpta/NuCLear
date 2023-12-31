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
% Training Module:
% Designed to train any number of major classes and sub classes of cells. 
% The training is performed on all major classes for classes defined as 
% "Maj" in the variable "ClassDef".
% For classes defined as "Sub" in "ClassDef", the training is done only 
% on data for all subclasses belonging to the same major class. For eg. to 
% train the classifiers for excitatory neurons, the training will be 
% performed on all neuronal subclass data, i.e. excitatory and inhibitory. 
% The "ClassDef" variable MUST contain ALL the training data available, with 
% correct denotion of "Sub" or "Maj" with the subclass names containing 
% parts of the major class. For eg. excitatory and inhibitory neurons are 
% labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which 
% defines they belong to the "Neuron" class. "toTrain" contains all the ids for the 
% classes that needs to be trained. 
% 
% Change directory to folder with NuCLearTrainingWorkspace.mat for default ground truth dataset.
% NuCLearTrainingWorkspace.mat contains 1 variable for each cell type with
% feature extraction data from pyradiomics. The classification model may be
% trained with real dataset (tableOrig) or combined with augmented / Synthetic
% datasets (refer to python script SynthGen.py) created from the real dataset (tableSynth).
% 
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

%% Training module
% Training multiple classifiers using datasets provided as tables in
% NuCLearTrainingWorkspace. Combines all data (including synthetically
% generated data in some case) to create dataset for training. The dataset
% is divided into training, validation and test sets with a ratio of
% 70:15:15 respectively. This module saves the training models as a 
% structure and the validation accuracies for each model as an excel to the
% export path.

% Open workspace: NuCLearTrainingWorkspace.mat. 
% Loads training datasets as variables for each class and subclass.
clearvars

[filename,path] = uigetfile('*.mat','Select workspace with training data')
load(fullfile(path,filename))

clearvars filename path

varnames = who();
varnames = varnames(string(varnames) ~= 'header' & string(varnames) ~= 'varnames');


% Create ground truth table with selected datasets (label : tableOrig).
% Default structure is below.

[indx,~] = listdlg('PromptString','Select datasets for training','ListString',varnames,'ListSize',[200,150]);
TrainingDat = [];

for f = 1:length(indx)
    tempdat = eval(string(varnames(indx(f))));
    TrainingDat = cat(1,TrainingDat,tempdat);
end

TrainingDat = array2table(TrainingDat);
TrainingDat.Properties.VariableNames = header;

% Define Name, id and type of classes provided in the training data in "ClassDef". "toTrain" defines ids for the cell types to be trained. Only the
% cell types provided in "toTrain" will be used for training models.

ClassDef = ["Neuron" "0" "Maj"; "Astroglia" "1" "Maj";"Microglia" "2" "Maj";"Oligodendroglia" "3" "Maj";"Endothelial" "4" "Maj";"ExciNeuron" "99" "Sub";"InhibNeuron" "100" "Sub"];

newDef = addCldef();
if newDef ~= ""
    ClassDef = cat(1,ClassDef,string(newDef).split(' '));
end

[indx,~] = listdlg('PromptString','Select models to train','ListString',ClassDef(:,1),'ListSize',[200,150]);
toTrain = ClassDef(indx,2);


% Train Classifiers selected datasets

[ModelInfo.trainedNet.model, ModelInfo.trainedNet.accuracy,ModelInfo.trainedNet.YPred, ModelInfo.trainedNet.YTest] = AD_trainModel(TrainingDat, ClassDef,toTrain);

% Extract accuracy for trained models

accuracy_all = array2table(ModelInfo.Orig.accuracy); 
accuracy_all.Properties.VariableNames = {'Accuracy','Class','SynthVar'};
writetable(accuracy_all,strcat(export_path,'Accuracy_Comparison.xlsx'));

ModelInfo.ClassDef = ClassDef;
ModelInfo.TrainedModels = ClassDef(indx,2);

% save classifier models
uisave('ModelInfo','NuCLearModels.mat');
fprintf("\nModel saved\n")