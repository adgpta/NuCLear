% 5-Classifier model design
% The input dataset (dataTable) is used to train all the models. Each model has one
% nuclei type to classify which is taken as the input class and the rest
% is labelled as 'others'. In the training dataset the classes ids are
% labelled as follows by default:
% Neurons = "0"
% Astroglia = "1"
% Microglia = "2"
% Oligodendroglia = "3"
% Endothelial = "4"
% Excitatory Neuron = "99"
% inhibitory Neuron = "100"

function [Model, ModelAccuracy, ModelPred, ModelTest] = AD_trainModel(dataTable, ClassDef,toTrain)

tStart = tic;

% Extract input tablename

SynthVar = string(inputname(1));
fprintf("\n\n\nInput tablename: %s\n\n\n", SynthVar)

% Run training for all available unique ids
dataNuc = string(unique(table2array(dataTable(:,1))));
unqID = string(unique(toTrain(:,1),'stable'));

% Get class definition for the input classes
ClassDef = ClassDef(ismember(ClassDef(:,2),dataNuc, 'rows' ),:);

ModelAccuracy = [];

for ii = 1:size(unqID,1)
    ClassID = unqID(ii);
    ClassName = ClassDef(ClassDef(:,2) == ClassID);
    ClassfType = ClassDef(ii,3);

    % Check for subclasses
    
    if ClassfType == "Maj"
        subchk = ClassDef(contains(ClassDef(:,1),ClassName),:);
        featureSet = {'Nuclei','original_firstorder_Minimum','original_shape_Flatness',...
            'original_shape_Elongation','original_shape_Maximum2DDiameterRow',...
            'original_shape_Maximum3DDiameter','original_shape_MinorAxisLength','original_firstorder_10Percentile',...
            'original_glcm_JointEntropy','original_glszm_LargeAreaEmphasis',...
            'original_ngtdm_Coarseness','original_glrlm_GrayLevelNonUniformity','original_firstorder_InterquartileRange'};

    else
        subchk = ClassDef(contains(ClassName,ClassDef(:,1)),:);
        MajType = subchk(:,1);
        
        % Features for subclasses
        featureSet = {'Nuclei','original_shape_SurfaceArea','original_ngtdm_Strength','original_glcm_Correlation','original_glcm_Imc1','original_glrlm_ShortRunEmphasis',...
            'original_gldm_GrayLevelVariance','original_firstorder_Uniformity','original_glszm_ZoneVariance','original_firstorder_10Percentile',...
            'original_glcm_JointAverage','original_gldm_SmallDependenceEmphasis','original_ngtdm_Coarseness'};

    end

    % Select datasets for training
    TrainingSettemp = dataTable(~ismember(dataTable.Nuclei,double(subchk(~contains(subchk(:,3),ClassfType),2))),:);

    if ClassfType == "Maj"
         % Remove subclasses while trainining Major class
        rmNuc = ClassDef(ClassDef(:,3)=="Maj",2);
        TrainingSet =  TrainingSettemp(ismember(TrainingSettemp.Nuclei,double(rmNuc)),featureSet);

    elseif ClassfType == "Sub"
        % Select all subclasses of the same major class type while training subclasses.
        rmSub = ClassDef(contains(ClassDef(:,1),MajType),:);
        rmNuc = rmSub(rmSub(:,3)=="Sub",2);
        TrainingSet =  TrainingSettemp(ismember(TrainingSettemp.Nuclei,double(rmNuc)),featureSet);
    end
    
    fprintf("TrainingSet size: %s",string(size(TrainingSet,1)));

    % Train dataset
    [model,YTest,YPred,accuracy] = trainFeatures(TrainingSet,ClassName,ClassID,SynthVar);
    

    Model.(ClassName) = model;
    ModelPred.(ClassName) = YPred;
    ModelTest.(ClassName) = YTest;
    
    %Compile all accuracies
    ModelAccuracy = cat(1,ModelAccuracy,accuracy); 

end

tEnd = toc(tStart);

fprintf('###################################################### \nTime Elaspsed: %d minutes and %f seconds\n######################################################\n',...
    floor(tEnd/60), rem(tEnd,60));
