function [net,YTest,YPred,ClassRes] = trainFeatures(dataTable,ClassName,ClassID,SynthVar)
% dataTable is the input dataset for training. Default tables are tableOrig
% & tableSynth. "ClassName" is the class type to train the model on. "ClassID"
% defines the default IDs for each class.


% Change labels to model class and others and convert to categorical

ctg =string(table2array(dataTable(:,1)));
ctg((ctg(:,1) ~= ClassID),1) = 'Others';
ctg((ctg(:,1) == ClassID),1) = ClassName;
dataTable(:,end+1) = table(ctg);
dataTable(:,'Nuclei') = [];
dataTable.Properties.VariableNames{end} = 'Nuclei';
labelName = "Nuclei";
dataTable = convertvars(dataTable,labelName,'categorical');
classNames = categories(dataTable{:,labelName});

fprintf("\nModel training in progess: %s\n", ClassName);


%% Partition the data set into training, validation, and test partitions.
% Set aside 15% of the data for validation, and 15% for testing.
% View the number of observations in the dataset.

numObservations = size(dataTable,1);

% Determine the number of observations for each partition.
numObservationsTrain = floor(0.7*numObservations);
numObservationsValidation = floor(0.15*numObservations);
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;


% Create an array of random indices corresponding to the observations and
% partition it using the partition sizes.
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);

% Select features to use
feature_sel = dataTable;

% Partition the table of data into training, validation, and testing
% partitions using the indices.
tblTrain = feature_sel(idxTrain,:);
tblValidation = feature_sel(idxValidation,:);
tblTest = feature_sel(idxTest,:);

%% Define Network Architecture
% Define the network for classification.
%
% Define a network with a feature input layer and specify the number 
% of features. Also, configure the input layer to normalize the data 
% using Z-score normalization. Next, include a fully connected layer 
% with output size 50 followed by a batch normalization layer and a 
% ReLU layer. For classification, specify another fully connected layer 
% with output size corresponding to the number of classes, followed by 
% a softmax layer and a classification layer.

numFeatures = size(feature_sel,2) - 1;
numClasses = numel(classNames);

layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


miniBatchSize = 16;

% Options for training classifier. Change 'ExecutionEnvironment' to 'auto'
% if no gpu is available.

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',tblValidation, ...
    'Plots','training-progress', ...
    'Verbose',false,...    
    'ExecutionEnvironment', 'gpu',...% Execution using GPU. Comment out if no GPU is available.
    'plots' , 'none'); % Remove to see training plots); ;


[net,info] = trainNetwork(tblTrain,labelName,layers,options);

%% Test prediction
YPred = classify(net,tblTest(:,1:end-1),'MiniBatchSize',miniBatchSize);

% Calculate the classification accuracy. The accuracy is the proportion of 
% the labels that the network predicts correctly.
YTest = tblTest{:,labelName};

ClassRes = [(sum(YPred == YTest)./numel(YTest))*100, ClassName, SynthVar];


% Close all training windows if training plots are turned on
%pause(3)
%delete(findall(0));

fprintf("Model training completed.\nFinal training accuracy: %f\nFinal validation accuracy: %f\nFinal testing accuracy: %s\n\n\n\n",info.TrainingAccuracy(end),info.FinalValidationAccuracy,ClassRes(1,1))

