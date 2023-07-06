%% NuCLear classification
% Batch process all csvs from pyradiomics feature extraction in inputFileDir.
% SynthVer specifies the model to be used for classification which depends
% on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))

function [] = NuCLassify(inputFileDir,SynthVer,ModelInfo)

% Select the model based on the SynthVer.
if SynthVer == "Orig"
    classNet = ModelInfo.Orig.model;
    classNet.ver = SynthVer;

elseif SynthVer == "Synth9"
    classNet = ModelInfo.Synth9.model;
    classNet.ver = SynthVer;

else
    fprintf("Please enter a valid SynthVer.")

end

fprintf("Running NuCLear classification.\n\nModel used: %s\n", SynthVer)

clearvars -except classNet inputFileDir SynthVer ModelInfo

ClassDef = ModelInfo.ClassDef;

t = string(datetime('now','Format','d-MMM-y'));

% Change directory to path containing Pyradiomics feature extraction csv files
cd (inputFileDir)
fprintf("\n\nInput file directory: %s\n", string(inputFileDir))

% Create directory to store results and set the folder as export path
mkdir(strcat('NuCLear',SynthVer,'_',t,'\'));
export_path = strcat(pwd,'\NuCLear',SynthVer,'_',t,'\');

% Select all csv files in the current directory
file = dir('*.csv');
len = size(file,1);


for ii = 1:len
    tStart = tic;

    % Extract filename, stack information
    filename = file(ii).name
    new_fl_nm = strsplit(filename,'.');
    new_fl_nm = string(new_fl_nm(1));
    fl = strsplit(new_fl_nm,'_');
    Mouse = upper(fl(1));
    Timepoint = upper(fl(2));
    Position = upper(fl(3));

    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing! %s\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>', filename);

    % Read CSV file and extract features
    path = strcat(file(ii).folder,'\');
    p_tab = readtable(strcat(path,filename));
    p_tab = p_tab(:,23:end);

    % Prepare data set for predictions and load models
    PredictionRawOrig = p_tab;
    PredictionRaw = PredictionRawOrig;

    % Load models for major cell types
    model = [];
    MajorModels = ClassDef(ClassDef(:,3) == "Maj");
    for jj = 1:size(MajorModels,1)
        ClassName = MajorModels(jj);
        model = cat(2,model,strcat("classNet.",ClassName));
    end


    % Predict cell types
    [FinalPrediction,PredictionMerged,PredictionRaw] =  NuCLearPRED(model,classNet,ClassDef,PredictionRaw,"CellType");

    
    % Load models for subclasses
    model = [];
    MajorClassType = [];
    SubInput = ClassDef(ismember(string(fieldnames(classNet)),ClassDef(ClassDef(:,3) == "Sub")));

    % Check if subclasses are present in training model
    if size(SubInput,1) > 0
        for i = 1:size(SubInput,1)
            ClassName = SubInput(i);
            ClassName = [ClassName ClassDef(contains(ClassName,ClassDef(:,1)),:)];
            MajorClassType = cat(1,MajorClassType,ClassName);
        end

        % Run prediction for all subclasses of each unique major cell type

        SubClassMerged = [];
        dCellType = unique(MajorClassType(:,2:end),'rows');
        for i = 1:size(dCellType,1)
            ClassName = dCellType(i,1);
            SubModels = MajorClassType(MajorClassType(:,2) == ClassName);
            model = strcat("classNet.",SubModels)';
            SubClassData.(ClassName) =  FinalPrediction(FinalPrediction.CellType == ClassName,:);
            [tempMerged,~,~] =  NuCLearPRED(model,classNet,ClassDef,SubClassData.(ClassName),"SubClass");
            % Vertically concatenate all classified data
            SubClassMerged = cat(1,tempMerged);
        end

        % Extract data without any subclass
        spillOver = FinalPrediction(~ismember(FinalPrediction.CellType,dCellType(:,1)),:);
        spillOver.SubClass(:,1) = "NA";

        % Merge all data and prepare for export
        FinalPrediction = cat(1,SubClassMerged,spillOver);
        FinalPrediction.Mouse(:) = Mouse;
        FinalPrediction.Timepoint(:) = Timepoint;
        FinalPrediction.Position(:) = Position;

        % Reorder table
        FinalPrediction = FinalPrediction(:,[end-2 end-1 end end-9 end-8 end-7 end-6 end-5 end-4 end-3 1:end-10]);
    else
        % Merge all data and prepare for export
        FinalPrediction.Mouse(:) = Mouse;
        FinalPrediction.Timepoint(:) = Timepoint;
        FinalPrediction.Position(:) = Position;

        % Reorder table:
        FinalPrediction = FinalPrediction(:,[end-2:end end-8:end-3 1:end-9]);
    end


    % Save predictions
    mkdir (strcat(export_path,'Predictions_raw\'))
    writetable(PredictionRaw,strcat(export_path,'Predictions_raw\',new_fl_nm,'_RawPredictions.xlsx'));


    mkdir (strcat(export_path,'Predictions_clean\'))
    writetable(PredictionMerged,strcat(export_path,'Predictions_clean\',new_fl_nm,'_CleanedPredictions.xlsx'));


    mkdir (strcat(export_path,'Predictions_results\'))
    writetable(FinalPrediction,strcat(export_path,'Predictions_results\',new_fl_nm,'_Classified.xlsx'));

    % Print file name and time
    tEnd = toc(tStart);
    fprintf('###################################################### \nProcessed! %s\nTime taken : %d minutes and %f seconds\n######################################################\n',...
        string(filename), floor(tEnd/60), rem(tEnd,60));
end