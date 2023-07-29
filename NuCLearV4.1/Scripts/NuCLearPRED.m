function [Class_results_merged,Class_results_category,predictResults] =  NuCLearPRED(model,classNet,ClassDef,predictResults,VarNam)

% Run prediction for each nuclei for each model
predictResultsOrig = predictResults;

for xx = 1:size(model,2)
    nmx = model(xx).split(".");
    nmx = nmx(2);
    if ClassDef(strcmp(ClassDef(:,1),nmx),3) == "Maj"
        predictData = predictResults(:,{'original_firstorder_Minimum','original_shape_Flatness',...
            'original_shape_Elongation','original_shape_Maximum2DDiameterRow',...
            'original_shape_Maximum3DDiameter','original_shape_MinorAxisLength','original_firstorder_10Percentile',...
            'original_glcm_JointEntropy','original_glszm_LargeAreaEmphasis',...
            'original_ngtdm_Coarseness','original_glrlm_GrayLevelNonUniformity','original_firstorder_InterquartileRange'});
    elseif ClassDef(strcmp(ClassDef(:,1),nmx),3) == "Sub"

        predictData = predictResults(:,{'original_shape_SurfaceArea','original_ngtdm_Strength','original_glcm_Correlation','original_glcm_Imc1','original_glrlm_ShortRunEmphasis',...
            'original_gldm_GrayLevelVariance','original_firstorder_Uniformity','original_glszm_ZoneVariance','original_firstorder_10Percentile',...
            'original_glcm_JointAverage','original_gldm_SmallDependenceEmphasis','original_ngtdm_Coarseness'});

        %
        %                 predictData = predictResults(:,{'original_shape_MajorAxisLength','original_shape_Maximum2DDiameterColumn','original_shape_Maximum2DDiameterRow',...
        %                     'original_shape_Maximum2DDiameterSlice','original_shape_Maximum3DDiameter','original_shape_MeshVolume','original_shape_SurfaceVolumeRatio',...
        %                     'original_glcm_Idm','original_glcm_Idn','original_gldm_DependenceNonUniformity','original_glrlm_RunEntropy',...
        %                     'original_glszm_ZoneEntropy','original_glszm_GrayLevelNonUniformity','original_glszm_SmallAreaEmphasis'});

    end
    YPred_L = classify(eval(model{xx}),table2array(predictData));
    predictResults = cat(2,predictResults,table(YPred_L));
    predictResults.Properties.VariableNames(end) = model(xx);
end

%% Combine prediction for all models into one
% For each nuclei if one model is positive and the rest is negative
% (defined by "other" label), classify nuclei as positive for the said
% class. If two or more models are positive, classify said nuclei as
% "undecided". If all models are negative, classify said nuclei as
% "unclassified".


% Extract classification output and convert labels to ids
%pred = string(table2array(predictResults(:,["classNet.Neuron" "classNet.Astroglia" "classNet.Microglia" "classNet.Oligodendroglia" "classNet.Endothelial"])));
pred = string(table2array(predictResults(:,model)));

% For old model, change tags if present
try
    pred(strcmp((pred),'Neurons')) = 'Neuron';
    pred(strcmp((pred),'Astrocytes')) = 'Astroglia';
    pred(strcmp((pred),'Oligodendrocytes')) = 'Oligodendroglia';
    % fprintf("/nUsing old model. Changing tags/n");
catch
    fprintf("Using new model");
end

pred(strcmp((pred),'Neuron')) = '1';
pred(strcmp((pred),'ExciNeuron')) = '99';
pred(strcmp((pred),'InhibNeuron')) = '100';
pred(strcmp((pred),'Astroglia')) = '2';
pred(strcmp((pred),'Microglia')) = '3';
pred(strcmp((pred),'Oligodendroglia')) = '4';
pred(strcmp((pred),'Endothelial')) = '5';
pred(strcmp((pred),'Others')) = '999';

pred = double(pred);

% Take all unique values from each cell
CellPred = num2cell(pred,2);
UniqueVal = cellfun(@unique,CellPred,'UniformOutput',false);

% Padding with zeros if number of elements is less than maximum
% elements in UniqueVal
idx.size = cellfun(@length,UniqueVal);
idx.padded = max(idx.size)-idx.size;
M = max(idx.size); % new length should be multiple of M
newN = M * ceil(idx.size / M); % new lengths of cell elements
padfun = @(k) [UniqueVal{k} zeros(1, newN(k) - idx.size(k))];
CellPad = arrayfun(padfun, 1:numel(UniqueVal) , 'un', 0) ; % apply padding to all elements of C
CellPad = reshape(CellPad, size(UniqueVal)); % reshape into cells
Class_pred = string(cell2mat(CellPad));

%% Convert all "others" label to 0 for the 2nd and 3rd column.
% Rationale: the primary classification is column 1. So "others" in
% column next columns are redundant. They will be further used to
% classify "undecided" category.

temp = Class_pred(:,1);
Class_pred(Class_pred(:,:) == "999") = "0";
Class_pred(:,1) = temp;


% Relabel the ids
Class_pred((Class_pred == "999")) = 'Unclassified';
Class_pred((Class_pred == "1")) = 'Neuron';
Class_pred((Class_pred == "99")) = 'ExciNeuron';
Class_pred((Class_pred == "100")) = 'InhibNeuron';
Class_pred((Class_pred == "2")) = 'Astroglia';
Class_pred((Class_pred == "3")) = 'Microglia';
Class_pred((Class_pred == "4")) = 'Oligodendroglia';
Class_pred((Class_pred == "5")) = 'Endothelial';

% Compile all data
Class_pred = splitvars(table(categorical(Class_pred(:,1:end))));

if size(Class_pred,2) == 1
    Class_pred.Properties.VariableNames = "PrimaryClass";
else
    Class_pred = Class_pred(:,1:2);
    Class_pred.Properties.VariableNames = ["PrimaryClass" "SubClass1"];
end
Class_results_category = cat(2,predictResultsOrig,Class_pred);

clear Class_category

% Check "undecided" nuclei category. Merge all data and prepare for export

if contains( "SubClass1", string(Class_pred.Properties.VariableNames)) == 1
    for ff = 1:size(Class_results_category,1)
        if strcmp(string(Class_results_category.SubClass1(ff)),'0') == 0
            Class_category(ff,1) = "Undecided";
        else
            Class_category(ff,1) = string(Class_results_category.PrimaryClass(ff));
        end
    end
    Class_category = table(Class_category);
    Class_category.Properties.VariableNames = VarNam;
    Class_results_merged = cat(2,Class_results_category(:,1:end-2),Class_category);
else
    Class_results_category.PrimaryClass = string(Class_results_category.PrimaryClass);
    Class_results_category.Properties.VariableNames(1,end) = VarNam;
    Class_results_merged = Class_results_category;
end




