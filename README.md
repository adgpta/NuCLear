
![7](https://github.com/adgpta/NuCLear/assets/77382748/98c744d8-f4a7-41a9-b624-a16651c3e6d1)


<img align="right" height = 240 src="https://github.com/adgpta/NuCLear/assets/77382748/b9d478f2-9fd3-4b71-8d13-59aea0c0fb59">


### Nucleus-instructed tissue composition using deep learning - _A neural network based cell type classification using nuclei features from fluorescent imaging in 3D_.
#### Version: 4.1
#### Author: Amrita Das Gupta
#### Publications: [Comprehensive monitoring of tissue composition using in vivo imaging of cell nuclei and deep learning](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1); Amrita Das Gupta, Jennifer John, Livia Asan, Carlo A. Beretta, Thomas Kuner, Johannes Knabbe

<br />

## OUTLINE

NuCLear is a neural network based cell type classification which can be used to train and classify cell types and their subtypes based on nuclei features obtained from fluorescent imaging as described [here](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1).

![image](https://github.com/adgpta/NuCLear/assets/77382748/95bec3e8-5c75-4008-a437-a0e599384d8d)


Our application of the NuCLear algorithm is performed on in-vivo two-photon 3D images of histone 2B-eGFP-labeled cell nuclei in the mouse brain. Using StarDist (https://github.com/stardist/stardist) to segment the nuclei and PyRadiomics (https://github.com/AIM-Harvard/pyradiomics) to extract various features, NuCLear trains several classifiers depending on the data provided by the user to classify all cells per imaging volume. Beyond in-vivo imaging, provided there is good training data, NuCLear would be able to work with any fluorescence-based microscopy images and perform in any organ or model system. 


![3](https://github.com/adgpta/NuCLear/assets/77382748/c43520bd-4fee-40f9-97c4-7e84ac45c072)


<br />

<br />

## NUCLEUS SEGMENTATION
Using the segmentation editor in Fiji, ground truth data for nucleus segmentation in 2-photon in vivo images was generated by manually labelling and segmenting nuclei in 3D as described below.


<div align="center">
  <video src="https://github.com/adgpta/NuCLear/assets/77382748/31870c3f-8b7e-4601-9b2b-be5ca4f3b658" width="50"/>
</div>


<div align="center">
  <video src="https://github.com/adgpta/NuCLear/assets/77382748/d8c1c2d4-013d-46f8-86b5-5293a379ee75" height = "100" />
</div>

These ground truth sets (labelled and raw images) were used for training a segmentation model using [StarDist](https://github.com/stardist/stardist), in 3D. The trained model can be found [here](https://github.com/adgpta/NuCLear/tree/main/StardistModel/SegModel).

![2](https://github.com/adgpta/NuCLear/assets/77382748/156658da-b136-4ba4-bdef-52b9e5d962fb)


<br />

<br />

## FEATURE EXTRACTION
  
Using the [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) python package, in total 107 radiomics features were extracted for each segmented nucleus after segmentation using StarDist, including, but not limited, to morphological features, intensity features and texture features. 

The features were further used to train classifier models for each cell type provided in the ground truth data. This included major cell types and sub types. This version can classify between the major cell types of neurons, glia and endothelial cells, along with the neuronal subtypes: excitatory and inhibitory neurons and glial subtypes: astroglia, microglia and oligodendroglia.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

> _**A GUI based version for segmentation (using StarDist) and extraction of radiomics features (using Pyradiomics) can be found [here](https://github.com/adgpta/NucleusAI). Follow the step by step guide in the link for instructions on how to install and use it. Sample data sets are provided in the [SampleData](https://github.com/adgpta/NucleusAI/tree/master/SampleData) folder.**_

> _**If you plan to use your own StarDist installation, our pre-trained nucleus segmentation model can be found [here](https://github.com/adgpta/NuCLear/tree/main/StardistModel/SegModel).**_

> _**To use the feature extraction pipeline without using the GUI, please download the python scripts and follow the instructions in [here](https://github.com/adgpta/NuCLear/tree/main/FeatureExtractionPython). Files required to use the scripts are raw images and binary segmented masks.**_ 

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br />

## GUIDE

This repository contains the MATLAB and python implementations of NuCLear. The following step-by-step guide describes ground truth extraction, training and classification.

<br />

### _Generate ground truth data for training_
To perform supervised training of the deep neural network for cell type classification, a ground truth dataset was created using the two-photon volumetric fluorescence data, automatically segmented nuclei, which were assigned a unique label, and the radiomics features. Using the red fluorescence channel, nuclei belonging to a specific cell type were manually identified after creating a composite image of the green and red fluorescence images. 

![4](https://github.com/adgpta/NuCLear/assets/77382748/20ca6b89-d0cd-4741-83b2-05f474252af9)

This allowed for identification of the label by synchronizing the composite image and the StarDist output in ImageJ/Fiji and extraction of radiomics features (saved in a .csv file) for these individual nuclei.  

![5](https://github.com/adgpta/NuCLear/assets/77382748/253dfe43-7bf5-462d-af99-2e91b96121e6)

<br />

### _Generate synthetic / augmented data_

To increase the number of nuclei for training, synthetic datasets were created from nuclei features using the [synthetic data vault (SDV)](https://github.com/sdv-dev/SDV) package (Patki, Wedge et al. 2016), which utilizes correlations between features of the original dataset as well as mean, minimum or maximum values and standard deviations. CSV files containing pre-generated synthetic data from the provided ground truths are available [here](https://github.com/adgpta/SynthGen/tree/master/output). To create your own, _**select only the radiomics features columns to be used for creating the synthetic datasets**_ and remove all non-essential columns (viz: radiomics version, python version,  etc.) from the csv files extracted via Pyradiomics. To create synthetic data follow the instructions below to run the python script:


The following scripts has been created with Python 3.10.

1. Install [git](https://git-scm.com/downloads). 
2. Clone the repository 'SynthGen'.
   ```
   git clone https://github.com/adgpta/SynthGen.git
   ```
3. **Conda**: Create a virtual environment and install "requirements.txt" file using
   ```
   conda create --name synthgen python=3.10
   pip install -r requirements.txt
   ```
   OR in **IDE**: Open cloned repository as a new project. In terminal enter:
   ```
   pip install -r requirements.txt
   ```
   
4. Run synthgen.py with the times of synthetic data to be generated. If no input it provided this script will create x9 fold augmented data by default.
   
   ```
   python synthgen.py 9
   ```

5. When the pop up boxes appear, navigate to the folder for the ground truth data (.csv files) to be used to create synthetic data and subsequently, set the export folder. The evaluation data is saved in the export folder as a dictionary file. 
   
6. To read evaluation data `Evaluations.pkl` file via any IDE console use the code snippet below. Replace 'CELLTYPE' with the name of the desired cell type: eg, 'Neurons'.
    ```
    file_name = os.path.join(exportPath, 'Evaluations.pkl')
    
    with open(file_name, 'rb') as file:
        Evaluations = pickle.load(file)

    Evaluations['CELLTYPE'].get_score()
    Evaluations['CELLTYPE'].get_properties()
    Evaluations['CELLTYPE'].get_details(property_name='Column Shapes')

    ```
    
<br />

### _Create training datasets_

For training datasets, all the ground truth and augmented data should have an assigned Id in the first column of the csvs (both ground truth and synthetic datasets). The default class names and ids are as follows:

<img align="right" height = "360" src="https://github.com/adgpta/NuCLear/assets/77382748/9e57f292-7dc7-4feb-a262-40eaab5f5137">



  ```
  Neuron = "0"
  Astroglia = "1"
  Microglia = "2"
  Oligodendroglia = "3"
  Endothelial = "4"
  Excitatory Neuron = "99"
  Inhibitory Neuron = "100"
  ```

- For MATLAB:
 
  For predefined training data, `NuCLearTrainingWorkspace.mat` is provided in the "[Workspace](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Workspace)" folder. To create your own training datasets, load all csvs (groundtruth + synthesized data) as arrays and save the headers to 'header' variable (refer to NuCLearTrainingWorkspace.mat for structure). Save the workspace.

- For Python:
 (_Will be added soon_)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

> **TO NOTE:**
_**While creating data for subtypes for cells, the name should contain partial match to the corresponding major cell type name. As an example, for excitatory neuronal data, the variable name should contain 'Neuron' which would be the major cell type (eg: ExciNeuron). The sample workspace `NuCLearTrainingWorkspace.mat` is shown above.**_

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br />

### _Feature selection_
To reduce overfitting, a sequential feature selection algorithm was used to select the most relevant features with minimum loss function (snippet below). 'tableOrig' is the table with all ground truth datasets without any synthetic data. For each training / classification ensemble, sequential feature selection was run to find the best columns to use, for eg., once on all the major classes and a second time for only excitatory and inhibitory neurons.

```
tableOrig = table2array(tableOrig);
tableOrig2 = tableOrig(tableOrig2(:,1) ~= 0,:);
nuclei = (tableOrig2(:,1));
c2 = cvpartition(nuclei,'k',10); 
opts = statset('Display','iter','UseParallel',true);
fun = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT),Xt,yt);
[fs_all,history_all] = sequentialfs(fun,(tableOrig2(:,2:end)),nuclei,'cv',c2,'options',opts)

```
-----------------------------

> **TO NOTE:**
_**Default features are pre-selected for the classes. The selected features are provided in the published article.**_

---------------------------------

<br />

### _Training NuCLear classifiers_

Designed to train any number of major classes and sub classes of cells. The training is performed on all major classes for classes defined as "Maj" in the variable "ClassDef". For classes defined as "Sub" in "ClassDef", the training is done only on data for all subclasses belonging to the same major class. For eg. to train the classifiers for excitatory neurons, the training will be performed on all neuronal subclass data, i.e. excitatory and inhibitory. The "ClassDef" variable MUST contain ALL the training data available, with correct denotion of "Sub" or "Maj" with the subclass names containing parts of the major class. For eg. excitatory and inhibitory neurons are labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which defines they belong to the "Neuron" class. 'ClassDef' is defined for default cell types. 

[NuCLearTrainingWorkspace.mat](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Workspace) contains 1 variable for each cell type with feature extraction data from pyradiomics. The classification model may be trained with real dataset or combined with augmented / Synthetic datasets (generated from the python script SynthGen.py) created from the real dataset. Multiple classifiers are trained using datasets provided as tables in NuCLearTrainingWorkspace. The dataset is divided into training, validation and test sets with a ratio of 70:15:15 respectively. 
  
#### MATLAB training

<img align="right" height = "450" src="https://github.com/adgpta/NuCLear/assets/77382748/28e5575f-a334-4386-b159-05315c80295a">

1. Add folder containing all scripts and previously saved workspaces to MATLAB path. _(Optional)_ add folders containing provided workspaces and models if using predefined ground truths and pretrained models.
2. Run [NuCLearTrainingModule.m](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Scripts).
3. Load training workspace containing all training datasets created earlier.
4. Select the datasets that are to be used for training. Training will vary depending on the datasets used. All major categories will be trained together and the sub categories will be trained with only the other sub categories of the same major category. eg. if training excitatory and inhibitory neurons, they will only be trained against each other.
5. Enter the definitions for new classes if any. The defaults are already added.
6. Select which models to train.
7. Save model when prompted.

The file saved is a structure containining the trained models. 'ClassDef' contains information about datasets used for training and 'TrainedModels' shows the names of the models that are trained. 'trainedNet' contains the trained classifiers, accuracy, test set and prediction sets. For more details, refer to the publication.

<br />

### _Cell type prediction_


This module classifies all the csvs extracted via pyradiomics feature extraction using either the [GUI](https://github.com/adgpta/NucleusAI) or the feature extraction [script](https://github.com/adgpta/NuCLear/tree/main/FeatureExtractionPython). SynthVer specifies the model to be used for classification which depends on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9)). To use the default training model, save the model from [here](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Model).


<img align="right" width = "440" src="https://github.com/adgpta/NuCLear/assets/77382748/76ae6db6-9ef6-4482-9ce2-22666df39d98">


#### MATLAB Classification

1. Add folder containing all scripts and previously saved models to MATLAB path.
2. Run [NuCLearClassificationModule.m](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Scripts).
3. Select the trained model to use for classification. A pre-trained model is provided [here](https://github.com/adgpta/NuCLear/tree/main/NuCLearV4.1/Model).
4. Select the directory containing csvs with radiomics features extracted via [here](https://github.com/adgpta/NucleusAI/blob/master/README.md#feature-extraction-based-on-pyradiomics). Sample extraction files can be found [here](https://github.com/adgpta/NucleusAI/tree/master/SampleData/PyradiomicsFiles).
5. Select training model. _(Optional: Only for multiple models save in a single file)_.
6. The results are exported in the input directory in a folder named as "NuCLear[modelname][datestamp]". Within the folder are the raw predictions for each classifier for each nuclei, cleaned predictions and final results.
7. The `Prediction_results` folder contains the final predictions for each file. 
8. The output `*_Classified.csv` file has information for animal number, timepoint, position of the stacks, binary mask label, centroids for each nuclei defining its position in 3D space, type of cell the nuclei belongs to and subclass (if any).



------------------------------------------------------------------------------

> **TO NOTE:**
**_The filenames of the .csv files should be in the format of `Animal_Timepoint_Position.tif`. The animal, timepoint and position MUST be separated by an underscore.**_

------------------------------------------------------------------------------


<br />

<br />

## Upcoming features
- Python scripts for celltype classification and training modules.


<br />


## Contact
For any issues, queries or to add more features please contact the authors directly or open an issue.


<br />


## How to cite 
```
@article {Gupta2022.10.03.510670,
	author = {Amrita Das Gupta and Livia Asan and Jennifer John and Carlo A. Beretta and Thomas Kuner and Johannes Knabbe},
	title = {Comprehensive monitoring of tissue composition using in vivo imaging of cell nuclei and deep learning},
	elocation-id = {2022.10.03.510670},
	year = {2022},
	doi = {10.1101/2022.10.03.510670},
}
```
-------------------------------------------------------------------------------------------------------------------------------------
