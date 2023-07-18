

![7](https://github.com/adgpta/NuCLear/assets/77382748/57eef5e2-cdde-4853-bc9a-32a09bb09d10)

<img align="right" height = 240 src="https://github.com/adgpta/NuCLear/assets/77382748/7e29c01e-1caa-43bb-8e29-ac332515a7c8">

### Nucleus-instructed tissue composition using deep learning - _A neural network based cell type classification using nuclei features from fluorescent imaging in 3D_.
#### Version: 4.1
#### Author: Amrita Das Gupta
#### Publications: [Comprehensive monitoring of tissue composition using in vivo imaging of cell nuclei and deep learning](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1); Amrita Das Gupta, Livia Asan, Jennifer John, Carlo A. Beretta, Thomas Kuner, Johannes Knabbe

## Outline

NuCLear is a MLP neural network based cell type classification which can be used to train and classify cell types and their subtypes based on nuclei features obtained from fluorescent imaging as described [here](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1).

![1111](https://github.com/adgpta/NuCLear/assets/77382748/94f90f01-bd83-421b-849c-1970c171756c)


Our application of the NuCLear algorithm is performed on in-vivo two-photon 3D images of histone 2B-eGFP-labeled cell nuclei. Using StarDist (https://github.com/stardist/stardist) to segment the nuclei and PyRadiomics (https://github.com/AIM-Harvard/pyradiomics) to extract various features, NuCLear trains sevaral classifiers depending on the data provided by the user to classify all cells per imaging volume. Beyond in-vivo imaging, With good training data, NuCLear would be able to work with any fluorescence-based microscopy images and perform in any organ or model system. 


![3](https://github.com/adgpta/NuCLear/assets/77382748/8a7ec983-fb8e-40a8-897f-1aeec0b46bc3)


## Methodology
Using segmentation editor in Fiji, ground truth were generated by manually labelling and segmenting nuclei in 3D as below.


<div align="center">
  <video src="https://github.com/adgpta/NuCLear/assets/77382748/3a002c61-110e-45cc-9141-8a5234d33911" width="200" height = "100" />
</div>

<div align="center">
  <video src="https://github.com/adgpta/NuCLear/assets/77382748/85b448fa-5190-4bb5-9e55-9c090099dbed" width="200"  height = "100" />
</div>


These ground truth sets (labelled and raw images) were used for training a segmentation model using [StarDist](https://github.com/stardist/stardist), in 3D via an in-house developed Jupyter Notebook.
![2](https://github.com/adgpta/NuCLear/assets/77382748/c1e4a16a-5574-4b60-adfb-3daabe800033)

  
Using the [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) python package, in total 107 radiomics features were extracted for each segmented nucleus after segmentation using StarDist, including, but not limited, to morphological features, intensity features and texture features. 

The features are further used to train classifier models for each cell type provided in the ground truth data. This includes major cell types and sub types. This version can classify between the major cell types of Neurons, Glia and endothelial cells, along with the neuronal subtypes: excitatory and inhibitory neurons and glial subtypes: astroglia, microglia and oligodendroglia.
A GUI based version for segmentation and extraction of radiomics feature can be found [here](https://github.com/SFB1158RDM/NucleusAI).
  

## Guide:

This repository contains the MATLAB and python implementations of NuCLear. The following describes step-by-step guide for ground truth extraction, training and classification for each case.


### _Generate ground truth data for training_
To generate classifier models for each cell type, supervised training was performed using ground truth datasets from two-photon volumetric fluorescence imaging. GFP and RFP stacks were acquired (with the RFP indicating different cell types). 
![4](https://github.com/adgpta/NuCLear/assets/77382748/337adb0c-5600-4fc1-b81b-723637f049f0)

The GFP images were segmented with each nuclei being assigned an unique value and their radiomics features were extracted in a .csv file. Positive cells for each type were identified by overlaying the RFP and GFP images as described in the methods [here](https://www.biorxiv.org/content/10.1101/2022.10.03.510670v1). For each positive cell, the corresponding label was identified from the segemented images and all the radiomics features of the said label was extracted and saved.


![5](https://github.com/adgpta/NuCLear/assets/77382748/374a30a0-15b3-4b57-a65b-3febc61fc130)

### _Generate synthetic / augmented data_ 
To increase the number of nuclei for training and include possible variations, synthetic datasets were created from nuclei features using the [synthetic data vault (SDV)](https://github.com/sdv-dev/SDV) package (Patki, Wedge et al. 2016), which utilizes correlations between features of the original dataset as well as mean, minimum or maximum values and standard deviations. CSV files containing pre-generated synthetic data from the provided ground truths are available in the (folder). To create your own, _**select only the radiomics features columns to be used for creating the synthetic datasets**_ and remove all non-essential columns from the csv files extracted via Pyradiomics. To create synthetic data follow the instructions below to run the python script:


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
   
6. To read evaluation data 'Evaluations.pkl' file via any IDE console use the code snippet below. Replace 'CELLTYPE' with the name of the desired cell type: eg, 'Neurons'.
    ```
    file_name = os.path.join(exportPath, 'Evaluations.pkl')
    
    with open(file_name, 'rb') as file:
        Evaluations = pickle.load(file)

    Evaluations['CELLTYPE'].get_score()
    Evaluations['CELLTYPE'].get_properties()
    Evaluations['CELLTYPE'].get_details(property_name='Column Shapes')

    ```
    

### _Create training datasets_
For training datasets, all the ground truth and augmented data should have an assigned Id in the first column of the csvs (both ground truth and synthetic datasets). The default class names and ids are as follows:
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
  For predefined training data, "NuCLearTrainingWorkspace.mat" is provided in the 'Workspace' folder. To create your own training datasets, load all csvs (groundtruth + synthesized data) as arrays and save the headers to 'header' variable (refer to NuCLearTrainingWorkspace.mat for structure). Save the workspace.
## CHECK
- For Python:
  For predefined training data, "NuCLearTrainingWorkspace.pkl" is provided in the 'Workspace' folder. To create your own training datasets, load all csvs as a dictionary (groundtruth + synthesized data) as tables and save as pickle file. 

**TO NOTE:**
_While creating data for subtypes for cells, the name should contain partial match to the corresponding major cell type name. As an example, for excitatory neuronal data, the variable name should contain 'Neuron' which would be the major cell type (eg: ExciNeuron)._

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

### _Training NuCLear classifiers_
#### **Training Module:**
Designed to train any number of major classes and sub classes of cells. The training is performed on all major classes for classes defined as "Maj" in the variable "ClassDef". For classes defined as "Sub" in "ClassDef", the training is done only on data for all subclasses belonging to the same major class. For eg. to train the classifiers for excitatory neurons, the training will be performed on all neuronal subclass data, i.e. excitatory and inhibitory. The "ClassDef" variable MUST contain ALL the training data available, with correct denotion of "Sub" or "Maj" with the subclass names containing parts of the major class. For eg. excitatory and inhibitory neurons are labelled as "ExciNeuron" and "InhibNeuron" containing "Neuron", which defines they belong to the "Neuron" class. 'ClassDef' is defined for default cell types. 

NuCLearTrainingWorkspace.mat contains 1 variable for each cell type with feature extraction data from pyradiomics. The classification model may be trained with real dataset or combined with augmented / Synthetic datasets (refer to python script SynthGen.py) created from the real dataset. Multiple classifiers are trained using datasets provided as tables in NuCLearTrainingWorkspace. The dataset is divided into training, validation and test sets with a ratio of 70:15:15 respectively. 
  
#### MATLAB

<img align="right" height = "450" src="https://github.com/adgpta/NuCLear/assets/77382748/2f193042-2c4b-4805-bc45-92fa704cb20a">

1. Add folder containing all scripts and previously saved workspaces to MATLAB path. _(Optional)_ add folders containing provided workspaces and models if using predefined ground truths and pretrained models.
2. Run NuCLearTrainingModule.m.
3. Load training workspace containing all training datasets created earlier.
4. Select the datasets that are to be used for training. Training will vary depending on the datasets used. All major categories will be trained together and the sub categories will be trained with only the other sub categories of the same major category. eg. if training excitatory and inhibitory neurons, they will only be trained against each other.
5. Enter the definitions for new classes if any. The defaults are already added.
6. Select which models to train.
7. Save model when prompted.

The file saved is a structure containining the trained models. 'ClassDef' contains information about datasets used for training and 'TrainedModels' shows the names of the models that are trained. 'trainedNet' contains the trained classifiers, accuracy, test set and prediction sets. For more details, refer to the publication.

### _Cell type classification_
#### **Classification Module:**
Add input file directory containing the feature extraction csvs from pyradiomics feature extraction script. Batch processes all csvs from pyradiomics feature extraction in inputFileDir. SynthVer specifies the model to be used for classification which depends on the training dataset (Original dataset (Orig) or Synthesized dataset (Synth9))

#### MATLAB

1. Add folder containing all scripts and previously saved models to MATLAB path.
2. Run NuCLearClassificationModule.m.
3. Select model to use for classification.
4. Specify path to input directory containing csvs extracted via PyRadiomics.
5. Select training model. _(Optional: Only for multiple models save in a single file)_.

The results are exported in the input directory in a folder named as "NuCLear[modelname][datestamp]". Within the folder are the raw predictions for each classifier for each nuclei, cleaned predictions and final results.

