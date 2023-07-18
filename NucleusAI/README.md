# NucleusAI
The goal of the proposed project is to implement a deep-learning application programming interface (API) for image data processing and to provide a platform for the scientific community to directly compare and integrate data generated across our consortium projects. The development of an open-source and extensible platform to train and share deep-learning models will guarantee high standards in many image analysis workflows and additionally reduce the amount of annotated data necessary for training supervised deep-learning algorithms. The initial efforts would involve further development of StarDist ( https://github.com/stardist/stardist ). Based on the existing StarDist GUI framework, we will further integrate other commonly used deep-learning image analysis tools to make the initial GUI more flexible for model training and inferences. Some of the considered tools include CARE (image enhancement) (Weigert et al. 2018), Noise2Void (image denoising) (Krull et al. 2018), CellPose (cell segmentation) (Stringer et al. 2021) and Elektronn3 (EM data segmentation, https://github.com/ELEKTRONN/elektronn3).

### Documentation

NucleusAI is a graphical user interface (GUI) developed using the pyQT library for training, image validation, and feature extraction tasks related to AI-based nucleus detection in microscopy images. This documentation will guide users through the functionality of the GUI and how to use it effectively.

###### Getting Started: 

Before running the Nucleus GUI, ensure that the following prerequisites are installed on your system:

- Python 3.7
- Git
- Anaconda or Miniconda

###### Installation: 

To install NucleusAI GUI, follow these steps:

Clone the GitHub repository using the command:
$ git clone https://github.com/SFB1158RDM/Imaging_tools.git

Navigate to the GUI folder using the command:
$ cd /path-to-repository/GUI/

Create a new environment named sd-gpu and install Python 3.7 using the command:
conda create --name sd-gpu python=3.7

Activate the sd-gpu environment using the command:
conda activate sd-gpu

Install the required dependencies using the command:
pip install -r requirements.txt

Download the sample dataset using the link:
https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip

Extract the downloaded dataset to the root directory of the cloned repository.

Running Nucleus GUI: To start the Nucleus GUI, navigate to the GUI folder and run the command: python stardistGUI.py 
This will launch the GUI application.

###### Functionality:
The Nucleus GUI offers the following features:

###### Training: 
Users can train a neural network on their own dataset for nucleus detection using the provided settings and parameters.

###### Validation: 
Users can validate the trained neural network on their own dataset or the sample dataset provided.
###### Feature Extraction:
Users can extract features of the nucleus from an input image using the trained neural network.
###### User Interface: 
The Nucleus GUI has a user-friendly interface that is easy to navigate. The main interface consists of three tabs: Train, Validate, and Feature Extraction. Each tab has its own set of parameters and settings that users can adjust as per their requirements.
###### Train Tab: 
The Train tab allows users to train a neural network on their own dataset. Users can specify the dataset directory, the number of epochs, learning rate, batch size, and other settings. Once the settings are configured, users can click on the Train button to start the training process. Users can monitor the training progress in the console window.
###### Validate Tab: 
The Validate tab allows users to validate the trained neural network on their own dataset or the sample dataset provided. Users can specify the validation dataset directory and other settings. Once the settings are configured, users can click on the Validate button to start the validation process. Users can monitor the validation progress and view the validation results in the console window.
###### Feature Extraction Tab: 
The Feature Extraction tab allows users to extract features of the nucleus from an input image using the trained neural network. Users can specify the input image, the trained neural network, and other settings. Once the settings are configured, users can click on the Extract Features button to start the feature extraction process. Users can view the extracted features in the console window.

###### Related Readings:

- StarDist (https://github.com/stardist/stardist)
- Weigert, Martin, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers. 2020. “Star-Convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.”
In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 3655–62.
- Weigert, Martin, Uwe Schmidt, Tobias Boothe, Andreas Müller, Alexander Dibrov, Akanksha Jain, Benjamin Wilhelm, et al. 2018. “Content-Aware Image Restoration:
Pushing the Limits of Fluorescence Microscopy.” Nature Methods 15 (12): 1090–97.
- Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. 2018. “Noise2Void - Learning Denoising from Single Noisy Images.” arXiv [cs.CV]. arXiv.
http://arxiv.org/abs/1811.10980.
- Krull, Alexander, Tomáš Vičar, Mangal Prakash, Manan Lalit, and Florian Jug. 2020. “Probabilistic Noise2Void: Unsupervised Content-Aware Denoising.” Frontiers in Computer Science 2 (February). https://doi.org/10.3389/fcomp.2020.00005.
- Stringer, Carsen, Tim Wang, Michalis Michaelos, and Marius Pachitariu. 2021. “Cellpose: A Generalist Algorithm for Cellular Segmentation.” Nature Methods 18 (1):
100–106.
- elektronn3: A PyTorch-Based Library for Working with 3D and 2D Convolutional Neural Networks, with Focus on Semantic Segmentation of Volumetric Biomedical Image
Data. n.d. Github. Accessed.
