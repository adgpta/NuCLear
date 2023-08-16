## Stardist for Nucleus Segmentation

In case NucleusAI does not work for nucleus segmentation, you can use the Stardist tool instead.

## Installation

Follow the instructions to install the Stardist Conda Environment from the Stardist GitHub repository. You can find these instructions [here](https://github.com/stardist/stardist).

## Activation of the Conda Environment

Activate the Conda environment by running:

```bash
conda activate stardist
```

## Usage

Once you've installed and activated the Stardist Conda environment, you can run the following command to perform nucleus segmentation:

```bash
python predict_stardist_3d.py -i {InputFolder} -n {ModelFolder} -m {NameOfModel} -o {OutputFolder} -r 80 --ext tif
```

Where:
* `{InputFolder}` is the directory containing the input images for segmentation.
* `{ModelFolder}` is the directory containing the Stardist model.
* `{NameOfModel}` is the name of the model you're using for prediction.
* `{OutputFolder}` is the directory where you want to store the segmented output images.

Please replace `{InputFolder}`, `{ModelFolder}`, `{NameOfModel}`, and `{OutputFolder}` with the appropriate paths on your system.

The `-r 80 --ext tif` part of the command sets memory usage at 80 %.
Sample datasets and a pretrained model for nucleus segmentation can be found [here](https://github.com/adgpta/NucleusAI/tree/master/SampleData)
