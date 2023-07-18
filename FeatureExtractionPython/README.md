Feature Extraction with a python script

In case NucleusAI does not work for feature extraction, the provided script "extract_features-conn-comp_v4-server.py" can be used. 

Create environment from .yml file: conda env create -f environment.yml
Activate conda environment
Run the following code: python extract_features-conn-comp_v4-server.py -i [RawFile] -m [MaskFile] -o [ResultsFile.csv] --threads [NrOfThreadsForParallelization]
