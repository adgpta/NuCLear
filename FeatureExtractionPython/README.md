# Feature Extraction with a Python Script

If NucleusAI is unable to perform the feature extraction task, you can utilize the provided script named "extract_features-conn-comp_v4-server.py."

To get started, follow these steps:

1. Create an environment using the provided .yml file:
   ```
   conda env create -f environment.yml
   ```

2. Activate the Conda environment.

3. Execute the following code:
   ```
   python extract_features-conn-comp_v4-server.py -i [RawFile] -m [MaskFile] -o [ResultsFile.csv] --threads [NrOfThreadsForParallelization]
   ```

Replace the placeholders with the appropriate values:

- `[RawFile]`: The path to the raw file from which features will be extracted.
- `[MaskFile]`: The path to the mask file that defines the region of interest for feature extraction.
- `[ResultsFile.csv]`: The desired name or path of the resulting CSV file that will store the extracted features.
- `[NrOfThreadsForParallelization]`: The number of threads to use for parallelization during the feature extraction process.

By following these steps, you can extract features using the provided Python script.
