
# Generate synthetic / augmented data using [SDV](https://github.com/sdv-dev/SDV) package

This repository is a part of generating synthetic data for [NuCLear](https://github.com/adgpta/NuCLear). Please refer to the original repository for more information.

To increase the number of nuclei for training and include possible variations, synthetic datasets were created from nuclei features using the [synthetic data vault (SDV)](https://github.com/sdv-dev/SDV) package (Patki, Wedge et al. 2016), which utilizes correlations between features of the original dataset as well as mean, minimum or maximum values and standard deviations. CSV files containing pre-generated synthetic data from the provided ground truths are available in the (folder). To create your own, _**select only the radiomics features columns to be used for creating the synthetic datasets**_ and remove all non-essential columns from the csv files extracted via Pyradiomics. To create synthetic data follow the instructions below to run the python script:


The following scripts has been created with Python 3.10.

1. Clone the repository 'SynthGen'.
   ```
   git clone https://github.com/adgpta/SynthGen.git
   ```
2. **Conda**: Create a virtual environment and install "requirements.txt" file using
   ```
   conda create --name synthgen python=3.10
   pip install -r requirements.txt
   ```
   OR in **IDE**: Open cloned repository as a new project. In terminal enter:
   ```
   pip install -r requirements.txt
   ```
   
3. Run synthgen.py with the times of synthetic data to be generated. If no input it provided this script will create x9 fold augmented data by default.
   
   ```
   python synthgen.py 9
   ```

4. When the pop up boxes appear, navigate to the folder for the ground truth data (.csv files) to be used to create synthetic data and subsequently, set the export folder. The evaluation data is saved in the export folder as a dictionary file. 
   
5. To read evaluation data 'Evaluations.pkl' file via any IDE console use the code snippet below. Replace 'CELLTYPE' with the name of the desired cell type: eg, 'Neurons'.
    ```
    file_name = os.path.join(exportPath, 'Evaluations.pkl')
    
    with open(file_name, 'rb') as file:
        Evaluations = pickle.load(file)

    Evaluations['CELLTYPE'].get_score()
    Evaluations['CELLTYPE'].get_properties()
    Evaluations['CELLTYPE'].get_details(property_name='Column Shapes')

    ```
    
