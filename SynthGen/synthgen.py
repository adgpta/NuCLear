import os
import sys

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata
import easygui
import pickle

pd.options.mode.chained_assignment = None


def synthesize(inputDir, synthSize, exportPath):
    allData = {}
    evaluate = {}
    fileList = os.listdir(inputDir)
    for raw_file in fileList:
        allData[raw_file.split(".")[0]] = pd.read_csv(os.path.join(inputDir, raw_file))

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=allData[raw_file.split(".")[0]])

        synthmodel = SingleTablePreset(metadata, name='FAST_ML')
        synthmodel.fit(allData[raw_file.split(".")[0]])

        SyntheticData = synthmodel.sample(round(len(allData[raw_file.split(".")[0]]) * synthSize))
        if os.path.isfile(exportPath + raw_file.split(".")[0] + '.csv'):
            print("File exists. Overwriting in progress..")

        SyntheticData.to_csv(exportPath + raw_file.split(".")[0] + '.csv', index=False)

        print("Generating " + str(synthSize) + " fold synthetic data...\nCellType: " + raw_file.split(".")[0])
        evaluate[raw_file.split(".")[0]] = evaluate_quality(real_data=allData[raw_file.split(".")[0]],
                                                            synthetic_data=SyntheticData, metadata=metadata)

    return evaluate


def main(synthSize):
    inputDir = easygui.diropenbox(title="Select directory for raw data")
    exportPath = easygui.diropenbox(title="Select export path") + '\\'

    # Number of samples to generate = synthSize * length(number of samples in ground truth dataset)

    if len(synthSize) == 0:
        synthSize = 9
        print("No synthetic data size found. Selecting default synthetic data size.")
    else:
        synthSize = int(str(synthSize[0]))

    # Synthesize and evaluate quality
    evaluate = synthesize(inputDir, synthSize, exportPath)

    # Save evaluations
    file_name = os.path.join(exportPath, 'Evaluations.pkl')
    with open(file_name, 'wb') as file:
        pickle.dump(evaluate, file)
        print(f'Object successfully saved to "{file_name}"')


if __name__ == "__main__":

    synthSize = sys.argv[1:]
    main(synthSize)

