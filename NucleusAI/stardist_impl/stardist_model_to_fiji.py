''''MIT License

Copyright (c) 2020 Constantin Pape

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import os
from stardist.models import StarDist2D


def stardist_model_to_fiji(model_path, model=None):

    if model is None:
        save_root, save_name = os.path.split(model_path)
        model = StarDist2D(None, name=save_name, basedir=save_root)

    fiji_save_path = os.path.join(model_path, 'TF_SavedModel.zip')
    print("Saving model for fiji", fiji_save_path)
    model.export_TF()


def main():
    parser = argparse.ArgumentParser(description="Save a stardist model for fiji")
    parser.add_argument('model_path', type=str, help="Where the model is saved.")

    args = parser.parse_args()
    stardist_model_to_fiji(args.model_path)


if __name__ == '__main__':
    main()
