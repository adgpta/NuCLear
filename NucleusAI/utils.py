import os



def run_local(config, destination='training'):
    print('Running Locally')
    from stardist_impl import train2d, train3d, predict2d, predict3d, validate2d, validate3d
    from pyrad import extract_features_copy
    from pyrad import extract_features_2D, extract_features_3D

    if destination == 'training':
        inputDir = config['inputDir']
        modelDir = config['outputDir']

        modelName = config['modelName']
        patchSizeH = int(config['patchSizeH'])
        patchSizeW = int(config['patchSizeW'])
        try:
            patchSizeD = int(config['patchSizeD'])
        except:
            patchSizeD = config['patchSizeD']

        valFraction = config['valFraction']
        extension = config['extension']

        twoDim = config['twoDim']
        saveForFiji = config['saveForFiji']
        multichannel = config['multichannel']

        n_rays = int(config['n_rays'])
        no_epochs = int(config['epochs'])
        steps_per_epoch = int(config['steps_per_epoch'])

        print('twoDim ::: ', twoDim)

        print('inputDir ::: ', inputDir)

        model_path = os.path.join(modelDir, modelName)

        #see model_path with print
        print('model_path ::: ', model_path)

        if twoDim:
            print('2-DIMENSION')
            pass
            train2d(inputDir,
                     model_path,
                     "images",
                     "masks",
                     extension,
                     valFraction,
                     tuple([int(patchSizeW), int(patchSizeH)]),
                     multichannel,
                     saveForFiji,
                     n_rays,
                     no_epochs,
                     steps_per_epoch)
        else:
            print('3-DIMENSION')
            train3d(inputDir,
                     model_path,
                     "images",
                     "masks",
                     extension,
                     valFraction,
                     tuple([int(patchSizeD), int(patchSizeW), int(patchSizeH)]),
                     anisotropy=None,
                     n_rays=n_rays,
                     no_epochs=no_epochs,
                     steps_per_epoch=steps_per_epoch)

    elif destination == 'feature_extraction':
        print('Starting feature extraction.')
        inputfile = config['inputDir']
        outputfile = config['outputDir']
        nrofthreads = config['nr_of_threads']
        twoDim = config['twoDim']
        #extract_features_copy.main(inputfile, outputfile, nrofthreads)
        if twoDim:
            print('2-DIMENSION')
            extract_features_2D.main(inputfile, outputfile)
        else:
            print('3-DIMENSION')
            extract_features_3D.main(inputfile, outputfile, nrofthreads)
            
        print('Finished Feature Extraction.')
        

    elif destination == 'validation':
        inputDir = config['inputDir']
        #modelDir = config['outputDir']
        model_path = config['modelDir']
        twoDim = config['twoDim']
        extension = config['extension']
        multichannel = config['multichannel']

        op_image_path = os.path.join(config['outputDir'], 'output_images')
        os.makedirs(op_image_path, exist_ok=True)
        outputDir = op_image_path

        memory_usage = 100

        print('model_path ::: ',model_path, 'destination ::', destination)
        #memory_usage = int(config['prediction']['memoryUsage'])
        #see
        
        if twoDim:
            print("Validate-2D")
            scores_dict = validate2d(model_path, inputDir, outputDir, extension, multichannel, memory_usage)

            return scores_dict
        else:
            print("Validate-3D")
            scores_dict = validate3d(model_path, inputDir, outputDir, extension, memory_usage)

            return scores_dict
            
    else:
        inputDir = config['inputDir']
        modelDir = config['outputDir']
        twoDim = config['twoDim']
        extension = config['extension']
        multichannel = config['multichannel']

        op_image_path = os.path.join(config['outputDir'], 'output_images')
        os.makedirs(op_image_path, exist_ok=True)
        outputDir = op_image_path

        if destination == 'prediction':
            modelName = config['modelName']
            model_path = os.path.join(modelDir, modelName)
        else:
            model_path = config['modelDir']
        memory_usage = 100

        print('model_path ::: ',model_path, 'destination ::', destination)
        
        if twoDim:
            print("PREDICT-2D")
            predict2d(model_path, inputDir, outputDir, extension, multichannel, memory_usage)
        else:
            print("PREDICT-3D")
            predict3d(model_path, inputDir, outputDir, extension, memory_usage)
      
