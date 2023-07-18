import cc3d
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from radiomics import featureextractor, imageoperations
import csv
import tifffile as tiff
from tic_toc import TicTocGenerator, tic, toc
import ray
import sys
import getopt

@ray.remote
def split_volume(image, mask, bbox_list, centroid_list, start_i, end_i):
    #Take bbox and extract sub_image and sub_mask from image and mask
    # For label identification take centroid of the label and read the value in mask
    print(start_i)
    print(end_i)
    sub = dict()
    sub['image'] = list()
    sub['mask'] = list()
    sub['label'] = list()
    #Add Centroids to the target dict:
    sub['centroids'] = list()
    for i in range(start_i,end_i):
        print(i)
        sub['image'].append(image[bbox_list[i][0][0]:bbox_list[i][0][1], bbox_list[i][1][0]:bbox_list[i][1][1],
                    bbox_list[i][2][0]:bbox_list[i][2][1]])
        sub['mask'].append(mask[bbox_list[i][0][0]:bbox_list[i][0][1], bbox_list[i][1][0]:bbox_list[i][1][1],
                    bbox_list[i][2][0]:bbox_list[i][2][1]])
        sub['label'].append(int(mask[centroid_list[i][0], centroid_list[i][1], centroid_list[i][2]]))
        sub['centroids'].append([centroid_list[i][0], centroid_list[i][1], centroid_list[i][2]])
    return sub

@ray.remote
def run_feature_extractor(target_dict, cells_to_exclude):
    feature_vector = OrderedDict()
    feature_vector_store = list()
    settings = {}
    settings['binWidth'] = 25
    settings[
        'resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['preCrop'] = True
    settings['shape'] = ['VoxelVolume', 'MeshVolume', 'SurfaceArea',
                         'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity',
                         'SphericalDisproportion', 'Maximum3DDiameter', 'Maximum2DDiameterSlice',
                         'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'MajorAxisLength',
                         'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness']

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    for i in range(0, len(target_dict['label'])):
        print(i)
        image_sitk = sitk.GetImageFromArray(target_dict['image'][i])
        mask_sitk = sitk.GetImageFromArray(target_dict['mask'][i])  # .transpose(2, 1, 0))
        conv_i = target_dict['label'][i]
        #Exclude labels of cells that touch borders:
        try:
            print("Calculating features")
            (ii, im) = imageoperations.resampleImage(image_sitk, mask_sitk, resampledPixelSpacing=[0, 0, 0], label=conv_i,
                                                     padDistance=10)
            featureVector = extractor.execute(ii, im, label=conv_i)
            featureVector.update({'Label': conv_i})
            #Add centroids to featureVector:
            #print(target_dict['centroids'][i])
            featureVector.update({'Centroid_Z': target_dict['centroids'][i][0]})
            featureVector.update({'Centroid_Y': target_dict['centroids'][i][1]})
            featureVector.update({'Centroid_X': target_dict['centroids'][i][2]})
            if conv_i in cells_to_exclude:
                featureVector.update({'BorderTouching': 1})
            else:
                featureVector.update({'BorderTouching': 0})
            feature_vector_store.append(featureVector)
        except:
            print("Error")
    return feature_vector_store



def main(argv):
    inputfile = ''
    outputfile = ''
    maskfile = ''
    nrofthreads = ''
    try:
        opts, args = getopt.getopt(argv, "hi:m:o:t", ["ifile=", "mfile=", "ofile=", "threads="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -m <maskfile> -o <outputfilename> -t <nr of threads>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -m <maskfile> -o <outputfilename> -t <nr of threads>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-m", "--mfile"):
            maskfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-l", "--lfile"):
            listfile = arg
        elif opt in ("-t", "--threads"):
            nrofthreads = arg
    if len(sys.argv) > 0:
        #Nr of parallel threads to run:
        nr_of_threads = int(nrofthreads)
        print("Number of threads: {}".format(nr_of_threads))
        # Output filename
        output_filename = outputfile
        # Read input image:
        image = tiff.imread(inputfile)
        # Read mask image
        mask = tiff.imread(maskfile)
        print("Images loaded")
        #Create list of cells touching the border:
        mask_shape = np.shape(mask)
        # Define 6 border layers, test which labels are touching this borders by np.unique of that layer:
        x_y_layer_top = np.unique(mask[0, :, :])
        x_y_layer_bottom = np.unique(mask[mask_shape[0] - 1, :, :])
        x_z_layer_y_0 = np.unique(mask[:, :, 0])
        x_z_layer_y_maxy = np.unique(mask[:, :, mask_shape[2] - 1])
        y_z_layer_x_0 = np.unique(mask[:, 0, :])
        y_z_layer_x_maxx = np.unique(mask[:, mask_shape[1] - 1, :])
        #Cells to exclude are in border_cells_array
        border_cells_array = np.unique(np.concatenate(
            (x_y_layer_top, x_y_layer_bottom, x_z_layer_y_0, x_z_layer_y_maxy, y_z_layer_x_0, y_z_layer_x_maxx)))

        #Calculate bounding boxes for labels with cc3d library:
        labels_in = np.copy(mask)
        labels_out = cc3d.connected_components(labels_in)  # 26-connected
        stats = cc3d.statistics(labels_out)
        print("Stats calculated")
        #extract the bounding boxes and centroids for label extraction:
        bboxes2 = list()
        centroids = list()
        counter = 0
        for i in stats['bounding_boxes']:
            centroid = stats['centroids'][counter]
            label_int = [int(centroid[0]), int(centroid[1]), int(centroid[2])]
            centroids.append(label_int)
            curr_bbox = [[int(i[0].start), int(i[0].stop)], [int(i[1].start), int(i[1].stop)],
                         [int(i[2].start), int(i[2].stop)]]
            bboxes2.append(curr_bbox)
            counter += 1
        print("BBox extracted")
        #split labels into chunks to compute / thread
        x = np.arange(counter)
        list_chunks = np.array_split(x, nr_of_threads)
        print("Number of volumes to split:")
        print(len(list_chunks))
        print(list_chunks)
        #Parallel Split of input volume into subvolumes consisting of mask and image:
        print("Start splitting")
        #TODO: less splitting threads for lower memory, e.g. list_chunks[i][0], list_chunks[i+1][-1] for i in range(0, less nr_of_threads)
        volume_split = ray.get([split_volume.remote(image, mask, bboxes2, centroids, list_chunks[i][0], list_chunks[i][-1]) for i in range(0, nr_of_threads)])
        print("Volume split done")
        #Delete original image and mask for better memory consumption:
        del image
        del mask
        #run feature extraction in parallel:
        result_table = ray.get([run_feature_extractor.remote(volume_split[x], border_cells_array) for x in range(0, nr_of_threads)])
        # export data:
        counter = 0
        for i in result_table:
            for k in i:
                if counter == 0:
                    with open("{}.csv".format(output_filename), 'a') as outputFile:
                        writer = csv.DictWriter(outputFile, fieldnames=list(k.keys()), lineterminator='\n')
                        writer.writeheader()
                        writer.writerow(k)
                        outputFile.close()
                else:
                    with open("{}.csv".format(output_filename), 'a') as outputFile:
                        writer = csv.DictWriter(outputFile, fieldnames=list(k.keys()), lineterminator='\n')
                        writer.writerow(k)
                        outputFile.close()
                counter += 1






if __name__ == "__main__":
    # For timing:
    TicToc = TicTocGenerator()
    tic()
    ray.init()
    main(sys.argv[1:])
    ray.shutdown()
    toc()
