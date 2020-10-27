from SharedMethods import create_paths, get_dict_of_paths, get_organ_label
from SharedMethods2D import get_organized_data_train2D
from tensorflow.keras.models import load_model
import numpy as np
import SimpleITK as sitk
from UNet3D.Apply import resample_files_reverse, crop_files_reverse

# check voxel values against threshold and get segmentationmask
def get_segmentation_mask2D(img_arr, organ, thresh):
    # get respective label for the given organ
    organ_label = get_organ_label(organ)

    # create empty (only zeros) segmentation mask with same size as result_img_arr
    # should be 96,96,1
    result_img_arr = np.zeros((img_arr.shape[0],
                               img_arr.shape[1]))

    min_val =9999
    max_val= 0

    # loop over every voxel and create segmentation mask
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
                # todo: remove later
                if img_arr[x][y][0] > max_val: max_val = img_arr[x][y][0]
                if img_arr[x][y][0] < min_val: min_val = img_arr[x][y][0]
                # values > thresh will be labeled as segmentation mask
                # result_img_arr should have shape 64,64,64,1
                if img_arr[x][y][0] > thresh:
                    result_img_arr[x, y] = organ_label

    print("min: {}, max: {}".format(min_val, max_val))
    return result_img_arr


def get_segmentation_masks2D(results, z_stack_length, path_ref_files, target_path, organ, threshold):
    dict_ref_file_paths = get_dict_of_paths(path_ref_files)

    print("get segmentation masks")

    number_of_images = int(len(results)/z_stack_length)
    for j in range(0, number_of_images):
        print("J ####################################################")
        print(j)
        image_index = j
        # get the i-th reference file (patients in ascending order)
        curr_key = sorted(dict_ref_file_paths.keys())[image_index]

        st = (j * z_stack_length)
        en = z_stack_length*(image_index+1)

        result_img_arr_stack3D = []
        # loop through output of the neural network
        for i in range(st, en):
            result = results[i]

            # check voxel values against treshold and get segmentationmask
            result_img_arr2D = get_segmentation_mask2D(result, organ, threshold)
            result_img_arr_stack3D.append(result_img_arr2D)

        #stack all 2D arrays to a 3D array
        result_img_arr3D = np.dstack(result_img_arr_stack3D)

        result_img3D = sitk.GetImageFromArray(result_img_arr3D)

        # save cropped array as nifti file with patient number in name
        result_img3D.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(result_img3D, '{}seg{}.nii.gz'.format(target_path, curr_key))


def apply2D(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, ORGAN, THRESH):
    # create paths
    path_results, path_results_cropped, path_results_resampled, path_results_orig = create_paths(SAVE_PATH, "results")
    path_x_test_resampled = "{}Xtest/resampled/".format(SAVE_PATH)

    # get test data
    x_test = get_organized_data_train2D(path_x_test_resampled, DIMENSIONS)

    # load and apply U-Net on test data and get results in format Width, Height, Depth, Channels
    model = load_model("{}{}U-Net2D.h5".format(SAVE_PATH, ORGAN))
    results = model.predict(x_test, verbose=1)

    # number of 2D slice results belonging to a 3Dimage
    z_stack_length = DIMENSIONS[2]

    # generate segmentation masks from results
    get_segmentation_masks2D(results, z_stack_length, path_x_test_resampled, path_results_resampled, ORGAN, THRESH)

    # resample files to make them fit into the respective Bounding Box (??x??x??)
    resample_files_reverse(path_results_resampled, path_results_cropped, RRF_BB_PATH, ORGAN)

    # put area of interest back into original position
    crop_files_reverse(path_results_cropped, path_results_orig, RRF_BB_PATH, SCAN_PATH, ORGAN)