from tensorflow.keras.models import load_model
from SharedMethods import create_paths, \
    get_organized_data, get_organ_label, \
    get_dict_of_paths, find_patient_no_in_file_name, \
    resample_file, get_bb_coordinates
import os
import SimpleITK as sitk
import numpy as np


# check voxel values against threshold and get segmentationmask
def get_segmentation_mask(img_arr, organ, thresh):
    # get respective label for the given organ
    organ_label = get_organ_label(organ)

    # create empty (only zeros) segmentation mask with same size as result_img_arr
    # should be 64,64,64
    result_img_arr = np.zeros((img_arr.shape[0],
                               img_arr.shape[1],
                               img_arr.shape[2]))

    min_val =9999
    max_val= 0

    # loop over every voxel and create segmentation mask
    #'''
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            for z in range(img_arr.shape[2]):
                # todo: remove later
                if img_arr[x][y][z][0] > max_val: max_val = img_arr[x][y][z][0]
                if img_arr[x][y][z][0] < min_val: min_val = img_arr[x][y][z][0]
                # values > thresh will be labeled as segmentation mask
                # result_img_arr should have shape 64,64,64,1
                if img_arr[x][y][z][0] > thresh:
                    result_img_arr[x, y, z] = organ_label
    #'''
    #result_img_arr[img_arr > thresh] = organ_label
    #print("min: {}, max: {}".format(min_val, max_val))
    return result_img_arr


def get_segmentation_masks(results, path_ref_files, target_path, organ, threshold):
    dict_ref_file_paths = get_dict_of_paths(path_ref_files)

    print("get segmentation masks")
    # loop through output of the neural network
    for i in range(0, len(results)):
        result = results[i]

        # get the i-th reference file (patients in ascending order)
        curr_key = sorted(dict_ref_file_paths.keys())[i]
        print(curr_key)
        ref_file_path = dict_ref_file_paths[curr_key]
        ref_file = sitk.ReadImage(ref_file_path)

        # check voxel values against treshold and get segmentationmask
        result_img_arr = get_segmentation_mask(result, organ, threshold)
        result_img = sitk.GetImageFromArray(result_img_arr)

        # save cropped array as nifti file with patient number in name
        #TODO: set original spacing (currently spacing of 2,2,2 expected)
        spacing = ref_file.GetSpacing()
        print(spacing)
        result_img.SetSpacing((2.0, 2.0, 2.0))
        result_img.SetSpacing(spacing)
        sitk.WriteImage(result_img, '{}seg{}.nii.gz'.format(target_path, curr_key))


def resample_files_reverse(path, target_path, bb_folder_path, ORGAN, ref_files_folder_path=None):
    # organize bb and reference files by patient number in dictionary
    bb_path_dict = get_dict_of_paths(bb_folder_path, ORGAN)
    if ref_files_folder_path is None:
        #TODO
        ref_files_folder_path = "/home/daria/Desktop/Data/Daria/Workflow/WF/Xtest/cropped/"
        #ref_files_folder_path = "/home/daria/Desktop/Data/Daria/PREP/Xtest/cropped/"
    ref_files_path_dict = get_dict_of_paths(ref_files_folder_path)

    print("")
    print("reverse resampling files in '{}'".format(path))
    for file in os.scandir(path):
        patient_no = find_patient_no_in_file_name(file.name)
        print("patient file #{}".format(patient_no))

        # load respective cropped CT-Scan as reference and get size
        ref_img_path = ref_files_path_dict[patient_no]
        ref_img = sitk.ReadImage(ref_img_path)
        target_dim = ref_img.GetSize()

        # resample to original cut-out size (Depth, Height, Width)
        img = sitk.ReadImage(file.path)
        result_img = resample_file(img, target_dim[2], target_dim[1], target_dim[0])
        sitk.WriteImage(result_img, "{}{}".format(target_path, file.name))

    print("done. saved reverse resampled files to '{}'".format(target_path))


def crop_files_reverse(path, target_path, bb_folder_path, ref_files_folder_path, ORGAN):
    # organize bb and reference files by patient number in dictionary
    bb_path_dict = get_dict_of_paths(bb_folder_path, ORGAN)
    ref_files_path_dict = get_dict_of_paths(ref_files_folder_path)

    print("")
    print("reverse cropping files in '{}'".format(path))
    for file in os.scandir(path):
        patient_no = find_patient_no_in_file_name(file.name)
        print("patient file #{}".format(patient_no))

        # load respective original CT-Scan as reference, get some info and create new array of same size
        ref_img_path = ref_files_path_dict[patient_no]
        ref_img = sitk.ReadImage(ref_img_path)
        ref_img_arr = sitk.GetArrayFromImage(ref_img)

        result_img_arr = np.zeros((ref_img_arr.shape[0],
                                   ref_img_arr.shape[1],
                                   ref_img_arr.shape[2]))

        # load file that has to be put back into its original position (reverse cropped)
        img = sitk.ReadImage(file.path)
        img_arr = sitk.GetArrayFromImage(img)

        # read bb and coordinates
        box_path = bb_path_dict[patient_no]
        x_min, x_max, y_min, y_max, z_min, z_max = get_bb_coordinates(box_path)

        # transform physical space to index points
        p_min = ref_img.TransformPhysicalPointToIndex((x_min, y_min, z_min))
        p_max = ref_img.TransformPhysicalPointToIndex((x_max, y_max, z_max))

        # x and y are negative because of nifti file orientation -x, -y, z
        # -> swap min and max and multiply by -1 to make positive
        # also substract 1 from all max...only god knows why
        x_min = (p_max[0] - 1) * -1
        x_max = (p_min[0]) * -1
        y_min = (p_max[1] - 1) * -1
        y_max = (p_min[1]) * -1
        z_min = p_min[2]
        z_max = p_max[2] - 1

        #print(x_min, x_max, y_min, y_max, z_min, z_max)

        # put the cut-out(cropped out area) back into its right position
        z_length = img_arr.shape[2]
        y_length = img_arr.shape[1]
        x_length = img_arr.shape[0]
        for x in range(x_length):
            for y in range(y_length):
                for z in range(z_length):
                    result_img_arr[z_min + x][y_min + y][x_min + z] = img_arr[x][y][z]

        # save nifti file with patient number in name
        result_img = sitk.GetImageFromArray(result_img_arr)
        #TODO
        spacing = ref_img.GetSpacing()
        print(spacing)
        result_img.SetSpacing((2,2,2))
        result_img.SetSpacing(spacing)
        result_img = sitk.Cast(result_img, sitk.sitkUInt16)
        sitk.WriteImage(result_img, "{}{}".format(target_path, file.name))

    print("done. saved reverse cropped files to '{}'".format(target_path))


def apply_3DUnet(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, ORGAN, THRESH):
    # create paths
    path_results, path_results_cropped, path_results_resampled, path_results_orig = create_paths(SAVE_PATH, "results")
    path_x_test_resampled = "{}Xtest/resampled/".format(SAVE_PATH)

    # get test data
    x_test = get_organized_data(path_x_test_resampled, DIMENSIONS)

    # load and apply U-Net on test data and get results in format Width, Height, Depth, Channels
    model = load_model("{}{}U-Net.h5".format(SAVE_PATH, ORGAN))
    results = model.predict(x_test, verbose=1)

    # generate segmentation masks from results
    get_segmentation_masks(results, path_x_test_resampled, path_results_resampled, ORGAN, THRESH)

    # resample files to make them fit into the respective Bounding Box (??x??x??)
    resample_files_reverse(path_results_resampled, path_results_cropped, RRF_BB_PATH, ORGAN, None)

    # put area of interest back into original position
    crop_files_reverse(path_results_cropped, path_results_orig, RRF_BB_PATH, SCAN_PATH, ORGAN)
