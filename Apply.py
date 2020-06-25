from tensorflow.keras.models import load_model
from SharedMethods import create_paths, \
    get_organized_data, get_organ_label, \
    get_dict_of_paths, find_patient_no_in_file_name, \
    resample_file, nifti_image_affine_reader, get_bb_coordinates, \
    bb_mm_to_vox
import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np


# check voxel values against threshold and get segmentationmask
def get_segmentation_mask(img_arr, organ, thresh):
    # get respective label for the given organ
    organ_label = get_organ_label(organ)

    # create empty (only zeros) segmentation mask with same siza as result_img_arr
    # should be 64,64,64
    result_img_arr = np.zeros((img_arr.shape[0],
                               img_arr.shape[1],
                               img_arr.shape[2]))

    # loop over every voxel and create segmentation mask
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            for z in range(img_arr.shape[2]):
                # values > thresh will be labeled as segmentation mask
                # result_img_arr should have shape 64,64,64,1
                if img_arr[x][y][z][0] > thresh:
                    result_img_arr[x, y, z] = organ_label

    return result_img_arr


def get_segmentation_masks(results, path_ref_files, target_path, organ, threshold):
    dict_ref_file_paths = get_dict_of_paths(path_ref_files)

    print("get segmentation masks")
    for i in range(0, len(results)):
        result = results[i]

        # get the i-th reference file (patients in ascending order)
        curr_key = sorted(dict_ref_file_paths.keys())[i]
        curr_file_path = dict_ref_file_paths[curr_key]

        # check voxel values against treshold and get segmentationmask
        result_img_arr = get_segmentation_mask(result, organ, threshold)

        # save cropped array as nifti file with patient number in name
        ref_file = nib.load(curr_file_path)    # reference file
        result_img = nib.Nifti1Image(result_img_arr, ref_file.affine, ref_file.header)
        nib.save(result_img, '{}seg{}.nii.gz'.format(target_path, curr_key))


def resample_files_reverse(path, target_path, bb_folder_path, ref_files_folder_path, ORGAN):
    # organize bb and reference files by patient number in dictionary
    bb_path_dict = get_dict_of_paths(bb_folder_path, ORGAN)
    ref_files_path_dict = get_dict_of_paths(ref_files_folder_path)

    print("reverse resampling files in '{}'".format(path))
    for file in os.scandir(path):
        patient_no = find_patient_no_in_file_name(file.name)

        # load respective original CT-Scan as reference and get some info
        ref_img_path = ref_files_path_dict[patient_no]
        ref_img = nib.load(ref_img_path)
        spacing, offset = nifti_image_affine_reader(ref_img)

        # get vox coordinates of respective bb to calculate dimensions
        bb_path = bb_path_dict[patient_no]
        bb_coords = get_bb_coordinates(bb_path)
        bb_coords_vox = bb_mm_to_vox(bb_coords, spacing, offset)

        # get start and end position of bb in CT-Scan
        # width
        x0_bb_coords_vox = int(bb_coords_vox[0])
        x1_bb_coords_vox = int(bb_coords_vox[1])
        # height
        y0_bb_coords_vox = int(bb_coords_vox[2])
        y1_bb_coords_vox = int(bb_coords_vox[3])
        # depth
        z0_bb_coords_vox = int(bb_coords_vox[4])
        z1_bb_coords_vox = int(bb_coords_vox[5])

        # calculate dimensions of bb
        width = x1_bb_coords_vox - x0_bb_coords_vox
        height = y1_bb_coords_vox - y0_bb_coords_vox
        depth = z1_bb_coords_vox - z0_bb_coords_vox

        # resample to original cut-out size (Depth, Height, Width)
        img = sitk.ReadImage(file.path)
        result_img = resample_file(img, depth, height, width)
        sitk.WriteImage(result_img, "{}{}".format(target_path, file.name))

    print("done. saved reverse resampled files to '{}'".format(target_path))


def crop_files_reverse(path, save_path, bb_folder_path, ref_files_folder_path, ORGAN):
    # organize bb and reference files by patient number in dictionary
    bb_path_dict = get_dict_of_paths(bb_folder_path, ORGAN)
    ref_files_path_dict = get_dict_of_paths(ref_files_folder_path)

    print("reverse cropping files in '{}'".format(path))
    for file in os.scandir(path):
        patient_no = find_patient_no_in_file_name(file.name)

        # load respective original CT-Scan as reference, get some info and create new array of same size
        ref_img_path = ref_files_path_dict[patient_no]
        ref_img = nib.load(ref_img_path)
        spacing, offset = nifti_image_affine_reader(ref_img)
        ref_img_arr = ref_img.get_fdata()
        result_img_arr = np.zeros((ref_img_arr.shape[0],
                                   ref_img_arr.shape[1],
                                   ref_img_arr.shape[2]))

        # get vox coordinates of respective bb to calculate dimensions
        bb_path = bb_path_dict[patient_no]
        bb_coords = get_bb_coordinates(bb_path)
        bb_coords_vox = bb_mm_to_vox(bb_coords, spacing, offset)

        # load file to be reverse cropped (w, h, d)
        img = nib.load(file.path)
        img_arr = img.get_fdata()

        # put the cut-out(cropped out area) back into its right position
        for x in range(img_arr.shape[0]):
            for y in range(img_arr.shape[1]):
                for z in range(img_arr.shape[2]):
                    if img_arr[x][y][z] > 0:
                        width = x + int(bb_coords_vox[0])
                        height = y + int(bb_coords_vox[2])
                        depth = z + int(bb_coords_vox[4])
                        result_img_arr[width, height, depth] = img_arr[x][y][z]

        # save cropped array as nifti file with patient number in name
        result_img = nib.Nifti1Image(result_img_arr, ref_img.affine, ref_img.header)
        nib.save(result_img, '{}{}.nii.gz'.format(save_path, patient_no))

    print("done. saved reverse cropped files to '{}'".format(save_path))


def apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, ORGAN, THRESH):
    # create paths
    path_results, path_results_resampled, path_results_cropped, path_results_orig = create_paths(SAVE_PATH, "results")
    path_x_test_resampled = "{}Xtest/resampled/".format(SAVE_PATH)

    # get test data
    x_test = get_organized_data(path_x_test_resampled, DIMENSIONS)

    # load U-Net
    model = load_model("{}{}U-Net.h5".format(SAVE_PATH, ORGAN))

    # apply U-Net on test data and get results in format Width, Height, Depth, Channels
    results = model.predict(x_test, verbose=1)

    # generate segmentation masks from results
    get_segmentation_masks(results, path_x_test_resampled, path_results_resampled, ORGAN, THRESH)

    # resample files to make them fit into the respective Bounding Box (??x??x??)
    resample_files_reverse(path_results_resampled, path_results_cropped, RRF_BB_PATH, SCAN_PATH, ORGAN)

    # put area of interest back into original position
    crop_files_reverse(path_results_cropped, path_results_orig, RRF_BB_PATH, SCAN_PATH, ORGAN)
