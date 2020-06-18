"""
SITK Image Basics
https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/plot_image.html
"""
import os
import shutil
import re
import numpy as np
import nibabel as nib
import SimpleITK as sitk  # https://simpleitk.readthedocs.io/en/master/index.html
from helpers import get_organ_label, get_bb_coordinates, \
    nifti_image_affine_reader, bb_mm_to_vox, delete_recreate_folder, \
    show_user_feedback

'''_____________________________________________________________________________________________'''
'''|.................................Methods for single file/data..............................|'''
'''_____________________________________________________________________________________________'''


# crops out the bounding box volume of the given CT-image or segmentation
def crop_out_bb(img, box_path):
    # get numpy array from image
    img_arr = img.get_fdata()

    # get bounding box coordinates
    bb_coords = get_bb_coordinates(box_path)

    # convert bounding box coordinates to voxel
    spacing, offset = nifti_image_affine_reader(img)
    bb_coords_vox = bb_mm_to_vox(bb_coords, spacing, offset)

    # width
    x0 = int(bb_coords_vox[0])
    x1 = int(bb_coords_vox[1])
    # height
    y0 = int(bb_coords_vox[2])
    y1 = int(bb_coords_vox[3])
    # depth
    z0 = int(bb_coords_vox[4])
    z1 = int(bb_coords_vox[5])

    # cut out bounding box of image
    result_img_arr = img_arr[x0:x1, y0:y1, z0:z1]

    return result_img_arr


# resamples an image to the given target dimensions
# returns image with new dimensions
# INFO: if output image origin is set, resampling some seg (f.e.patient 33) doesn't work
# INFO: SITK loads and saves images in format: Depth, Height, Width
# INFO: but GetSpacing() returns Width, Height, Depth
# INFO: SetSize takes dimensions in format Width, Height, Depth.
# SITK keeps switching formats...
def resample_file(sitk_img, target_img_depth, target_img_height, target_img_width):
    # get old Image size
    img_depth = sitk_img.GetDepth()
    img_height = sitk_img.GetHeight()
    img_width = sitk_img.GetWidth()

    # get old Spacing (old voxel size)
    old_spacing = sitk_img.GetSpacing()
    old_vox_width = old_spacing[0]
    old_vox_height = old_spacing[1]
    old_vox_depth = old_spacing[2]

    # calculate and set new Spacing (new voxel size)
    target_vox_width = img_width * old_vox_width / target_img_width
    target_vox_height = img_height * old_vox_height / target_img_height
    target_vox_depth = img_depth * old_vox_depth / target_img_depth
    new_spacing = [target_vox_width, target_vox_height, target_vox_depth]

    # define and apply resampling filter
    resampler = sitk.ResampleImageFilter()  # create filter object
    resampler.SetReferenceImage(sitk_img)
    #resampler.SetOutputOrigin([0, 0, 0])  # start of coordinate system of new image
    resampler.SetOutputSpacing(new_spacing)  # spacing (voxel size) of new image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize((target_img_width, target_img_height, target_img_depth))  # size of new image
    result_img = resampler.Execute(sitk_img)  # apply filter object on old image

    return result_img


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


'''_____________________________________________________________________________________________'''
'''|.................................Methods for all files/data................................|'''
'''_____________________________________________________________________________________________'''


# load all .nii files in a folder into a list that is sorted by ascending patient numbers
# assuming files contain patient numbers anywhere in the filename
# returns dict of nibabel loaded files in format Width, Height, Depth
def get_dict_of_files(path):
    dict_of_files = {}

    # go through every file in directory
    # (this isn't done in a organized way, files seem to be accessed rather randomly)
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # write filepath into dictionary with patient number as key
        nib_file = nib.load(file.path)
        dict_of_files[patient_no] = nib_file

    return dict_of_files


# load all paths to all .vtk files in a folder into a list that is sorted by ascending patient numbers
# but only use the .vtk files of a given organ
# assuming files contain patient numbers anywhere in the filename
# assuming files contain organ numbers followed by "_bb" anywhere in the filename
def get_dict_of_paths(path, organ=None):
    # if an organ was given, check if name is valid and get label for organ
    if organ is not None:
        organ_label = get_organ_label(organ)
        organ_label = "{}_bb".format(organ_label)

    dict_of_paths = {}

    # go through every file in directory
    # (this isn't done in a organized way, files seem to be accessed rather randomly)
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        if organ is not None:
            # write filepath (to file that contains the organ label) into dictionary with patient number as key
            if organ_label in file.name:
                dict_of_paths[patient_no] = file.path
        else:
            # write filepath into dictionary with patient number as key
            dict_of_paths[patient_no] = file.path

    return dict_of_paths


# raises an error message if all 3 are not the exact same length
def check_if_all_files_are_complete(scan_files, gt_seg_files, box_paths):
    # get number of elements
    len1 = len(scan_files)
    len2 = len(gt_seg_files)
    len3 = len(box_paths)

    #  check for error
    if not (len1 == len2 == len3):
        raise ValueError('Every Patient needs a Scan, BBs for all organs and segmentations. '
                         'One of theses is missing. Scans: {}, BBs: {}, seg: {}'.format(len1, len3, len2))
    else:
        return len1


# crops out the areas of interest defined by the given bounding boxes
# (where to organ is supposed to be)
def crop_out_bbs(dict_files, dict_box_paths, save_path, organ=None):
    delete_recreate_folder(save_path)

    print("cropping out bounding boxes (area of interest)")
    feedback = 0
    for key in sorted(dict_files.keys()):
        # show feedback to user via print
        show_user_feedback(feedback)

        # access relevant patient files
        img = dict_files[key]
        bb_path = dict_box_paths[key]

        # crop out box area
        result_img_arr = crop_out_bb(img, bb_path)

        if organ is not None:
            # extract segmentation of given organ
            # (filter out overlapping segmentations of other organs)
            organ_label = get_organ_label(organ)
            result_img_arr[result_img_arr < organ_label] = 0
            result_img_arr[result_img_arr > organ_label] = 0

        # save cropped array as nifti file with patient number in name
        result_img = nib.Nifti1Image(result_img_arr, img.affine, img.header)
        nib.save(result_img, '{}{}.nii.gz'.format(save_path, "{}".format(key)))

    print("done. saved cropped files to '{}'".format(save_path))


# resamples all files in a folder to a given size and saves it to the given path
def resample_files(path, save_path, depth, height, width):
    delete_recreate_folder(save_path)

    print("resampling files in '{}'".format(path))
    feedback = 0
    for file in os.scandir(path):
        # show feedback to user via print
        show_user_feedback(feedback)

        orig_img = sitk.ReadImage(file.path)
        result_img = resample_file(orig_img, depth, height, width)
        sitk.WriteImage(result_img, "{}{}".format(save_path, file.name))

    print("done. saved resampled files to '{}'".format(save_path))


def get_training_data(path, y_or_X="X"):
    # read file paths in patient order
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between training data/Scans (X_train) and labels/Segmentations (y_train)
    number_of_files = len(dict_file_paths)
    if y_or_X == "X":
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.uint8)  # define X_train array
    else:
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.bool)  # define y_train array

    # load .nii files and transform files into numpy arrays
    # in format Width, Height, Depth, Channels
    # return array of numpy arrays
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = nib.load(file_path)
        img_arr = img.get_fdata()
        img_arr = np.expand_dims(img_arr, axis=3)  # add a fourth dimension
        data[index] = img_arr
        index = index + 1

    return data


def get_segmentation_masks(results, path_ref_files, save_path, organ, threshold):
    delete_recreate_folder(save_path)

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
        nib.save(result_img, '{}seg{}.nii.gz'.format(save_path, curr_key))


def split_train_test(path_train, path_test, split):
    # move files from test folder to train folder
    for file in os.scandir(path_test):
        shutil.move(file.path, "{}{}".format(path_train, file.name))

    # check how many files are in train folder now
    dict_train_file_paths = get_dict_of_paths(path_train)
    counter = len(dict_train_file_paths)

    # split into train and test
    test = int(counter * split)
    train = int(counter - test)
    print("splitting {} files into {} TRAIN and {} TEST files".format(counter, train, test))

    # sort files in train in descending order
    # and move x amount of test data to test folder
    x_amount = 0
    for key in sorted(dict_train_file_paths.keys(), reverse=True):
        if x_amount == test: break;
        x_amount = x_amount + 1

        file_path = dict_train_file_paths[key]
        file_name = os.path.basename(file_path)
        target_path = "{}{}".format(path_test, file_name)
        shutil.move(file_path, target_path)


def resample_files_reverse(path, save_path, dict_bb_paths, dict_scan_files):
    delete_recreate_folder(save_path)

    print("reverse resampling files in '{}'".format(path))
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # load respective original CT-Scan as reference and get some info
        ref_img = dict_scan_files[patient_no]  # returns a nibabel file
        spacing, offset = nifti_image_affine_reader(ref_img)

        # get vox coordinates of respective bb to calculate dimensions
        bb_path = dict_bb_paths[patient_no]
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
        sitk.WriteImage(result_img, "{}{}".format(save_path, file.name))

    print("done. saved reverse resampled files to '{}'".format(save_path))


def crop_files_reverse(path, save_path, dict_bb_paths, dict_scan_files):
    delete_recreate_folder(save_path)

    print("reverse cropping files in '{}'".format(path))
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # load respective original CT-Scan as reference, get some info and create new array of same size
        ref_img = dict_scan_files[patient_no]  # returns a nibabel file (w, h, d)
        spacing, offset = nifti_image_affine_reader(ref_img)
        ref_img_arr = ref_img.get_fdata()
        result_img_arr = np.zeros((ref_img_arr.shape[0],
                                   ref_img_arr.shape[1],
                                   ref_img_arr.shape[2]))

        # get vox coordinates of respective bb to calculate dimensions
        bb_path = dict_bb_paths[patient_no]
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

