"""

"""

import os
import re
import numpy as np
import nibabel as nib
import SimpleITK as sitk  # https://simpleitk.readthedocs.io/en/master/index.html
from helpers import get_organ_label, get_bb_coordinates, \
    nifti_image_affine_reader, bb_mm_to_vox

'''_____________________________________________________________________________________________'''
'''|.................................Methods for single file/data..............................|'''
'''_____________________________________________________________________________________________'''


# crops out the bounding box volume of the given CT-image or segmentation
def crop_out_bb(img, box_path):
    # get numpy array from image
    array_of_img = img.get_fdata()

    # get bounding box coordinates
    bb_coords = get_bb_coordinates(box_path)

    # convert bounding box coordinates to voxel
    spacing, offset = nifti_image_affine_reader(img)
    bc_vox = bb_mm_to_vox(bb_coords, spacing, offset)
    x1 = int(bc_vox[0])
    x2 = int(bc_vox[1])
    y1 = int(bc_vox[2])
    y2 = int(bc_vox[3])
    z1 = int(bc_vox[4])
    z2 = int(bc_vox[5])

    # cut out bounding box of image
    array_of_img = array_of_img[x1:x2, y1:y2, z1:z2]

    return array_of_img


# resamples an image to the given target dimensions
# returns image with new dimensions
def resample_file(sitk_img, target_img_x, target_img_y, target_img_z):
    # get old Image size
    img_x = sitk_img.GetWidth()
    img_y = sitk_img.GetHeight()
    img_z = sitk_img.GetDepth()

    # get old Spacing (oSpac = old Voxel Size)
    oSpac = sitk_img.GetSpacing()
    vox_x = oSpac[0]
    vox_y = oSpac[1]
    vox_z = oSpac[2]

    # calculate and set new Spacing (nSpac = new Voxel Size)
    target_vox_x = img_x * vox_x / target_img_x
    target_vox_y = img_y * vox_y / target_img_y
    target_vox_z = img_z * vox_z / target_img_z
    nSpac = [target_vox_x, target_vox_y, target_vox_z]

    # define and apply  resampling filter
    resampler = sitk.ResampleImageFilter()  # create filter object
    resampler.SetReferenceImage(sitk_img)
    resampler.SetOutputOrigin([0, 0, 0])  # start of coordinate system of new image
    resampler.SetOutputSpacing(nSpac)  # spacing (voxel size) of new image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize((target_img_x, target_img_y, target_img_z))  # size of new image
    img_resampled_3D = resampler.Execute(sitk_img)  # apply filter object on old image

    return img_resampled_3D


# check voxel values against threshold and get segmentationmask
def get_segmentation_mask(result_img_arr, organ, thresh):

    organ_label = get_organ_label(organ)

    pred_map = np.zeros((64,64,64))

    # loop over every voxel
    for x in range(result_img_arr.shape[2]):    #64
        for y in range(result_img_arr.shape[1]):    #64
            for z in range(result_img_arr.shape[0]):    #64
                # values > thresh will be labeled as segmentation mask
                if result_img_arr[x][y][z][0] > thresh:
                    pred_map[z, y, x] = organ_label # TODO: z,y,x?

    return pred_map


'''_____________________________________________________________________________________________'''
'''|.................................Methods for all files/data................................|'''
'''_____________________________________________________________________________________________'''


# load all .nii files in a folder into a list that is sorted by ascending patient numbers
# assuming files contain patient numbers anywhere in the filename
# Beginning from 0. Without skipping a number*
def get_dict_of_files(path):
    dict_of_files = {}

    # go through every file in directory
    # (this isn't done in a organized way, files seem to be accessed rather randomly)
    for file in os.scandir(path):
        regex = re.compile(r'\d+')  # regex for finding numbers
        patient_no = int(regex.search(file.name).group(0))  # find patient number in file name
        nib_file = nib.load(file.path)  # load file with nibabel
        dict_of_files[patient_no] = nib_file  # write filepath into dictionary with patient number as key

    return dict_of_files


# load all paths to all .vtk files in a folder into a list that is sorted by ascending patient numbers
# but only use the .vtk files of a given organ
# assuming files contain patient numbers anywhere in the filename
# Beginning from 0. Without skipping a number*
# assuming files contain organ numbers followed by "_bb" anywhere in the filename
def get_dict_of_paths(path, organ=None):
    if organ is not None:
        organ_label = get_organ_label(organ)
        organ_label = "{}_bb".format(organ_label)

    dict_of_paths = {}

    # go through every file in directory
    # (this isn't done in a organized way, files seem to be accessed rather randomly)
    for file in os.scandir(path):
        regex = re.compile(r'\d+')  # regex for finding numbers
        patient_no = int(regex.search(file.name).group(0))  # find patient number in file name

        if organ is not None:
            if organ_label in file.name:
                dict_of_paths[patient_no] = file.path  # write filepath into dictionary with patient number as key
        else:
            dict_of_paths[patient_no] = file.path  # write filepath into dictionary with patient number as key

    return dict_of_paths


# raises an error message if all 3 are not the exact same length
def check_if_all_files_are_complete(scan_files, gt_seg_files, box_paths):
    len1 = len(scan_files)
    len2 = len(gt_seg_files)
    len3 = len(box_paths)
    if not (len1 == len2 == len3):
        raise ValueError('Every Patient needs a Scan, BBs for all organs and segmentations. One of theses is missing')
    else:
        return len1


# crops out the areas of interest defined by the given bounding boxes
# (where to organ is supposed to be)
def crop_out_bbs(dict_files, dict_box_paths, save_path, organ=None):
    # loop through all patients
    number_of_patients = len(dict_files)
    for i in range(0, number_of_patients):
        print(i)
        # access relevant patient files
        img = dict_files[i]
        box_path = dict_box_paths[i]

        # crop out box area
        array_cropped_img = crop_out_bb(img, box_path)

        if organ is not None:
            # extract segmentation of given organ
            # (filter out overlapping segmentations of other organs)
            organ_label = get_organ_label(organ)
            array_cropped_img[array_cropped_img < organ_label] = 0
            array_cropped_img[array_cropped_img > organ_label] = 0

        # save cropped array as nifti file with patient number in name
        new_img = nib.Nifti1Image(array_cropped_img, img.affine, img.header)
        nib.save(new_img, '{}{}.nii.gz'.format(save_path, "{}".format(i)))


# resamples all files in a folder to a given size and saves it to the given path
def resample_files(path, save_path, x, y, z):
    for file in os.scandir(path):
        sitk_img = sitk.ReadImage(file.path)
        resampled = resample_file(sitk_img, x, y, z)
        sitk.WriteImage(resampled, "{}{}".format(save_path, file.name))


def get_training_data(path, number_of_files, y_or_X="X"):

    # differentiate between training data/Scans (X_train) and labels/Segmentations (y_train)
    if y_or_X == "X":
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.uint8)  # define X_train array
    else:
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.bool)  # define y_train array

    # read files in patient order and write them into data
    dict_data = get_dict_of_paths(path)

    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_data.keys()):
        file_path = dict_data[key]
        img = sitk.ReadImage(file_path)  # load file
        arr_img = sitk.GetArrayFromImage(img)  # convert to numpy array
        arr_img = np.expand_dims(arr_img, axis=3)  # add a fourth dimension
        data[index] = arr_img  # add to data
        index = index + 1

    return data

