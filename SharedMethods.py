"""
"""

import shutil
import os, re
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from pathlib import Path

'''_____________________________________________________________________________________________'''
'''|.................................Helping Methods...........................................|'''
'''_____________________________________________________________________________________________'''


def find_patient_no_in_file_name(file_name):
    # find patient number in file name
    regex = re.compile(r'\d+')
    patient_no = int(regex.search(file_name).group(0))

    return patient_no


# given the name of an organ it returns the organs label number
def get_organ_label(organ):
    # define dictionary (to simulate switch-case)
    switcher = {
        "liver": 170,
        "left_kidney": 156,
        "right_kidney": 157,
        "spleen": 160,
        "pancreas": 150
    }

    # if given organ isn't a defined key, return "no valid organ"
    organ_label = switcher.get(organ, "no valid organ")

    # raise error message if no valid organ name was given
    if organ_label == "no valid organ":
        raise ValueError("'{}' is no valid organ name. Valid names are: "
                         "'liver', 'left_kidney', 'right_kidney', 'spleen', "
                         "'pancreas'".format(organ))
    else:
        return organ_label


# transform coordinate list (x,y,z) from mm-space to voxelspace
# return new coordinate list
def mm_to_vox(coord_list, spacing, offset):
    # calculate coordinates from mm-space to voxelspace
    x_vox = (coord_list[0] - offset[0]) / spacing[0]
    y_vox = (coord_list[1] - offset[1]) / spacing[1]
    z_vox = (coord_list[2] - offset[2]) / spacing[2]
    coord_vox = [x_vox, y_vox, z_vox]

    return coord_vox


# transform bounding coordinates from mm to vox
def bb_mm_to_vox(bb_coords, spacing, offset):
    # split coordinates into min and max coordinate values of bounding box
    bb_coords_min_mm = [bb_coords[0], bb_coords[2], bb_coords[4]]  # x1, y1, z1
    bb_coords_max_mm = [bb_coords[1], bb_coords[3], bb_coords[5]]  # x2, y2, z2

    # transform to vox coordinates
    bb_coords_min_vox = mm_to_vox(bb_coords_min_mm, spacing, offset)
    bb_coords_max_vox = mm_to_vox(bb_coords_max_mm, spacing, offset)

    # merge min and max coordinates again
    bb_coords_vox = []
    bb_coords_vox.append(bb_coords_min_vox[0])
    bb_coords_vox.append(bb_coords_max_vox[0])
    bb_coords_vox.append(bb_coords_min_vox[1])
    bb_coords_vox.append(bb_coords_max_vox[1])
    bb_coords_vox.append(bb_coords_min_vox[2])
    bb_coords_vox.append(bb_coords_max_vox[2])

    # if negative x spacing, switch x1 and x2
    if spacing[0] < 0:
        temp_space_x = bb_coords_vox[0]
        bb_coords_vox[0] = bb_coords_vox[1]
        bb_coords_vox[1] = temp_space_x

    # if negative y spacing, switch y1 and y2
    if spacing[1] < 0:
        temp_space_y = bb_coords_vox[2]
        bb_coords_vox[2] = bb_coords_vox[3]
        bb_coords_vox[3] = temp_space_y

    # if negative z spacing, switch z1 and z2
    if spacing[2] < 0:
        temp_space_z = bb_coords_vox[4]
        bb_coords_vox[4] = bb_coords_vox[5]
        bb_coords_vox[5] = temp_space_z

    return bb_coords_vox


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


def create_paths(SAVE_PATH, name):
    # define paths
    parent_path = "{}{}/".format(SAVE_PATH, name)
    path_cropped = "{}cropped/".format(parent_path)
    path_resampled = "{}resampled/".format(parent_path)
    path_orig = "{}orig/".format(parent_path)

    # delete preexisting folders
    shutil.rmtree(parent_path, ignore_errors=True)

    # create folders
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    Path(path_cropped).mkdir(parents=True, exist_ok=True)
    Path(path_resampled).mkdir(parents=True, exist_ok=True)
    Path(path_orig).mkdir(parents=True, exist_ok=True)

    return parent_path, path_cropped, path_resampled, path_orig


# read bounding box coordinates
def get_bb_coordinates(box_path):
    # open vtk file and get coordinates
    bb_file = open(box_path, 'r')
    lines = bb_file.readlines()

    # get coordinates
    numbers1 = lines[6].split()
    x0 = float(numbers1[0])
    y0 = float(numbers1[1])
    z0 = float(numbers1[2])

    # get coordinates
    numbers2 = lines[12].split()
    x1 = float(numbers2[0])
    y1 = float(numbers2[1])
    z1 = float(numbers2[2])

    # close file
    bb_file.close()

    # add coordinates to array
    bb_coords = [x0, x1, y0, y1, z0, z1]

    return bb_coords


# get image affine from header
# for coordinate system handling
# return spacing and offset
def nifti_image_affine_reader(img):
    # read spacing
    spacing_x = img.affine[0][0]
    spacing_y = img.affine[1][1]
    spacing_z = img.affine[2][2]
    spacing = [spacing_x, spacing_y, spacing_z]

    # read offset
    offset_x = img.affine[0][3]
    offset_y = img.affine[1][3]
    offset_z = img.affine[2][3]
    offset = [offset_x, offset_y, offset_z]

    return spacing, offset


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
    # resampler.SetOutputOrigin([0, 0, 0])  # start of coordinate system of new image
    resampler.SetOutputSpacing(new_spacing)  # spacing (voxel size) of new image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize((target_img_width, target_img_height, target_img_depth))  # size of new image
    result_img = resampler.Execute(sitk_img)  # apply filter object on old image

    return result_img


def get_organized_data(path, DIMENSIONS, isSegmentation=False):
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]), dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]), dtype=np.uint8)  # define X array

    # load files and transform into arrays (Width, Height, Depth, Channels) and put in data (array of arrays)
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = nib.load(file_path)
        img_arr = img.get_fdata()
        img_arr = np.expand_dims(img_arr, axis=3)  # add a fourth dimension
        data[index] = img_arr
        index = index + 1

    return data

