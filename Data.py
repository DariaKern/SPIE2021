"""

"""

import os
import re
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk  # https://simpleitk.readthedocs.io/en/master/index.html
from pathlib import Path
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
    #TODO: Before: width, height, depth. but resampled image was rotated
    # get old Image size
    img_x = sitk_img.GetDepth()
    img_y = sitk_img.GetHeight()
    img_z = sitk_img.GetWidth()
    print("resample file: {},{},{}".format(img_x, img_y, img_z))

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
    #INFO: if output image origin is set, resampling segmentations begging from patient 33 doesn't work
    #resampler.SetOutputOrigin([0, 0, 0])  # start of coordinate system of new image
    resampler.SetOutputSpacing(nSpac)  # spacing (voxel size) of new image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize((target_img_x, target_img_y, target_img_z))  # size of new image
    img_resampled_3D = resampler.Execute(sitk_img)  # apply filter object on old image

    return img_resampled_3D


# check voxel values against threshold and get segmentationmask
def get_segmentation_mask(result_img_arr, organ, thresh):
    # get respective label for the given organ
    organ_label = get_organ_label(organ)

    # create empty (only zeros) segmentation mask with same siza as result_img_arr
    # should be 64,64,64
    pred_map = np.zeros((result_img_arr.shape[0],
                         result_img_arr.shape[1],
                         result_img_arr.shape[2]))

    # loop over every voxel and create segmentation mask
    for x in range(result_img_arr.shape[0]):
        for y in range(result_img_arr.shape[1]):
            for z in range(result_img_arr.shape[2]):
                # values > thresh will be labeled as segmentation mask
                # result_img_arr should have shape 64,64,64,1
                if result_img_arr[x][y][z][0] >  thresh:
                    pred_map[z, y, x] = organ_label # TODO: WIESO? die resampleten Bilder sind ja jetzt auch nicht mehr verdreht

    return pred_map


'''_____________________________________________________________________________________________'''
'''|.................................Methods for all files/data................................|'''
'''_____________________________________________________________________________________________'''


# load all .nii files in a folder into a list that is sorted by ascending patient numbers
# assuming files contain patient numbers anywhere in the filename
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
        regex = re.compile(r'\d+')  # regex for finding numbers
        patient_no = int(regex.search(file.name).group(0))  # find patient number in file name

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
    print("delete and recreate '{}'".format(save_path))
    # delete and recreate folder with results
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # loop through all patients
    #number_of_patients = len(dict_files)

    print("cropping out bounding boxes (area of interest)")
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_files.keys()):
        # access relevant patient files
        img = dict_files[key]
        box_path = dict_box_paths[key]

        # show feedback to user via print
        index = index +1
        if index%10 == 0:
            print(".", end='')

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
        nib.save(new_img, '{}{}.nii.gz'.format(save_path, "{}".format(key)))

    print("done. saved cropped files to '{}'".format(save_path))


# resamples all files in a folder to a given size and saves it to the given path
def resample_files(path, save_path, x, y, z):
    # delete and recreate folder with results
    print("delete and recreate {}".format(save_path))
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)


    print("resampling files in '{}'".format(path))
    counter = 0
    for file in os.scandir(path):
        # show feedback to user via print
        counter = counter + 1;
        if counter%10 == 0:
            print(".", end='')

        sitk_img = sitk.ReadImage(file.path)
        resampled = resample_file(sitk_img, x, y, z)
        sitk.WriteImage(resampled, "{}{}".format(save_path, file.name))

    print("done. saved resampled files to '{}'".format(save_path))


def get_training_data(path, y_or_X="X"):
    # read files in patient order and write them into data
    dict_data = get_dict_of_paths(path)

    number_of_files = len(dict_data)

    # differentiate between training data/Scans (X_train) and labels/Segmentations (y_train)
    if y_or_X == "X":
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.uint8)  # define X_train array
    else:
        data = np.zeros((number_of_files, 64, 64, 64, 1), dtype=np.bool)  # define y_train array

    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_data.keys()):
        file_path = dict_data[key]
        img = sitk.ReadImage(file_path)  # load file
        arr_img = sitk.GetArrayFromImage(img)  # convert to numpy array
        arr_img = np.expand_dims(arr_img, axis=3)  # add a fourth dimension
        print(arr_img.shape)
        data[index] = arr_img  # add to data
        index = index + 1

    return data


def get_segmentation_masks(results, path_original_files, save_path, organ, threshold):
    # delete and recreate folder with results
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # read test files in patient order and write them into data
    dict_original_files_paths = get_dict_of_paths(path_original_files)

    print("get segmentation masks")
    for i in range(0, len(results)):
        result = results[i]
        curr_key = sorted(dict_original_files_paths.keys())[i]
        curr_file_path = dict_original_files_paths[curr_key]

        # check voxel values against treshold and get segmentationmask
        pred_map = get_segmentation_mask(result, organ, threshold)

        # save cropped array as nifti file with patient number in name
        input_file = nib.load(curr_file_path)    # reference file
        new_img = nib.Nifti1Image(pred_map, input_file.affine, input_file.header)
        nib.save(new_img, '{}seg{}.nii.gz'.format(save_path, curr_key))


def split_train_test(path_train, path_test, split):
    # move all files that may be in test to train
    for file in os.scandir(path_test):
        shutil.move(file.path, "{}{}".format(path_train, file.name))

    # check how many files are in train folder
    dict_train_file_paths = get_dict_of_paths(path_train)
    counter = len(dict_train_file_paths)

    # split into train and test
    test = int(counter * split)
    train = int(counter - test)
    print("splitting {} files into {} TRAIN and {} TEST files".format(counter, train, test))

    # sort files in train in descending order
    # and move x amount of test data to test folder
    test_count = 0
    for key in sorted(dict_train_file_paths.keys(), reverse=True):
        if test_count == test:
            break
        original_path = dict_train_file_paths[key]
        original_file_name = os.path.basename(original_path)
        shutil.move(original_path, "{}{}".format(path_test, original_file_name))
        test_count = test_count + 1


def resample_files_reverse(path, save_path, dict_bb_paths, dict_scan_files):
    # delete and recreate folder with results
    print("delete and recreate {}".format(save_path))
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print("resampling files in '{}'".format(path))
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # load respective original CT-Scan and get some info
        orig_img = dict_scan_files[patient_no]  # returns a nibabel file
        orig_img_arr = orig_img.get_fdata()
        spacing, offset = nifti_image_affine_reader(orig_img)

        # get vox coordinates of respective bb to calculate dimensions
        orig_bb = dict_bb_paths[patient_no]
        bb_coords = get_bb_coordinates(orig_bb)
        vox_bb = bb_mm_to_vox(bb_coords, spacing, offset)

        # get start and end position of bb in CT-Scan
        x1_vox_bb = int(vox_bb[1])
        x0_vox_bb = int(vox_bb[0])
        y1_vox_bb = int(vox_bb[3])
        y0_vox_bb = int(vox_bb[2])
        z1_vox_bb = int(vox_bb[5])
        z0_vox_bb = int(vox_bb[4])

        # calculate dimensions of bb
        y_vox_bb = y1_vox_bb - y0_vox_bb
        x_vox_bb = x1_vox_bb - x0_vox_bb
        z_vox_bb = z1_vox_bb - z0_vox_bb

        # resample to original cut-out size and save file
        sitk_img = sitk.ReadImage(file.path)
        resampled = resample_file(sitk_img, x_vox_bb, y_vox_bb, z_vox_bb)
        resmapled_img_arr = sitk.GetArrayFromImage(resampled)

        # put the cut-out(cropped out area) back into its right position
        # in an image with same size as original CT-Scan
        result_img_arr = np.zeros((orig_img_arr.shape[0],
                                   orig_img_arr.shape[1],
                                   orig_img_arr.shape[2]))

        #print out some stuff
        print("bb dimensions: {}, {}, {}".format(x_vox_bb, y_vox_bb, z_vox_bb))
        print("resampled image shape: {}".format(resmapled_img_arr.shape))
        print("result image shape: {}".format(result_img_arr.shape))
        print("vox_bb: {}, {}, {}".format(int(vox_bb[0]),int(vox_bb[2]),int(vox_bb[4])))

        for a in range(resmapled_img_arr.shape[0]):
            for b in range(resmapled_img_arr.shape[1]):
                for c in range(resmapled_img_arr.shape[2]):
                    if resmapled_img_arr[a][b][c] > 0:

                        x_result = c + y0_vox_bb
                        y_result = b + x0_vox_bb
                        z_result = a + z0_vox_bb
                        #print("x_result = {} + {} = {}".format(x, int(vox_bb[0]), x_result))
                        #print("y_result = {} + {} = {}".format(y, int(vox_bb[2]), y_result))
                        #print("z_result = {} + {} = {}".format(z, int(vox_bb[4]), z_result))
                        result_img_arr[x_result, y_result, z_result] = resmapled_img_arr[a][b][c]

        # save cropped array as nifti file with patient number in name
        new_img = nib.Nifti1Image(resmapled_img_arr, orig_img.affine, orig_img.header)
        nib.save(new_img, '{}{}.nii.gz'.format(save_path, "{}".format(patient_no)))
        #sitk.WriteImage(resampled, "{}{}".format(save_path, file.name))

    print("done. saved resampled files to '{}'".format(save_path))
