"""

"""

import os
import shutil
import re
import numpy as np
import nibabel as nib
import SimpleITK as sitk  # https://simpleitk.readthedocs.io/en/master/index.html
from helpers import get_organ_label, get_bb_coordinates, \
    nifti_image_affine_reader, bb_mm_to_vox, delete_recreate_folder, \
    move_files

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
    # width
    x1 = int(bc_vox[0])
    x2 = int(bc_vox[1])
    # height
    y1 = int(bc_vox[2])
    y2 = int(bc_vox[3])
    # depth
    z1 = int(bc_vox[4])
    z2 = int(bc_vox[5])

    # cut out bounding box of image
    array_of_img = array_of_img[x1:x2, y1:y2, z1:z2]

    return array_of_img


# resamples an image to the given target dimensions
# returns image with new dimensions
# INFO: if output image origin is set, resampling some seg (f.e.patient 33) doesn't work
# INFO: sitk loads and saves images in format: Depth, Height, Width
# INFO: but GetSpacing() returns Width, Height, Depth
# INFO: SetSize takes dimensions in format Width, Height, Depth.
# sitk keeps swichting formats...
def resample_file(sitk_img, target_img_depth, target_img_height, target_img_width):
    # get old Image size
    img_depth = sitk_img.GetDepth()
    img_height = sitk_img.GetHeight()
    img_width = sitk_img.GetWidth()

    # get old Spacing (oSpac = old Voxel Size)
    oSpac = sitk_img.GetSpacing()
    vox_width = oSpac[0]
    vox_height = oSpac[1]
    vox_depth = oSpac[2]

    print("old image w,h,d: {}".format(sitk_img.GetSize()))
    #print("old spacing w,h,d: {}".format(oSpac))

    # calculate and set new Spacing (nSpac = new Voxel Size)
    target_vox_width = img_width * vox_width / target_img_width
    target_vox_height = img_height * vox_height / target_img_height
    target_vox_depth = img_depth * vox_depth / target_img_depth
    nSpac = [target_vox_width, target_vox_height, target_vox_depth]
    #print("new spacing w,h,d: {}".format(nSpac))

    # define and apply resampling filter
    resampler = sitk.ResampleImageFilter()  # create filter object
    resampler.SetReferenceImage(sitk_img)
    #resampler.SetOutputOrigin([0, 0, 0])  # start of coordinate system of new image
    resampler.SetOutputSpacing(nSpac)  # spacing (voxel size) of new image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetSize((target_img_width, target_img_height, target_img_depth))  # size of new image
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
                if result_img_arr[x][y][z][0] > thresh:
                    pred_map[x, y, z] = organ_label

    return pred_map


'''_____________________________________________________________________________________________'''
'''|.................................Methods for all files/data................................|'''
'''_____________________________________________________________________________________________'''


# load all .nii files in a folder into a list that is sorted by ascending patient numbers
# assuming files contain patient numbers anywhere in the filename
# returns dict of nibable loaded files in format Width, Height, Depth
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
    delete_recreate_folder(save_path)

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
def resample_files(path, save_path, depth, height, width):
    delete_recreate_folder(save_path)

    print("resampling files in '{}'".format(path))
    counter = 0
    for file in os.scandir(path):
        # show feedback to user via print
        counter = counter + 1;
        if counter%10 == 0:
            print(".", end='')
        print(file.name)
        sitk_img = sitk.ReadImage(file.path)
        resampled = resample_file(sitk_img, depth, height, width)
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

    # load .nii files and transform files into numpy arrays
    # in format Width, Height, Depth, Channels
    # return array of numpy arrays
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_data.keys()):
        file_path = dict_data[key]
        img = nib.load(file_path)
        arr_img = img.get_fdata()
        arr_img = np.expand_dims(arr_img, axis=3)  # add a fourth dimension
        print(arr_img.shape)
        data[index] = arr_img
        index = index + 1

    return data


def get_segmentation_masks(results, path_original_files, save_path, organ, threshold):
    delete_recreate_folder(save_path)

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
    # move files from test folder to train folder
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
    delete_recreate_folder(save_path)

    print("reverse resampling files in '{}'".format(path))
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # load respective original CT-Scan and get some info
        orig_img = dict_scan_files[patient_no]  # returns a nibabel file
        spacing, offset = nifti_image_affine_reader(orig_img)

        # get vox coordinates of respective bb to calculate dimensions
        orig_bb = dict_bb_paths[patient_no]
        bb_coords = get_bb_coordinates(orig_bb)
        vox_bb = bb_mm_to_vox(bb_coords, spacing, offset)

        # get start and end position of bb in CT-Scan
        # width
        x1_vox_bb = int(vox_bb[1])
        x0_vox_bb = int(vox_bb[0])
        # height
        y1_vox_bb = int(vox_bb[3])
        y0_vox_bb = int(vox_bb[2])
        # depth
        z1_vox_bb = int(vox_bb[5])
        z0_vox_bb = int(vox_bb[4])

        # calculate dimensions of bb
        width = x1_vox_bb - x0_vox_bb
        height = y1_vox_bb - y0_vox_bb
        depth = z1_vox_bb - z0_vox_bb

        # resample to original cut-out size (Depth, Height, Width)
        sitk_img = sitk.ReadImage(file.path)
        resampled = resample_file(sitk_img, depth, height, width)
        sitk.WriteImage(resampled, "{}{}".format(save_path, file.name))

    print("done. saved reverse resampled files to '{}'".format(save_path))


def crop_files_reverse(path, save_path, dict_bb_paths, dict_scan_files):
    delete_recreate_folder(save_path)

    print("reverse cropping files in '{}'".format(path))
    for file in os.scandir(path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(file.name).group(0))

        # load respective original CT-Scan, get some info and create new array of same size
        orig_img = dict_scan_files[patient_no]  # returns a nibabel file (w, h, d)
        spacing, offset = nifti_image_affine_reader(orig_img)
        orig_img_arr = orig_img.get_fdata()
        result_img_arr = np.zeros((orig_img_arr.shape[0],
                                   orig_img_arr.shape[1],
                                   orig_img_arr.shape[2]))

        # get vox coordinates of respective bb to calculate dimensions
        orig_bb = dict_bb_paths[patient_no]
        bb_coords = get_bb_coordinates(orig_bb)
        vox_bb = bb_mm_to_vox(bb_coords, spacing, offset)

        # load file to be reverse cropped (w, h, d)
        nib_file = nib.load(file.path)
        nib_file_arr = nib_file.get_fdata()

        print("")
        print("orig/target file shape w, h, d: {}".format(orig_img_arr.shape))
        print("result file shape w, h, d: {}".format(result_img_arr.shape))
        print("nib file shape w, h, d: {}".format(nib_file_arr.shape))
        print("vox bb start in target w, h ,d: {}, {}, {}".format(int(vox_bb[0]), int(vox_bb[2]), int(vox_bb[4])))

        # put the cut-out(cropped out area) back into its right position
        for x in range(nib_file_arr.shape[0]):
            for y in range(nib_file_arr.shape[1]):
                for z in range(nib_file_arr.shape[2]):
                    if nib_file_arr[x][y][z] > 0:
                        x_result = x + int(vox_bb[0])
                        y_result = y + int(vox_bb[2])
                        z_result = z + int(vox_bb[4])
                        #print("x_result = {} + {} = {}".format(x, int(vox_bb[0]), x_result))
                        #print("y_result = {} + {} = {}".format(y, int(vox_bb[2]), y_result))
                        #print("z_result = {} + {} = {}".format(z, int(vox_bb[4]), z_result))
                        result_img_arr[x_result, y_result, z_result] = nib_file_arr[x][y][z]

        # save cropped array as nifti file with patient number in name
        new_img = nib.Nifti1Image(result_img_arr, orig_img.affine, orig_img.header)
        nib.save(new_img, '{}{}.nii.gz'.format(save_path, patient_no))

    print("done. saved reverse cropped files to '{}'".format(save_path))

