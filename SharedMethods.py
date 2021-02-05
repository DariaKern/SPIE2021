import shutil
import os, re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import vtk
import matplotlib.pyplot as plt
'''_____________________________________________________________________________________________'''
'''|.................................Helping Methods...........................................|'''
'''_____________________________________________________________________________________________'''


def find_patient_no_in_file_name(file_name):
    '''
       finds the patient number (1 or more digits from 0-9) in a file name and returns it as integer

       :param file_name: name of the file

       Usage::
           path = "/path to files"

           for file in os.scandir(path):
               find_patient_no_in_file_name(file.name)
       '''
    regex = re.compile(r'\d+') # 1 or more digits (0-9)
    patient_no = int(regex.search(file_name).group(0)) #if not an integer 1 and 10 could cause problems

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
        patient_no = find_patient_no_in_file_name(file.name)

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
    result_img.SetSpacing((2,2,2))

    return result_img


def get_organized_data(path, DIMENSIONS, isSegmentation=False):
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]), dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]), dtype=np.uint16)  # define X array

    # load files and transform into arrays (Width, Height, Depth, Channels) and put in data (array of arrays)
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.expand_dims(img_arr, axis=3)  # add a fourth dimension
        data[index] = img_arr
        index = index + 1

    return data


def get_bb_coordinates(box_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(box_path)
    reader.Update()
    box = reader.GetOutput()
    x_min, x_max, y_min, y_max, z_min, z_max = box.GetBounds()
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_organized_data_2D(path, DIMENSIONS, direction, isSegmentation=False):
    if isSegmentation:
        a = "segmentation"
    else:
        a = "CT"
    print("")
    print("slice 2D {} data in {} direction".format(a, direction))
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    # assuming dimensions are x*x*x and not x*a*b
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files*DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files*DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.uint16)  # define X array



    # load files and transform into arrays (Width, Height, Channels) and put in data (array of arrays)
    index = 0
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img) # z, y, x ?????

        switcher = {
            "axial": 0,  # axial
            "coronal": 1, # coronal
            "sagittal": 2 # sagittal
        }
        axis = switcher.get(direction, 0)
        shape = img_arr.shape[axis] + 1

        # loop over entire 3D image stack and add every 2D slice to data
        if direction == "coronal":
            #img_arr = np.flip(img_arr, 0)

            #TODO: schauen wie man es richtig dreht und spiegelt damit sie Segmentierung nachher nicht verdreht ist
            img_arr = np.flip(img_arr, 2)
            img_arr = np.flip(img_arr, 1)

            #count = shape
            for z in range(1, shape):
                #curr_arr2D = img_arr[:, z - 1:z, :]  # coronal
                curr_arr2D = img_arr[:, z-1:z, :]  # coronal
                curr_arr2D = curr_arr2D.transpose(2, 0, 1)  # coronal


                #print(index + (count-z-1))
                data[index] = curr_arr2D
                if index < 20:
                    plt.imshow(curr_arr2D[:, :, 0])
                    plt.show()
                index = index + 1
                #count = count - 1

        elif direction == "axial":
            for z in range(1, shape):
                curr_arr2D = img_arr[z - 1:z, :, :]  # axial
                curr_arr2D = curr_arr2D.transpose(1, 2, 0)  # axial
                data[index] = curr_arr2D
                #if index < 20:
                    #plt.imshow(curr_arr2D[:, :, 0])
                    #plt.show()
                index = index + 1

        elif direction == "sagittal":
            for z in range(1, shape):
                curr_arr2D = img_arr[:, :, z - 1:z]  # sagittal
                data[index] = curr_arr2D
                #if index < 20:
                    #plt.imshow(curr_arr2D[:, :, 0])
                    #plt.show()
                index = index + 1


    #img_arr = np.expand_dims(img_arr, axis=2)  # add a fourth dimension
    #data[index] = np.expand_dims(curr_arr2D.squeeze(), -1)
    #data = np.expand_dims(np.stack(data,0), -1)

    print("Data Shape")
    print(data.shape)
    return data