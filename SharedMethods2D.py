from SharedMethods import get_dict_of_paths
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def get_organized_data_train2D(path, DIMENSIONS, isSegmentation=False):
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files*DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files*DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.uint16)  # define X array

    print("data shape train")
    print(data.shape)

    # load files and transform into arrays (Width, Height, Channels) and put in data (array of arrays)
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img) # z, y, x ?????

        # loop over entire 3D image stack and add every 2D slice to data
        sagittal_shape = img_arr.shape[2] + 1
        axial_shape = img_arr.shape[1] + 1
        coronal_shape = img_arr.shape[0] + 1
        for z in range(1, axial_shape):
            #curr_arr2D = img_arr[:,:,z-1:z] #sagittal
            curr_arr2D = img_arr[:, z - 1:z, :]  # axial?
            #curr_arr2D = img_arr[z - 1:z, :, :]  # coronal?
            #img_arr = np.expand_dims(img_arr, axis=2)  # add a fourth dimension
            data[index] = np.expand_dims(curr_arr2D.squeeze(), -1)
            #plt.imshow(curr_arr2D[:, 0, :])
            #plt.show()
            index = index + 1

    return data


def get_organized_data_test2D(path, DIMENSIONS, isSegmentation=False):
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files, DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files, DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]), dtype=np.uint16)  # define X array

    print("data shape test")
    print(data.shape)
    # load files and transform into arrays (Width, Height, Channels) and put in data (array of arrays)
    index = 0  # keep extra index in case patients skip a number
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img)
        #img_arr = np.expand_dims(img_arr, axis=2)  # add a fourth dimension
        data[index] = img_arr
        index = index + 1

    return data