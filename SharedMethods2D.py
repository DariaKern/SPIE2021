from SharedMethods import get_dict_of_paths
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def get_organized_data_train2D(path, DIMENSIONS, direction, isSegmentation=False):
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
            for z in range(1, shape):
                curr_arr2D = img_arr[:, z - 1:z, :]  # coronal
                curr_arr2D = curr_arr2D.transpose(2, 0, 1)  # coronal
                data[index] = curr_arr2D
                #if index < 20:
                    #plt.imshow(curr_arr2D[:, :, 0])
                    #plt.show()
                index = index + 1

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


'''

def get_organized_data_test2D(path, DIMENSIONS, direction, isSegmentation=False):
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
'''

'''coronal_data = img_arr.transpose((0,1,2))
axial_data = img_arr.transpose((1,0,2))
sagittal_data = img_arr.transpose((2,0,1))

data += [axial_data]
'''