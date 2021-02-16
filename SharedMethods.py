import shutil
import os, re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import vtk
import matplotlib.pyplot as plt
import collections

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
    regex = re.compile(r'\d+')  # 1 or more digits (0-9)
    patient_no = int(regex.search(file_name).group(0))  # if not an integer 1 and 10 could cause problems

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
    result_img.SetSpacing((2, 2, 2))

    return result_img


def get_organized_data(path, DIMENSIONS, isSegmentation=False):
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]),
                        dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3]),
                        dtype=np.uint16)  # define X array

    # load files and transform into arrays (Width, Height, Depth, Channels) and put in img_arr (array of arrays)
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
    print("slice 2D {} img_arr in {} direction".format(a, direction))
    # organize file paths by patient number in dictionary
    dict_file_paths = get_dict_of_paths(path)

    # differentiate between images (X) and labels/Segmentations (y)
    # assuming dimensions are x*x*x and not x*a*b
    number_of_files = len(dict_file_paths)
    if isSegmentation:
        data = np.zeros((number_of_files * DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]),
                        dtype=np.bool)  # define y array
    else:
        data = np.zeros((number_of_files * DIMENSIONS[2], DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3]),
                        dtype=np.uint16)  # define X array

    # load files and transform into arrays (Width, Height, Channels) and put in img_arr (array of arrays)
    index = 0
    for key in sorted(dict_file_paths.keys()):
        file_path = dict_file_paths[key]
        img = sitk.ReadImage(file_path)
        img_arr = sitk.GetArrayFromImage(img)  # z, y, x ?????

        switcher = {
            "axial": 0,  # axial
            "coronal": 1,  # coronal
            "sagittal": 2  # sagittal
        }
        axis = switcher.get(direction, 0)
        shape = img_arr.shape[axis] + 1

        # loop over entire 3D image stack and add every 2D slice to img_arr
        if direction == "coronal":
            # img_arr = np.flip(img_arr, 0)

            # TODO: schauen wie man es richtig dreht und spiegelt damit sie Segmentierung nachher nicht verdreht ist
            img_arr = np.flip(img_arr, 2)
            img_arr = np.flip(img_arr, 1)

            # count = shape
            for z in range(1, shape):
                # curr_arr2D = img_arr[:, z - 1:z, :]  # coronal
                curr_arr2D = img_arr[:, z - 1:z, :]  # coronal
                curr_arr2D = curr_arr2D.transpose(2, 0, 1)  # coronal

                # print(index + (count-z-1))
                data[index] = curr_arr2D
                if index < 20:
                    plt.imshow(curr_arr2D[:, :, 0])
                    plt.show()
                index = index + 1
                # count = count - 1

        elif direction == "axial":
            for z in range(1, shape):
                curr_arr2D = img_arr[z - 1:z, :, :]  # axial
                curr_arr2D = curr_arr2D.transpose(1, 2, 0)  # axial
                data[index] = curr_arr2D
                # if index < 20:
                # plt.imshow(curr_arr2D[:, :, 0])
                # plt.show()
                index = index + 1

        elif direction == "sagittal":
            for z in range(1, shape):
                curr_arr2D = img_arr[:, :, z - 1:z]  # sagittal
                data[index] = curr_arr2D
                # if index < 20:
                # plt.imshow(curr_arr2D[:, :, 0])
                # plt.show()
                index = index + 1

    # img_arr = np.expand_dims(img_arr, axis=2)  # add a fourth dimension
    # img_arr[index] = np.expand_dims(curr_arr2D.squeeze(), -1)
    # img_arr = np.expand_dims(np.stack(img_arr,0), -1)
    return data


'''
RRF
'''


# the training subset is located in the middle of the image
# the axial center of the image has to be found
# an area around this center has to be allocated
# returns min, max for x, y
def training_subset_generator(img, p_img_arr):
    img_arr = p_img_arr.copy()
    '''
        blablab returns the axial center of a 3D image in format x, y, z
        TODO

       :param img_arr: numpy array (z, y, x)

       Usage::
           TODO
    '''

    # calculate axial center
    rows = img_arr.shape[1]  # y
    columns = img_arr.shape[2]  # x
    z_axis = img_arr.shape[0]  # z
    # axial_center = [z_axis / 2, rows / 2, columns / 2]  # z, y, x
    axial_center = [columns / 2, rows / 2, z_axis / 2]  # x, y, z
    # print('axial center: ', axial_center)

    # transform index to physical space (mm)
    axial_center_mm = img.TransformIndexToPhysicalPoint(
        (int(axial_center[0]), int(axial_center[1]), int(axial_center[2])))
    # print('axial center mm: ', axial_center_mm)

    # calculate voxel training subset, +-100mm
    training_xyz_min_mm = [axial_center_mm[0] - 15, axial_center_mm[1] - 15, axial_center_mm[2] - 15]
    training_xyz_max_mm = [axial_center_mm[0] + 15, axial_center_mm[1] + 15, axial_center_mm[2] + 15]
    # print('training xyz max min mm: ', training_xyz_max_mm, training_xyz_min_mm)

    # transform physical space (mm) to index
    training_xyz_min = img.TransformPhysicalPointToContinuousIndex(training_xyz_min_mm)
    training_xyz_max = img.TransformPhysicalPointToContinuousIndex(training_xyz_max_mm)

    t_min = [int(training_xyz_min[0]), int(training_xyz_min[1]), int(training_xyz_min[2])]
    t_max = [int(training_xyz_max[0]), int(training_xyz_max[1]), int(training_xyz_max[2])]

    return t_min, t_max


def calc_bb_coordinates(img, p_redictions, p_training_xyz_min, p_training_xyz_max):
    predictions = p_redictions.copy()
    training_xyz_min = p_training_xyz_min.copy()
    training_xyz_max = p_training_xyz_max.copy()

    # init lists for coordinates
    all_x_min = []
    all_x_max = []
    all_y_min = []
    all_y_max = []
    all_z_min = []
    all_z_max = []

    pred_Counter = 0
    # loop over voxels in this window
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                temp_train_coord = [training_x, training_y, training_z]

                # transform index to physical space (mm)
                temp_train_coord_mm = img.TransformIndexToPhysicalPoint(temp_train_coord)

                # set y_pred as offset
                all_x_min.append(temp_train_coord_mm[0] - predictions[pred_Counter][0])
                all_x_max.append(temp_train_coord_mm[0] - predictions[pred_Counter][1])
                all_y_min.append(temp_train_coord_mm[1] - predictions[pred_Counter][2])
                all_y_max.append(temp_train_coord_mm[1] - predictions[pred_Counter][3])
                all_z_min.append(temp_train_coord_mm[2] - predictions[pred_Counter][4])
                all_z_max.append(temp_train_coord_mm[2] - predictions[pred_Counter][5])

                pred_Counter += 1
    return all_x_min, all_x_max, all_y_min, all_y_max, all_z_min, all_z_max


# loop used to apply the trained models on new img_arr
# variant of loop_subset_training
def loop_apply(p_img_arr, training_xyz_min, training_xyz_max, p_displacement):
    displacement = p_displacement.copy()
    img_arr = p_img_arr.copy()

    final_feature_vec = []
    # loop over voxels in this window
    for z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create mean feature boxes
                temp_feature_vec = feature_box_generator(img_arr, x, y, z, displacement)

                # add feature vector of current voxel to the complete feature vector
                final_feature_vec.append(temp_feature_vec)

    return final_feature_vec


def displacement_calc(img, p_training_xyz_min):
    training_xyz_min = p_training_xyz_min.copy()

    # transform index to physical space (mm)
    training_xyz_min_mm = img.TransformIndexToPhysicalPoint(training_xyz_min)

    # displacement
    displacement_mm = [0, 0, 0]
    displacement_mm[0] = training_xyz_min_mm[0] + 25
    displacement_mm[1] = training_xyz_min_mm[1] + 25
    displacement_mm[2] = training_xyz_min_mm[2] + 25

    # transform physical space (mm) to index
    displacement = img.TransformPhysicalPointToContinuousIndex(displacement_mm)

    xyz = [0, 0, 0]
    xyz[0] = abs(training_xyz_min[0] - int(displacement[0]))
    xyz[1] = abs(training_xyz_min[1] - int(displacement[1]))
    xyz[2] = abs(training_xyz_min[2] - int(displacement[2]))
    #print('dari trainxyz {} - displacement {} = {}'.format(training_xyz_min, displacement, xyz))

    return xyz


def bb_finalize(x_min_test, x_max_test, y_min_test, y_max_test, z_min_test, z_max_test):
    x_min_test = np.around(x_min_test)
    x_max_test = np.around(x_max_test)
    y_min_test = np.around(y_min_test)
    y_max_test = np.around(y_max_test)
    z_min_test = np.around(z_min_test)
    z_max_test = np.around(z_max_test)

    # find the top 3 predicted coords for each wall
    c_x_min = collections.Counter(x_min_test).most_common(3)
    c_x_max = collections.Counter(x_max_test).most_common(3)
    c_y_min = collections.Counter(y_min_test).most_common(3)
    c_y_max = collections.Counter(y_max_test).most_common(3)
    c_z_min = collections.Counter(z_min_test).most_common(3)
    c_z_max = collections.Counter(z_max_test).most_common(3)

    new_bb = []
    # this version creates a bigger bb by selecting extreme values
    new_bb.append(np.min([c_x_min[0][0], c_x_min[1][0], c_x_min[2][0]]))
    new_bb.append(np.max([c_x_max[0][0], c_x_max[1][0], c_x_max[2][0]]))
    new_bb.append(np.min([c_y_min[0][0], c_y_min[1][0], c_y_min[2][0]]))
    new_bb.append(np.max([c_y_max[0][0], c_y_max[1][0], c_y_max[2][0]]))
    new_bb.append(np.min([c_z_min[0][0], c_z_min[1][0], c_z_min[2][0]]))
    new_bb.append(np.max([c_z_max[0][0], c_z_max[1][0], c_z_max[2][0]]))

    # uncomment for majority vote
    # new_bb.append(c_x_min[0][0])
    # new_bb.append(c_x_max[0][0])
    # new_bb.append(c_y_min[0][0])
    # new_bb.append(c_y_max[0][0])
    # new_bb.append(c_z_min[0][0])
    # new_bb.append(c_z_max[0][0])
    return new_bb


def make_bounding_box(new_bb, file, save_path):
    patient_number = find_patient_no_in_file_name(file.name)
    bb_name = "{}{}_{}_bb.vtk".format(save_path, patient_number, new_bb[6])

    x_min = new_bb[0]
    x_max = new_bb[1]
    y_min = new_bb[2]
    y_max = new_bb[3]
    z_min = new_bb[4]
    z_max = new_bb[5]
    bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

    # define bb as cube
    vtk_cube = vtk.vtkCubeSource()
    vtk_cube.SetBounds(bounds)
    vtk_cube.Update()
    output = vtk_cube.GetOutput()

    # save bounding box object to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(output)
    writer.SetFileName(bb_name)
    writer.Update()


def get_final_vectors(img, img_arr, training_xyz_min, training_xyz_max, bb_coordinates, displacement,
                      final_feature_vec, final_offset_vec):
    # loop over voxels in this window
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                temp_train_coord = []
                temp_train_coord.append(training_x)
                temp_train_coord.append(training_y)
                temp_train_coord.append(training_z)
                #print(temp_train_coord)

                # transform index to physical space (mm)
                temp_train_coord_mm = img.TransformIndexToPhysicalPoint(temp_train_coord)
                #print(temp_train_coord_mm)
                # calculate offset between bounding box and voxel(mm) in mm
                # v-bc = (vx, vx, vy, vy, vz, vz)
                bb_offset = []
                bb_offset.append(temp_train_coord_mm[0] - bb_coordinates[0])
                bb_offset.append(temp_train_coord_mm[0] - bb_coordinates[1])
                bb_offset.append(temp_train_coord_mm[1] - bb_coordinates[2])
                bb_offset.append(temp_train_coord_mm[1] - bb_coordinates[3])
                bb_offset.append(temp_train_coord_mm[2] - bb_coordinates[4])
                bb_offset.append(temp_train_coord_mm[2] - bb_coordinates[5])

                # create mean feature boxes
                temp_feature_vec = feature_box_generator(img_arr, training_x, training_y, training_z, displacement)

                # add feature vector of current voxel to the complete feature vector
                final_feature_vec.append(temp_feature_vec)
                final_offset_vec.append(bb_offset)

    return final_feature_vec, final_offset_vec


def feature_box_generator(p_img_arr, train_x, train_y, train_z, p_displacement):
    displacement = p_displacement.copy()
    img_arr = p_img_arr.copy()



    # create feature boxes in each direction
    iterator_disp_x = displacement[0] // 2
    iterator_disp_y = displacement[1] // 2
    iterator_disp_z = displacement[2] // 2

    # init array for feature vector of selected voxel
    temp_feature_vec = []

    # generate feature box around selected voxel and append mean to feature vector
    feature_box_0 = img_arr[
                    train_z - 2:train_z + 3,
                    train_y - 2:train_y + 3,
                    train_x - 2:train_x + 3]

    temp_feature_vec.append(np.mean(feature_box_0))

    # generate 26 feature boxes in a certain distance to the selected pixel
    # add half of the displacement to the displacement and generate another 26 boxes
    # this can be reapeated
    # in this case 3 iterations with 1+26+26+26=79 feature boxes
    counter = 0
    while counter < 3:
        if counter > 0:
            displacement[0] = displacement[0] + iterator_disp_x
            displacement[1] = displacement[1] + iterator_disp_y
            displacement[2] = displacement[2] + iterator_disp_z
        '''
        print(" ")
        print("img arr shape", img_arr.shape)
        print("img arr x y z ", img_arr.shape[2], img_arr.shape[1], img_arr.shape[0])
        print("train x,y,z", train_x, train_y, train_z)
        print("calc displacement ", displacement)
        '''


        train_y = int(train_y)
        train_z = int(train_z)
        train_x = int(train_x)

        # can't be out of bounds of array 'img_arr' otherwise np.mean will sometimes return NaN
        x1 = ((train_x - displacement[0]) - 2)
        x2 = ((train_x - displacement[0]) + 3)
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - 2)
        y2 = (train_y + 3)
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - 2)
        z2 = (train_z + 3)
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_1 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_2 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_3 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_4 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_5 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_6 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_7 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_8 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_9 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = train_z - 2
        z2 = train_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_10 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_11 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_12 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_13 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_14 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_15 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = train_y - 2
        y2 = train_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_16 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_17 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = train_x - 2
        x2 = train_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_18 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_19 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_20 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_21 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x - displacement[0]) - 2
        x2 = (train_x - displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_22 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_23 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y + displacement[1]) - 2
        y2 = (train_y + displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_24 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z - displacement[2]) - 2
        z2 = (train_z - displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_25 = img_arr[z1:z2, y1:y2, x1:x2]

        x1 = (train_x + displacement[0]) - 2
        x2 = (train_x + displacement[0]) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= img_arr.shape[2]): x2 = img_arr.shape[2] - 1
        y1 = (train_y - displacement[1]) - 2
        y2 = (train_y - displacement[1]) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= img_arr.shape[1]): y2 = img_arr.shape[1] - 1
        z1 = (train_z + displacement[2]) - 2
        z2 = (train_z + displacement[2]) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= img_arr.shape[0]): z2 = img_arr.shape[0] - 1
        feature_box_26 = img_arr[z1:z2, y1:y2, x1:x2]

        '''
        print("0",feature_box_0.shape)
        print(feature_box_1.shape)
        print(feature_box_2.shape)
        print(feature_box_3.shape)
        print(feature_box_4.shape)
        print("5",feature_box_5.shape)
        print(feature_box_6.shape)
        print(feature_box_7.shape)
        print(feature_box_8.shape)
        print(feature_box_9.shape)
        print("10",feature_box_10.shape)
        print(feature_box_11.shape)
        print(feature_box_12.shape)
        print(feature_box_13.shape)
        print(feature_box_14.shape)
        print("15",feature_box_15.shape)
        print(feature_box_16.shape)
        print(feature_box_17.shape)
        print(feature_box_18.shape)
        print(feature_box_19.shape)
        print("20",feature_box_20.shape)
        print(feature_box_21.shape)
        print(feature_box_22.shape)
        print(feature_box_23.shape)
        print(feature_box_24.shape)
        print("25",feature_box_25.shape)
        print(feature_box_26.shape)
        
        
        '''


        temp_feature_vec.append(np.mean(feature_box_1))
        temp_feature_vec.append(np.mean(feature_box_2))
        temp_feature_vec.append(np.mean(feature_box_3))
        temp_feature_vec.append(np.mean(feature_box_4))
        temp_feature_vec.append(np.mean(feature_box_5))
        temp_feature_vec.append(np.mean(feature_box_6))
        temp_feature_vec.append(np.mean(feature_box_7))
        temp_feature_vec.append(np.mean(feature_box_8))
        temp_feature_vec.append(np.mean(feature_box_9))
        temp_feature_vec.append(np.mean(feature_box_10))
        temp_feature_vec.append(np.mean(feature_box_11))
        temp_feature_vec.append(np.mean(feature_box_12))
        temp_feature_vec.append(np.mean(feature_box_13))
        temp_feature_vec.append(np.mean(feature_box_14))
        temp_feature_vec.append(np.mean(feature_box_15))
        temp_feature_vec.append(np.mean(feature_box_16))
        temp_feature_vec.append(np.mean(feature_box_17))
        temp_feature_vec.append(np.mean(feature_box_18))
        temp_feature_vec.append(np.mean(feature_box_19))
        temp_feature_vec.append(np.mean(feature_box_20))
        temp_feature_vec.append(np.mean(feature_box_21))
        temp_feature_vec.append(np.mean(feature_box_22))
        temp_feature_vec.append(np.mean(feature_box_23))
        temp_feature_vec.append(np.mean(feature_box_24))
        temp_feature_vec.append(np.mean(feature_box_25))
        temp_feature_vec.append(np.mean(feature_box_26))

        counter += 1
    return temp_feature_vec
