'''
altered mietzner code
'''

import collections
import numpy as np
import vtk
import SharedMethods as sm


# ----------------------------------------------------------------------------

# get image affine from header
# for coordinate system handling
# save spacing and offset for future calculations
# @autojit
def nifti_image_affine_reader(img):
    spacing_x = img.affine[0][0]
    spacing_y = img.affine[1][1]
    spacing_z = img.affine[2][2]
    offset_x = img.affine[0][3]
    offset_y = img.affine[1][3]
    offset_z = img.affine[2][3]
    return spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z


# -------------------------------------------------------------------------------


# calculate displacement of feature boxes
# due to different spacing, the feature boxes cant be created using voxels as measurement
# displacement has to be calculated in mm-space, to achieve the same result in different images
# a sample from the image is the starting point for the calculations
def displacement_calc_old(training_xyz_min, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    trainxyz = []
    trainxyz.append(training_xyz_min[0])
    trainxyz.append(training_xyz_min[1])
    trainxyz.append(training_xyz_min[2])

    displacement_samp_mm = vox_to_mm(trainxyz,
                                     spacing_x, spacing_y, spacing_z,
                                     offset_x, offset_y, offset_z)

    displacement_samp_mm[0] = displacement_samp_mm[0] + 25
    displacement_samp_mm[1] = displacement_samp_mm[1] + 25
    displacement_samp_mm[2] = displacement_samp_mm[2] + 25

    displacement = mm_to_vox(displacement_samp_mm,
                                     spacing_x, spacing_y, spacing_z,
                                     offset_x, offset_y, offset_z)

    displacement[0] = int(displacement[0])
    displacement[1] = int(displacement[1])
    displacement[2] = int(displacement[2])

    x = trainxyz[0] - displacement[0]
    y = trainxyz[1] - displacement[1]
    z = trainxyz[2] - displacement[2]

    displacement_x = abs(x)
    displacement_y = abs(y)
    displacement_z = abs(z)

    return displacement_x, displacement_y, displacement_z


# ----------------------------------------------------------------------------------

def feature_box_generator_old(data, training_x, training_y, training_z, displacement_x, displacement_y, displacement_z):
    #print(data.shape)
    #print(training_x, training_y, training_z)
    #print(displacement_x, displacement_y, displacement_z)

    # create feature boxes in each direction
    iterator_disp_x = displacement_x // 2
    iterator_disp_y = displacement_y // 2
    iterator_disp_z = displacement_z // 2

    # init array for feature vector of selected voxel
    temp_feature_vec = []

    # generate feature box around selected voxel and append mean to feature vector
    feature_box_0 = data[training_x - 2:training_x + 3,
                    training_y - 2:training_y + 3,
                    training_z - 2:training_z + 3]
    temp_feature_vec.append(np.mean(feature_box_0))

    # generate 26 feature boxes in a certain distance to the selected pixel
    # add half of the displacement to the displacement and generate another 26 boxes
    # this can be reapeated
    # in this case 3 iterations with 1+26+26+26=79 feature boxes
    counter = 0
    # print("TRAINING_Y!!! {}".format(training_y))
    while counter < 3:
        if counter > 0:
            displacement_x = displacement_x + iterator_disp_x
            displacement_y = displacement_y + iterator_disp_y
            displacement_z = displacement_z + iterator_disp_z

        training_y = int(training_y)
        training_z = int(training_z)
        training_x = int(training_x)

        # can't be out of bounds of array 'img_arr' otherwise np.mean will sometimes return NaN
        x1 = ((training_x - displacement_x) - 2)
        x2 = ((training_x - displacement_x) + 3)
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - 2)
        y2 = (training_y + 3)
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - 2)
        z2 = (training_z + 3)
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_1 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_2 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_3 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_4 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_5 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_6 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_7 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_8 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_9 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = training_z - 2
        z2 = training_z + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_10 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_11 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_12 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_13 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_14 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_15 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = training_y - 2
        y2 = training_y + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_16 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_17 = data[x1:x2, y1:y2, z1:z2]

        x1 = training_x - 2
        x2 = training_x + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_18 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_19 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_20 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_21 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x - displacement_x) - 2
        x2 = (training_x - displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_22 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_23 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y + displacement_y) - 2
        y2 = (training_y + displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_24 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z - displacement_z) - 2
        z2 = (training_z - displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_25 = data[x1:x2, y1:y2, z1:z2]

        x1 = (training_x + displacement_x) - 2
        x2 = (training_x + displacement_x) + 3
        if (x1 < 0): x1 = 0
        if (x2 >= data.shape[0]): x2 = data.shape[0] - 1
        y1 = (training_y - displacement_y) - 2
        y2 = (training_y - displacement_y) + 3
        if (y1 < 0): y1 = 0
        if (y2 >= data.shape[1]): y2 = data.shape[1] - 1
        z1 = (training_z + displacement_z) - 2
        z2 = (training_z + displacement_z) + 3
        if (z1 < 0): z1 = 0
        if (z2 >= data.shape[2]): z2 = data.shape[2] - 1
        feature_box_26 = data[x1:x2, y1:y2, z1:z2]

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


def get_final_vectors(data,
                      training_xyz_min, training_xyz_max,
                      spacing_x, spacing_y, spacing_z,
                      offset_x, offset_y, offset_z,
                      bb_coordinates,
                      displacement_x, displacement_y, displacement_z,
                      final_feature_vec,
                      final_offset_vec):
    # loop over voxels in this window
    counter = 0
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                # transform voxel to mm
                temp_train_coord = []
                temp_train_coord.append(training_x)
                temp_train_coord.append(training_y)
                temp_train_coord.append(training_z)

                temp_train_coord = vox_to_mm(temp_train_coord,
                                             spacing_x, spacing_y, spacing_z,
                                             offset_x, offset_y, offset_z)

                # calculate offset between bounding box and voxel(mm) in mm
                # v-bc = (vx, vx, vy, vy, vz, vz)
                bb_offset_150 = bb_offset_calc(temp_train_coord, bb_coordinates)

                # create mean feature boxes
                temp_feature_vec = feature_box_generator(data,
                                                         training_x, training_y, training_z,
                                                         displacement_x, displacement_y, displacement_z)

                '''if(np.any(np.isnan(temp_feature_vec))):
                    print("!!!!!!!!!")
                    print(temp_feature_vec)
                    print("!!!!!!!!!")
                    exit()
                    '''
                # add feature vector of current voxel to the complete feature vector
                final_feature_vec.append(temp_feature_vec)
                final_offset_vec.append(bb_offset_150)

    return final_feature_vec, final_offset_vec


def loop_subset_training(data, training_xyz_min, training_xyz_max,
                         spacing_x, spacing_y, spacing_z,
                         offset_x, offset_y, offset_z,
                         bc_150, bc_156, bc_157, bc_160, bc_170,
                         displacement_x, displacement_y, displacement_z,
                         final_feature_vec,
                         final_offset_vec_150, final_offset_vec_156, final_offset_vec_157,
                         final_offset_vec_160, final_offset_vec_170):
    # loop over voxels in this window
    counter = 0
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                # transform voxel to mm
                temp_train_coord = []
                temp_train_coord.append(training_x)
                temp_train_coord.append(training_y)
                temp_train_coord.append(training_z)

                temp_train_coord = vox_to_mm(temp_train_coord,
                                             spacing_x, spacing_y, spacing_z,
                                             offset_x, offset_y, offset_z)

                # calculate offset between bounding box and voxel(mm) in mm
                # v-bc = (vx, vx, vy, vy, vz, vz)
                bb_offset_150 = bb_offset_calc(temp_train_coord, bc_150)
                bb_offset_156 = bb_offset_calc(temp_train_coord, bc_156)
                bb_offset_157 = bb_offset_calc(temp_train_coord, bc_157)
                bb_offset_160 = bb_offset_calc(temp_train_coord, bc_160)
                bb_offset_170 = bb_offset_calc(temp_train_coord, bc_170)

                # create mean feature boxes
                temp_feature_vec = feature_box_generator(data,
                                                         training_x, training_y, training_z,
                                                         displacement_x, displacement_y, displacement_z)

                '''if(np.any(np.isnan(temp_feature_vec))):
                    print("!!!!!!!!!")
                    print(temp_feature_vec)
                    print("!!!!!!!!!")
                    exit()
                    '''
                # add feature vector of current voxel to the complete feature vector
                final_feature_vec.append(temp_feature_vec)
                final_offset_vec_150.append(bb_offset_150)
                final_offset_vec_156.append(bb_offset_156)
                final_offset_vec_157.append(bb_offset_157)
                final_offset_vec_160.append(bb_offset_160)
                final_offset_vec_170.append(bb_offset_170)
                counter = counter + 1

    return final_feature_vec, final_offset_vec_150, final_offset_vec_156, final_offset_vec_157, final_offset_vec_160, final_offset_vec_170


# -----------------------------------------------------------------------------
# calculates coordinates from voxelspace to mm-space
# "coordlist" is an array with 3 entries
def vox_to_mm(coord_list, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    x_mm = coord_list[0] * spacing_x + offset_x
    y_mm = coord_list[1] * spacing_y + offset_y
    z_mm = coord_list[2] * spacing_z + offset_z
    coord_mm = []
    coord_mm.append(x_mm)
    coord_mm.append(y_mm)
    coord_mm.append(z_mm)
    return coord_mm


# -----------------------------------------------------------------------------
# calculate coordinates from mm-space to voxelspace
def mm_to_vox(coord_list, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    x_vox = (coord_list[0] - offset_x) / spacing_x
    y_vox = (coord_list[1] - offset_y) / spacing_y
    z_vox = (coord_list[2] - offset_z) / spacing_z
    coord_vox = []
    coord_vox.append(x_vox)
    coord_vox.append(y_vox)
    coord_vox.append(z_vox)
    return coord_vox


# ------------------------------------------------------------------------------
# calculate offset between bounding box and voxel in mm
def bb_offset_calc(temp_train_coord, bc):
    bb_offset = []
    bb_offset.append(temp_train_coord[0] - bc[0])
    bb_offset.append(temp_train_coord[0] - bc[1])
    bb_offset.append(temp_train_coord[1] - bc[2])
    bb_offset.append(temp_train_coord[1] - bc[3])
    bb_offset.append(temp_train_coord[2] - bc[4])
    bb_offset.append(temp_train_coord[2] - bc[5])
    return bb_offset


def dummy(y_pred, training_xyz_min, training_xyz_max, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    # Counter iterates over the predicted offsets
    pred_Counter = 0

    # init lists for coordinates
    bb_x_min = []
    bb_x_max = []
    bb_y_min = []
    bb_y_max = []
    bb_z_min = []
    bb_z_max = []

    # loop over voxels in this window
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                # transform voxel to mm
                temp_train_coord = []
                temp_train_coord.append(training_x)
                temp_train_coord.append(training_y)
                temp_train_coord.append(training_z)

                temp_train_coord = vox_to_mm(temp_train_coord, spacing_x, spacing_y, spacing_z, offset_x, offset_y,
                                             offset_z)

                # set y_pred as offset
                bb_x_min.append(temp_train_coord[0] - y_pred[pred_Counter][0])
                bb_x_max.append(temp_train_coord[0] - y_pred[pred_Counter][1])
                bb_y_min.append(temp_train_coord[1] - y_pred[pred_Counter][2])
                bb_y_max.append(temp_train_coord[1] - y_pred[pred_Counter][3])
                bb_z_min.append(temp_train_coord[2] - y_pred[pred_Counter][4])
                bb_z_max.append(temp_train_coord[2] - y_pred[pred_Counter][5])

                pred_Counter += 1
    return bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max


# -----------------------------------------------------------------------------------------------------------------------

def loop_apply_old(data, training_xyz_min, training_xyz_max, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z,
               displacement_x, displacement_y, displacement_z, final_feature_vec):

    print(displacement_x, displacement_y, displacement_z)
    # loop over voxels in this window
    counter = 0
    for training_z in range(training_xyz_min[2], training_xyz_max[2] + 1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1] + 1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0] + 1):
                # create new variable for the current voxel
                # transform voxel to mm
                temp_train_coord = []
                temp_train_coord.append(training_x)
                temp_train_coord.append(training_y)
                temp_train_coord.append(training_z)

                temp_train_coord = vox_to_mm(temp_train_coord, spacing_x, spacing_y, spacing_z, offset_x, offset_y,
                                             offset_z)
                # print("TRAINING_Y!!! {}, {}, {}".format(training_y, training_z, training_x))
                # create mean feature boxes
                temp_feature_vec = feature_box_generator_old(data, training_x, training_y, training_z, displacement_x,
                                                         displacement_y, displacement_z)
                # print(temp_feature_vec)

                # add feature vector of current voxel to the complete feature vector
                final_feature_vec.append(temp_feature_vec)
                counter = counter + 1
    return final_feature_vec


# -------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------



def training_subset_generator_old(data, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    rows = data.shape[0]
    columns = data.shape[1]
    z_axis = data.shape[2]
    axial_center = [columns / 2, rows / 2, z_axis / 2]
    #print('axial center: ', axial_center)

    # calculate voxel to mm coordinates
    axial_center_mm = vox_to_mm(axial_center, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z)
    #print('axial center in mm: ', axial_center_mm)

    # calculate voxel training subset, +-100mm
    training_xyz_min = []
    training_xyz_max = []

    training_xyz_min.append(axial_center_mm[0] - 15)
    training_xyz_min.append(axial_center_mm[1] - 15)
    training_xyz_min.append(axial_center_mm[2] - 15)

    training_xyz_max.append(axial_center_mm[0] + 15)
    training_xyz_max.append(axial_center_mm[1] + 15)
    training_xyz_max.append(axial_center_mm[2] + 15)
    #print('training xyz max min mm: ', training_xyz_max, training_xyz_min)

    # calculate coordinates of voxel training subset
    training_xyz_min = mm_to_vox(training_xyz_min,
                                 spacing_x, spacing_y, spacing_z,
                                 offset_x, offset_y, offset_z)

    training_xyz_max = mm_to_vox(training_xyz_max,
                                 spacing_x, spacing_y, spacing_z,
                                 offset_x, offset_y, offset_z)

    # round coordinates
    training_xyz_min[0] = int(training_xyz_min[0])
    training_xyz_min[1] = int(training_xyz_min[1])
    training_xyz_min[2] = int(training_xyz_min[2])
    # print('Trainingxyzmin (rounded): ', training_xyz_min)

    training_xyz_max[0] = int(training_xyz_max[0])
    training_xyz_max[1] = int(training_xyz_max[1])
    training_xyz_max[2] = int(training_xyz_max[2])
    # print('Trainingxyzmax (rounded): ', training_xyz_max)
    #print('training xyz max min: ', training_xyz_max, training_xyz_min)

    return training_xyz_min, training_xyz_max

