from timeit import default_timer as timer
#from numba import autojit
import collections
import nibabel as nib
import numpy as np
import os
import shutil
import vtk
#--------------------------------------------------------------------------

# loads a Nifti image using Nibabel
# saves the image in two different variables
# "data" is a numpy array of the image data
# "img" is used to get the image affine
#@autojit
def nifti_loader(file):
    # load Nifti format
    img = nib.load(file)

    # save image in numpy array
    data = img.get_fdata()

    return img, data
#----------------------------------------------------------------------------

# get image affine from header
# for coordinate system handling
# save spacing and offset for future calculations
#@autojit
def nifti_image_affine_reader(img):
    spacing_x = img.affine[0][0]
    spacing_y = img.affine[1][1]
    spacing_z = img.affine[2][2]
    offset_x = img.affine[0][3]
    offset_y = img.affine[1][3]
    offset_z = img.affine[2][3]
    return spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z

#-------------------------------------------------------------------------------

# read bounding box coordinates
# open vtk file and get coordinates
# returns coordinates
#@autojit
def bounding_box_reader(bb, bb_counter):
    #TODO DARIA read VTK file

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(bb[bb_counter])
    reader.Update()
    box = reader.GetOutput()
    x_min, x_max, y_min, y_max, z_min, z_max = box.GetBounds()
    return x_min, y_min, z_min, x_max, y_max, z_max


#------------------------------------------------------------------------------
# finds the bounding boxes for each organ
# writes bb's in one array per box
# returns 3 arrays that each describe one box as a 6-vector
#@autojit
def multi_bounding_box_organizer(bb_counter, bb_150, bb_156, bb_157, bb_160, bb_170):
    bb_150_xmin, bb_150_ymin, bb_150_zmin,\
    bb_150_xmax, bb_150_ymax, bb_150_zmax = bounding_box_reader(bb_150, bb_counter)
    bc_150 = []
    bc_150.append(bb_150_xmin)
    bc_150.append(bb_150_xmax)
    bc_150.append(bb_150_ymin)
    bc_150.append(bb_150_ymax)
    bc_150.append(bb_150_zmin)
    bc_150.append(bb_150_zmax)

    bb_156_xmin, bb_156_ymin, bb_156_zmin, \
    bb_156_xmax, bb_156_ymax, bb_156_zmax = bounding_box_reader(bb_156, bb_counter)
    bc_156 = []
    bc_156.append(bb_156_xmin)
    bc_156.append(bb_156_xmax)
    bc_156.append(bb_156_ymin)
    bc_156.append(bb_156_ymax)
    bc_156.append(bb_156_zmin)
    bc_156.append(bb_156_zmax)

    bb_157_xmin, bb_157_ymin, bb_157_zmin, \
    bb_157_xmax, bb_157_ymax, bb_157_zmax = bounding_box_reader(bb_157, bb_counter)
    bc_157 = []
    bc_157.append(bb_157_xmin)
    bc_157.append(bb_157_xmax)
    bc_157.append(bb_157_ymin)
    bc_157.append(bb_157_ymax)
    bc_157.append(bb_157_zmin)
    bc_157.append(bb_157_zmax)

    bb_160_xmin, bb_160_ymin, bb_160_zmin, \
    bb_160_xmax, bb_160_ymax, bb_160_zmax = bounding_box_reader(bb_160, bb_counter)
    bc_160 = []
    bc_160.append(bb_160_xmin)
    bc_160.append(bb_160_xmax)
    bc_160.append(bb_160_ymin)
    bc_160.append(bb_160_ymax)
    bc_160.append(bb_160_zmin)
    bc_160.append(bb_160_zmax)

    bb_170_xmin, bb_170_ymin, bb_170_zmin, \
    bb_170_xmax, bb_170_ymax, bb_170_zmax = bounding_box_reader(bb_170, bb_counter)
    bc_170 = []
    bc_170.append(bb_170_xmin)
    bc_170.append(bb_170_xmax)
    bc_170.append(bb_170_ymin)
    bc_170.append(bb_170_ymax)
    bc_170.append(bb_170_zmin)
    bc_170.append(bb_170_zmax)

    return bc_150, bc_156, bc_157, bc_160, bc_170


# calculate displacement of feature boxes
# due to different spacing, the feature boxes cant be created using voxels as measurement
# displacement has to be calculated in mm-space, to achieve the same result in different images
# a sample from the image is the starting point for the calculations
#@autojit
def displacement_calc(training_xyz_min, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    displacement_samp = []
    displacement_samp.append(training_xyz_min[0])
    displacement_samp.append(training_xyz_min[1])
    displacement_samp.append(training_xyz_min[2])

    displacement_samp_mm = vox_to_mm(displacement_samp,
                                     spacing_x, spacing_y, spacing_z,
                                     offset_x, offset_y, offset_z)

    displacement_samp_mm[0] = displacement_samp_mm[0] + 25
    displacement_samp_mm[1] = displacement_samp_mm[1] + 25
    displacement_samp_mm[2] = displacement_samp_mm[2] + 25

    displacement_samp_mm = mm_to_vox(displacement_samp_mm,
                                     spacing_x, spacing_y, spacing_z,
                                     offset_x, offset_y, offset_z)

    displacement_samp_mm[0] = int(displacement_samp_mm[0])
    displacement_samp_mm[1] = int(displacement_samp_mm[1])
    displacement_samp_mm[2] = int(displacement_samp_mm[2])

    displacement_x = displacement_samp[0] - displacement_samp_mm[0]
    displacement_y = displacement_samp[1] - displacement_samp_mm[1]
    displacement_z = displacement_samp[2] - displacement_samp_mm[2]

    displacement_x = abs(displacement_x)
    displacement_y = abs(displacement_y)
    displacement_z = abs(displacement_z)

    return displacement_x, displacement_y, displacement_z
#----------------------------------------------------------------------------------

#@autojit
def feature_box_generator(data, training_x, training_y, training_z, displacement_x, displacement_y, displacement_z):
    # create feature boxes in each direction
    iterator_disp_x = displacement_x//2
    iterator_disp_y = displacement_y//2
    iterator_disp_z = displacement_z//2

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
    #print("TRAINING_Y!!! {}".format(training_y))
    while counter < 3:
        if counter > 0:
            displacement_x = displacement_x + iterator_disp_x
            displacement_y = displacement_y + iterator_disp_y
            displacement_z = displacement_z + iterator_disp_z
        #TODO
        training_y = int(training_y)
        training_z = int(training_z)
        training_x = int(training_x)

        feature_box_1 = data[
                        ((training_x - displacement_x) - 2):((training_x - displacement_x) + 3),
                        (training_y - 2):(training_y + 3),
                        (training_z - 2):(training_z + 3)
                        ]
        feature_box_2 = data[
                        training_x - 2:training_x + 3,
                        (training_y + displacement_y) - 2:(training_y + displacement_y) + 3,
                        training_z - 2:training_z + 3
                        ]
        feature_box_3 = data[
                        (training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                        training_y - 2:training_y + 3,
                        training_z - 2:training_z + 3
                        ]
        feature_box_4 = data[
                        training_x - 2:training_x + 3,
                        (training_y - displacement_y) - 2:(training_y - displacement_y) + 3,
                        training_z - 2:training_z + 3
                        ]
        feature_box_5 = data[
                        training_x - 2:training_x + 3,
                        training_y - 2:training_y + 3,
                        (training_z - displacement_z) - 2:(training_z - displacement_z) + 3
                        ]
        feature_box_6 = data[
                        training_x - 2:training_x + 3,
                        training_y - 2:training_y + 3,
                        (training_z + displacement_z) - 2:(training_z + displacement_z) + 3
                        ]
        feature_box_7 = data[
                        (training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                        (training_y - displacement_y) - 2:(training_y - displacement_y) + 3,
                        training_z - 2:training_z + 3
                        ]
        feature_box_8 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, training_z - 2:training_z + 3]
        feature_box_9 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, training_z - 2:training_z + 3]
        feature_box_10 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                     (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, training_z - 2:training_z + 3]
        feature_box_11 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    training_y - 2:training_y + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_12 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    training_y - 2:training_y + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_13 = data[training_x - 2:training_x + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z - displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_14 = data[training_x - 2:training_x + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_15 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                    training_y - 2:training_y + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_16 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                    training_y - 2:training_y + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_17 = data[training_x - 2:training_x + 3,
                    (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_18 = data[training_x - 2:training_x + 3,
                    (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_19 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_20 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_21 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_22 = data[(training_x - displacement_x) - 2:(training_x - displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_23 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_24 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                    (training_y + displacement_y) - 2:(training_y + displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]
        feature_box_25 = data[
                         (training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                         (training_y - displacement_y) - 2:(training_y - displacement_y) + 3,
                         (training_z - displacement_z) - 2:(training_z - displacement_z) + 3]
        feature_box_26 = data[(training_x + displacement_x) - 2:(training_x + displacement_x) + 3,
                     (training_y - displacement_y) - 2:(training_y - displacement_y) + 3, (training_z + displacement_z) - 2:(training_z + displacement_z) + 3]


        # calculate mean of feature boxes
        if(counter == 2):
            print((training_x + displacement_x) - 2)
            print((training_x + displacement_x) + 3)
            #print(data[271:276])
            print((training_y - displacement_y) - 2)
            print((training_y - displacement_y) + 3)
            #print(data[221:226])
            print((training_z - displacement_z) - 2)
            print((training_z - displacement_z) + 3)
            print(data[0:3])


            print(feature_box_25)
            print(np.mean(feature_box_25))
            print("")

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

#-------------------------------------------------------------------------------------------
# the training subset is located in the middle of the image
# the axial center of the image has to be found
# an area around this center has to be allocated
# returns min, max for x, y
#@autojit()
def training_subset_generator(data, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    rows = data.shape[0]
    columns = data.shape[1]
    z_axis = data.shape[2]
    axial_center = [columns / 2, rows / 2, z_axis / 2]
    print('axial center: ', axial_center)

    # calculate voxel to mm coordinates

    axial_center_mm = vox_to_mm(axial_center, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z)
    print('axial center in mm: ', axial_center_mm)

    # calculate voxel training subset, +-100mm
    training_xyz_min = []
    training_xyz_max = []

    training_xyz_min.append(axial_center_mm[0] - 15)
    training_xyz_min.append(axial_center_mm[1] - 15)
    training_xyz_min.append(axial_center_mm[2] - 15)

    training_xyz_max.append(axial_center_mm[0] + 15)
    training_xyz_max.append(axial_center_mm[1] + 15)
    training_xyz_max.append(axial_center_mm[2] + 15)

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
    print('Trainingxyzmin (rounded): ', training_xyz_min)

    training_xyz_max[0] = int(training_xyz_max[0])
    training_xyz_max[1] = int(training_xyz_max[1])
    training_xyz_max[2] = int(training_xyz_max[2])
    print('Trainingxyzmax (rounded): ', training_xyz_max)
    return training_xyz_min, training_xyz_max


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
    for training_z in range(training_xyz_min[2], training_xyz_max[2]+1):
        for training_y in range(training_xyz_min[1], training_xyz_max[1]+1):
            for training_x in range(training_xyz_min[0], training_xyz_max[0]+1):

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

                # add feature vector of current voxel to the complete feature vector
                if(np.any(np.isnan(temp_feature_vec))):
                    #TODO Daria
                    print("!!!!!!!!!")
                    print(temp_feature_vec)
                    print("!!!!!!!!!")
                    exit()
                final_feature_vec.append(temp_feature_vec)
                final_offset_vec_150.append(bb_offset_150)
                final_offset_vec_156.append(bb_offset_156)
                final_offset_vec_157.append(bb_offset_157)
                final_offset_vec_160.append(bb_offset_160)
                final_offset_vec_170.append(bb_offset_170)
                counter = counter+1

    return final_feature_vec, final_offset_vec_150, final_offset_vec_156, final_offset_vec_157, final_offset_vec_160, final_offset_vec_170



#-----------------------------------------------------------------------------
#calculates coordinates from voxelspace to mm-space
#"coordlist" is an array with 3 entries
#@autojit
def vox_to_mm(coord_list, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    x_mm = coord_list[0] * spacing_x + offset_x
    y_mm = coord_list[1] * spacing_y + offset_y
    z_mm = coord_list[2] * spacing_z + offset_z
    coord_mm = []
    coord_mm.append(x_mm)
    coord_mm.append(y_mm)
    coord_mm.append(z_mm)
    return coord_mm
#-----------------------------------------------------------------------------
#calculate coordinates from mm-space to voxelspace
#@autojit
def mm_to_vox(coord_list, spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z):
    x_vox = (coord_list[0] - offset_x)/spacing_x
    y_vox = (coord_list[1] - offset_y)/spacing_y
    z_vox = (coord_list[2] - offset_z) / spacing_z
    coord_vox = []
    coord_vox.append(x_vox)
    coord_vox.append(y_vox)
    coord_vox.append(z_vox)
    return coord_vox

#------------------------------------------------------------------------------
# calculate offset between bounding box and voxel in mm
#@autojit()
def bb_offset_calc(temp_train_coord, bc):
    bb_offset = []
    bb_offset.append(temp_train_coord[0] - bc[0])
    bb_offset.append(temp_train_coord[0] - bc[1])
    bb_offset.append(temp_train_coord[1] - bc[2])
    bb_offset.append(temp_train_coord[1] - bc[3])
    bb_offset.append(temp_train_coord[2] - bc[4])
    bb_offset.append(temp_train_coord[2] - bc[5])
    return bb_offset

'''

    boxfile = open(bb[bb_counter], 'r')
    lines = boxfile.readlines()
    numbers1 = lines[5].split()
    zmin = numbers1[2]
    zmin = float(zmin)
    y2 = numbers1[1]
    y2 = float(y2)
    x = numbers1[0]
    x = float(x)
    numbers2 = lines[11].split()
    zmax = numbers2[2]
    zmax = float(zmax)
    y = numbers2[1]
    y = float(y)
    x2 = numbers2[0]
    x2 = float(x2)
    boxfile.close()

'''