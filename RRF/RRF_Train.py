from sklearn.ensemble import RandomForestRegressor
import os
from timeit import default_timer as timer
import argparse
import textwrap
import numpy as np
from sklearn.externals import joblib
# -----methods-----
from mietzner_methods import nifti_loader, nifti_image_affine_reader, multi_bounding_box_organizer,\
    displacement_calc, training_subset_generator, loop_subset_training
from SharedMethods import get_dict_of_paths

# funktioniert mit Mietzner Daten
#GT_BB_PATH = "/Data/Daria/RRF/GT-BB/"
GT_BB_PATH = "/Data/Daria/RRF/BB-TEST/"
#GT_BB_PATH = "/Data/Daria/RRF/_GT-MIETZNER/"
#SCAN_PATH = "/Data/Daria/RRF/CT-SCANS/"
SCAN_PATH= "/Data/Daria/RRF/SCANS-TEST/"
#SCAN_PATH = "/Data/Daria/RRF/_CT-SCANS-MIETZNER/"
SAVE_PATH = "/Data/Daria/RRF/"
trees = 50
depth = 10

start = timer()
# feature vector for one data-point: v(p) = (v1,..., vd)
# voxel: p = (px, py, pz)

files = []
bb_150 = []
bb_156 = []
bb_157 = []
bb_160 = []
bb_170 = []

final_offset_vec_150 = []
final_offset_vec_156 = []
final_offset_vec_157 = []
final_offset_vec_160 = []
final_offset_vec_170 = []

final_feature_vec = []

scan_paths_dict = get_dict_of_paths(SCAN_PATH)
bb_paths_dict_170 = get_dict_of_paths(GT_BB_PATH, 'liver')
bb_paths_dict_156 = get_dict_of_paths(GT_BB_PATH, 'left_kidney')
bb_paths_dict_157 = get_dict_of_paths(GT_BB_PATH, 'right_kidney')
bb_paths_dict_160 = get_dict_of_paths(GT_BB_PATH, 'spleen')
bb_paths_dict_150 = get_dict_of_paths(GT_BB_PATH, 'pancreas')

for key in sorted(scan_paths_dict.keys()):
    files.append(scan_paths_dict[key])
    bb_170.append(bb_paths_dict_170[key])
    bb_156.append(bb_paths_dict_156[key])
    bb_157.append(bb_paths_dict_157[key])
    bb_160.append(bb_paths_dict_160[key])
    bb_150.append(bb_paths_dict_150[key])

bb_counter = 0

for file in files:
    print(file)
    # load Nifti format image and save image in numpy array
    img, data = nifti_loader(file)

    # get image affine from header
    spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)

    # read bounding box for every organ
    # Bounding Box 6 vector bc=(bL, bR, bA, bP, bH, bF), position of BB walls

    bc_150, bc_156, bc_157, \
    bc_160, bc_170 = multi_bounding_box_organizer(bb_counter,
                                                  bb_150, bb_156,
                                                  bb_157, bb_160, bb_170)

    # the training subset is defined as the middle of the image and a box radius around it
    # the method finds this box and calculates the starting and end point for the loop
    training_xyz_min, training_xyz_max = training_subset_generator(data,
                                                                   spacing_x, spacing_y, spacing_z,
                                                                   offset_x, offset_y, offset_z)
    training_xyz_min[0] = int(training_xyz_min[0])
    training_xyz_min[1] = int(training_xyz_min[1])
    training_xyz_min[2] = int(training_xyz_min[2])
    training_xyz_max[0] = int(training_xyz_max[0])
    training_xyz_max[1] = int(training_xyz_max[1])
    training_xyz_max[2] = int(training_xyz_max[2])

    # a negative spacing would switch the max and min coordinates (xmin->xmax, xmax->xmin)
    # if spacing is negative min and max will be switched to match the correct values
    if spacing_x < 0:
        temp_spac_x = training_xyz_min[0]
        training_xyz_min[0] = training_xyz_max[0]
        training_xyz_max[0] = temp_spac_x

    if spacing_y < 0:
        temp_spac_y = training_xyz_min[1]
        training_xyz_min[1] = training_xyz_max[1]
        training_xyz_max[1] = temp_spac_y


    # init array for the image and its feature vectors
    image_feature_vec = []

    # init array for image offset vector
    image_offset_vec = []

    # calculate displacement of feature boxes
    # due to different spacing, the feature boxes cant be created using voxels as measurement
    # displacement has to be calculated in mm-space, to achieve the same result in different images
    # a sample from the image is the starting point for the calculations

    displacement_x, displacement_y, displacement_z = displacement_calc(training_xyz_min,
                                                                       spacing_x, spacing_y, spacing_z,
                                                                       offset_x, offset_y, offset_z)
    displacement_x = int(displacement_x)
    displacement_y = int(displacement_y)
    displacement_z = int(displacement_z)
    print('Displacement of feature boxes in voxels (x y z): ', displacement_x, displacement_y, displacement_z)
    print("")
    # loop over training subset to generate feature vector and offset vector
    final_feature_vec, \
    final_offset_vec_150, final_offset_vec_156, \
    final_offset_vec_157, final_offset_vec_160, \
    final_offset_vec_170 = loop_subset_training(
        data, training_xyz_min, training_xyz_max,
        spacing_x, spacing_y, spacing_z,
        offset_x, offset_y, offset_z,
        bc_150, bc_156, bc_157, bc_160, bc_170,
        displacement_x, displacement_y, displacement_z, final_feature_vec,
        final_offset_vec_150, final_offset_vec_156, final_offset_vec_157,
        final_offset_vec_160, final_offset_vec_170)

    bb_counter += 1

print("Elapsed time: ", (timer() - start) / 60)

# load files for training
start = timer()
#exit()
# Dauer pro RandomForestRegressor ca 3h
# -------------------------------------------------------------------------------------------
# use files to train a random forest regressor
print("start training RRF for 150")
regressor_150 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
print(final_feature_vec)
print(np.any(np.isnan(final_feature_vec)))


regressor_150.fit(final_feature_vec, final_offset_vec_150)
# Output a pickle file for the model
joblib.dump(regressor_150, '{}model_150.pkl'.format(SAVE_PATH))
print("150 done")
final_offset_vec_150 = []

# use files to train a random forest regressor
print("start training RRF for 156")
regressor_156 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
regressor_156.fit(final_feature_vec, final_offset_vec_156)
# Output a pickle file for the model
joblib.dump(regressor_156, '{}model_156.pkl'.format(SAVE_PATH))
print("156 done")
final_offset_vec_156 = []

# use files to train a random forest regressor
regressor_157 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
regressor_157.fit(final_feature_vec, final_offset_vec_157)
# Output a pickle file for the model
joblib.dump(regressor_157, '{}model_157.pkl'.format(SAVE_PATH))
print("157 done")
final_offset_vec_157 = []

# use files to train a random forest regressor
print("start training RRF for 160")
regressor_160 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
regressor_160.fit(final_feature_vec, final_offset_vec_160)
# Output a pickle file for the model
joblib.dump(regressor_160, '{}model_160.pkl'.format(SAVE_PATH))
print("160 done")
final_offset_vec_160 = []

# use files to train a random forest regressor
print("start training RRF for 170")
regressor_170 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
regressor_170.fit(final_feature_vec, final_offset_vec_170)
# Output a pickle file for the model
joblib.dump(regressor_170, '{}model_170.pkl'.format(SAVE_PATH))
print("170 done")
final_offset_vec_170 = []

print("Elapsed time: ", (timer() - start) / 60)



'''
for entry in os.scandir(SCAN_PATH):
    if entry.is_file():
        files.append(entry.path)


for f_name in os.listdir(GT_BB_PATH):
    path_bb = GT_BB_PATH
    if f_name.endswith('_157_bb.vtk'):
        bb_157.append(path_bb + f_name)
    elif f_name.endswith('_170_bb.vtk'):
        bb_170.append(path_bb + f_name)
    elif f_name.endswith('_156_bb.vtk'):
        bb_156.append(path_bb + f_name)
    elif f_name.endswith('_150_bb.vtk'):
        bb_150.append(path_bb + f_name)
    elif f_name.endswith('_160_bb.vtk'):
        bb_160.append(path_bb + f_name)

BEFORE:
order or files in list didn't match
input:
print(bb_170)
print(files)
output:
['/Data/Daria/RRF/_GT-MIETZNER/seg0.nii_170_bb.vtk', '/Data/Daria/RRF/_GT-MIETZNER/seg1.nii_170_bb.vtk', ...
['/Data/Daria/RRF/_CT-SCANS-MIETZNER/26.nii.gz', '/Data/Daria/RRF/_CT-SCANS-MIETZNER/0.nii.gz', ...
'''

