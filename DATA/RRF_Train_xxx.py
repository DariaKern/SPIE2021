from sklearn.ensemble import RandomForestRegressor
import os
from timeit import default_timer as timer
import argparse
import textwrap
from sklearn.externals import joblib
# -----methods-----
from mietzner_methods import nifti_loader, nifti_image_affine_reader, multi_bounding_box_organizer,\
    displacement_calc, training_subset_generator
from loop_methods import loop_subset_training

#number of trees default=50
#forest depth default=10
SCAN_PATH = "/Data/Daria/RRF/CT-SCANS/"
GT_BB_PATH = "/Data/Daria/RRF/GT-BB/"
trees = 50
depth = 10

start = timer()

files = []
bb_170 = []


final_offset_vec_170 = []

final_feature_vec = []

# todo save files in dictionoary
for entry in os.scandir(SCAN_PATH):
    if entry.is_file():
        files.append(entry.path)

# todo save files to dictionary
for f_name in os.listdir(GT_BB_PATH):
    path_bb = GT_BB_PATH
    if f_name.endswith('_170_bb.vtk'):
        bb_170.append(path_bb + '\\' + f_name)


print('170: ', bb_170)
print(len(bb_170))

bb_counter = 0

for file in files:
    print(bb_170[bb_counter])

    # todo load as sitk file?
    # load Nifti format image and save image in numpy array
    img, data = nifti_loader(file)

    # todo get spacing and offset?
    spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)

    # read bounding box for every organ
    # Bounding Box 6 vector bc=(bL, bR, bA, bP, bH, bF), position of BB walls

    bc_170 = multi_bounding_box_organizer(bb_counter, bb_170)
    print('bc_170: ', bc_170)

    # the training subset is defined as the middle of the image and a box radius around it
    # the method finds this box and calculates the starting and end point for the loop
    training_xyz_min, training_xyz_max = training_subset_generator(data, spacing_x, spacing_y, spacing_z, offset_x,
                                                                   offset_y, offset_z)
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

    print("Min-Max:")
    print(training_xyz_min)
    print(training_xyz_max)

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

    # loop over training subset to generate feature vector and offset vector
    final_feature_vec, final_offset_vec_170 = loop_subset_training(
        data, training_xyz_min, training_xyz_max,
        spacing_x, spacing_y, spacing_z,
        offset_x, offset_y, offset_z,
        bc_170,
        displacement_x, displacement_y, displacement_z,
        final_feature_vec,
        final_offset_vec_170)

    bb_counter += 1

print("Elapsed time: ", (timer() - start) / 60)

# load files for training
start = timer()
# -------------------------------------------------------------------------------------------

# use files to train a random forest regressor
regressor_170 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
regressor_170.fit(final_feature_vec, final_offset_vec_170)
# Output a pickle file for the model
joblib.dump(regressor_170, './models/model_170.pkl')
print("170 done")
final_offset_vec_170 = []

print("Elapsed time: ", (timer() - start) / 60)
