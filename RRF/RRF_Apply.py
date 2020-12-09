from timeit import default_timer as timer
import os
from sklearn.externals import joblib

# -------------------methods-----------------------
from RRF.mietzner_methods import nifti_loader, nifti_image_affine_reader,\
    training_subset_generator, displacement_calc, \
    loop_offset_to_bb, loop_apply, bb_finalize, make_bounding_box

image_path = "/Data/Daria/RRF/CT-SCANS/"
SAVE_PATH = "/Data/Daria/RRF/"
BB_PATH = "/Data/Daria/RRF/BB/"

# load model

regressor_150 = joblib.load('{}model_150.pkl'.format(SAVE_PATH))
regressor_156 = joblib.load('{}model_156.pkl'.format(SAVE_PATH))
regressor_157 = joblib.load('{}model_157.pkl'.format(SAVE_PATH))
regressor_160 = joblib.load('{}model_160.pkl'.format(SAVE_PATH))
regressor_170 = joblib.load('{}model_170.pkl'.format(SAVE_PATH))

files = []

for entry in os.scandir(image_path):
    if entry.is_file():
        files.append(entry.path)

print(files)

for file in files:
    print(file)
    # load Nifti format image and save image in numpy array
    img, data = nifti_loader(file)

    # get image affine from header
    spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)

    training_xyz_min, training_xyz_max = training_subset_generator(data,
                                                                   spacing_x, spacing_y, spacing_z,
                                                                   offset_x, offset_y, offset_z)
    training_xyz_min[0] = int(training_xyz_min[0])
    training_xyz_min[1] = int(training_xyz_min[1])
    training_xyz_min[2] = int(training_xyz_min[2])
    training_xyz_max[0] = int(training_xyz_max[0])
    training_xyz_max[1] = int(training_xyz_max[1])
    training_xyz_max[2] = int(training_xyz_max[2])

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


    # init array for all voxels and their feature vectors
    voxel_feature_vec = []

    displacement_x, displacement_y, displacement_z = displacement_calc(training_xyz_min,
                                                                       spacing_x, spacing_y, spacing_z,
                                                                       offset_x, offset_y, offset_z)
    displacement_x = int(displacement_x)
    displacement_y = int(displacement_y)
    displacement_z = int(displacement_z)
    print('Displacement of feature boxes in voxels (x y z): ', displacement_x, displacement_y, displacement_z)
    # loop over testing data to generate feature vector and offset vector
    final_feature_vec = []

    final_feature_vec = loop_apply(
        data, training_xyz_min, training_xyz_max,
        spacing_x, spacing_y, spacing_z,
        offset_x, offset_y, offset_z,
        displacement_x, displacement_y, displacement_z,
        final_feature_vec)

    # predict bb's for image
    start = timer()
    y_pred_150 = regressor_150.predict(final_feature_vec)
    y_pred_156 = regressor_156.predict(final_feature_vec)
    y_pred_157 = regressor_157.predict(final_feature_vec)
    y_pred_160 = regressor_160.predict(final_feature_vec)
    y_pred_170 = regressor_170.predict(final_feature_vec)
    print("Elapsed time: ", timer() - start)

    # calculate bb from predicted vector
    # loop over voxels in this window to calculate offsets and generate lists of the predicted coords for its walls
    bc_150_x_min_test, bc_150_x_max_test, bc_150_y_min_test, \
    bc_150_y_max_test, bc_150_z_min_test, bc_150_z_max_test, \
    bc_156_x_min_test, bc_156_x_max_test, bc_156_y_min_test, \
    bc_156_y_max_test, bc_156_z_min_test, bc_156_z_max_test, \
    bc_157_x_min_test, bc_157_x_max_test, bc_157_y_min_test, \
    bc_157_y_max_test, bc_157_z_min_test, bc_157_z_max_test, \
    bc_160_x_min_test, bc_160_x_max_test, bc_160_y_min_test, \
    bc_160_y_max_test, bc_160_z_min_test, bc_160_z_max_test, \
    bc_170_x_min_test, bc_170_x_max_test, bc_170_y_min_test, \
    bc_170_y_max_test, bc_170_z_min_test, bc_170_z_max_test = loop_offset_to_bb(
        y_pred_150, y_pred_156, y_pred_157, y_pred_160, y_pred_170,
        training_xyz_min, training_xyz_max,
        spacing_x, spacing_y, spacing_z,
        offset_x, offset_y, offset_z)

    # vote for the final wall position
    new_bb_150 = bb_finalize(bc_150_x_min_test, bc_150_x_max_test, bc_150_y_min_test, bc_150_y_max_test,
                             bc_150_z_min_test, bc_150_z_max_test)

    new_bb_156 = bb_finalize(bc_156_x_min_test, bc_156_x_max_test, bc_156_y_min_test, bc_156_y_max_test,
                             bc_156_z_min_test, bc_156_z_max_test)

    new_bb_157 = bb_finalize(bc_157_x_min_test, bc_157_x_max_test, bc_157_y_min_test, bc_157_y_max_test,
                             bc_157_z_min_test, bc_157_z_max_test)

    new_bb_160 = bb_finalize(bc_160_x_min_test, bc_160_x_max_test, bc_160_y_min_test, bc_160_y_max_test,
                             bc_160_z_min_test, bc_160_z_max_test)

    new_bb_170 = bb_finalize(bc_170_x_min_test, bc_170_x_max_test, bc_170_y_min_test, bc_170_y_max_test,
                             bc_170_z_min_test, bc_170_z_max_test)

    print("Predicted Bounding Box 150:", new_bb_150)
    print("Predicted Bounding Box 156:", new_bb_156)
    print("Predicted Bounding Box 157:", new_bb_157)
    print("Predicted Bounding Box 160:", new_bb_160)
    print("Predicted Bounding Box 170:", new_bb_170)


    new_bb_150.append('150')
    new_bb_156.append('156')
    new_bb_157.append('157')
    new_bb_160.append('160')
    new_bb_170.append('170')

    make_bounding_box(new_bb_150, file, BB_PATH)
    make_bounding_box(new_bb_156, file, BB_PATH)
    make_bounding_box(new_bb_157, file, BB_PATH)
    make_bounding_box(new_bb_160, file, BB_PATH)
    make_bounding_box(new_bb_170, file, BB_PATH)

print('done')
