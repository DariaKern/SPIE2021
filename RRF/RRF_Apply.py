from timeit import default_timer as timer
import os
from sklearn.externals import joblib
import SharedMethods as sm

# -------------------methods-----------------------
from RRF.mietzner_methods import nifti_loader, nifti_image_affine_reader,\
    training_subset_generator, displacement_calc, \
    dummy, loop_apply, bb_finalize, make_bounding_box

# load model
def apply_RRF(ct_scan_path, rrf_path, bb_path,  organ):
    regressor = joblib.load('{}RRF_{}.pkl'.format(rrf_path, organ))

    files = []
    for file in os.scandir(ct_scan_path):
        if file.is_file():
            print("")
            print(file)

            #TODO replace with sitk
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
            y_pred_150 = regressor.predict(final_feature_vec)
            print("Elapsed time: ", timer() - start)

            # calculate bb from predicted vector
            # loop over voxels in this window to calculate offsets and generate lists of the predicted coords for its walls
            bc_150_x_min_test, bc_150_x_max_test, bc_150_y_min_test, \
            bc_150_y_max_test, bc_150_z_min_test, bc_150_z_max_test = dummy(
                y_pred_150,
                training_xyz_min, training_xyz_max,
                spacing_x, spacing_y, spacing_z,
                offset_x, offset_y, offset_z)

            # vote for the final wall position
            new_bb_150 = bb_finalize(bc_150_x_min_test, bc_150_x_max_test, bc_150_y_min_test, bc_150_y_max_test,
                                     bc_150_z_min_test, bc_150_z_max_test)

            print("Predicted Bounding Box 150:", new_bb_150)

            new_bb_150.append(sm.get_organ_label(organ))

            make_bounding_box(new_bb_150, file, bb_path)

    print('done')