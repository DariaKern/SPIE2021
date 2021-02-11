from timeit import default_timer as timer
import os
#from sklearn.externals import joblib
import joblib
import SharedMethods as sm
import nibabel as nib
import SimpleITK as sitk

# -------------------methods-----------------------
from RRF.mietzner_methods import displacement_calc_old, \
    dummy, loop_apply_old, training_subset_generator_old, nifti_image_affine_reader


# load model
def apply_RRF(ct_scan_path, rrf_path, bb_path, organ):
    regressor = joblib.load('{}RRF_{}.pkl'.format(rrf_path, organ))

    for file in os.scandir(ct_scan_path):
        if file.is_file():
            print("")
            print(file)
            img = sitk.ReadImage(file.path)  # x, y, z
            img_arr = sitk.GetArrayFromImage(img)  # z, y, x

            training_xyz_min, training_xyz_max = sm.training_subset_generator(img, img_arr)
            displacement = sm.displacement_calc(img, training_xyz_min)

            # loop over testing data to generate feature vector and offset vector
            final_feature_vec = sm.loop_apply(img_arr, training_xyz_min, training_xyz_max, displacement)

            # predict bb's for image
            start = timer()
            predictions = regressor.predict(final_feature_vec)
            print("Elapsed time: ", timer() - start)

            # calculate bb from predicted vector
            bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max = sm.calc_bb_coordinates(img, predictions, training_xyz_min, training_xyz_max)
            # vote for the final wall position
            final_bb = sm.bb_finalize(bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max)
            final_bb.append(sm.get_organ_label(organ))
            print("Predicted Bounding Box :", final_bb)
            sm.make_bounding_box(final_bb, file, bb_path)

def apply_RRF_old(ct_scan_path, rrf_path, bb_path, organ):
    regressor = joblib.load('{}RRF_{}.pkl'.format(rrf_path, organ))

    for file in os.scandir(ct_scan_path):
        if file.is_file():
            print("")
            print(file)

            # load Nifti format image and save image in numpy array
            # load Nifti format
            img = nib.load(file)
            # save image in numpy array
            img_arr = img.get_fdata()

            img = nib.load(file)
            img_arr = img.get_fdata()
            spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)
            print("NIFTI")
            print(img_arr.shape)
            training_xyz_min, training_xyz_max = training_subset_generator_old(img_arr,
                                                                               spacing_x, spacing_y, spacing_z,
                                                                               offset_x, offset_y, offset_z)

            if spacing_x < 0:
                temp_spac_x = training_xyz_min[0]
                training_xyz_min[0] = training_xyz_max[0]
                training_xyz_max[0] = temp_spac_x

            if spacing_y < 0:
                temp_spac_y = training_xyz_min[1]
                training_xyz_min[1] = training_xyz_max[1]
                training_xyz_max[1] = temp_spac_y

            print("Mietzner max min", training_xyz_max, training_xyz_min)
            displacement_x, displacement_y, displacement_z = displacement_calc_old(training_xyz_min,
                                                                                   spacing_x, spacing_y, spacing_z,
                                                                                   offset_x, offset_y, offset_z)
            displacement_x = int(displacement_x)
            displacement_y = int(displacement_y)
            displacement_z = int(displacement_z)
            print("Mietzner dsiplacement", displacement_x, displacement_y, displacement_z)

            # loop over testing data to generate feature vector and offset vector
            final_feature_vec = []
            final_feature_vec = loop_apply_old(
                img_arr, training_xyz_min, training_xyz_max,
                spacing_x, spacing_y, spacing_z,
                offset_x, offset_y, offset_z,
                displacement_x, displacement_y, displacement_z,
                final_feature_vec)

            # predict bb's for image
            start = timer()
            predicted_bb = regressor.predict(final_feature_vec)
            print("Elapsed time: ", timer() - start)

            # calculate bb from predicted vector
            # loop over voxels in this window to calculate offsets and generate lists of the predicted coords for its walls
            bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max = dummy(
                predicted_bb,
                training_xyz_min, training_xyz_max,
                spacing_x, spacing_y, spacing_z,
                offset_x, offset_y, offset_z)

            # vote for the final wall position
            final_bb = sm.bb_finalize(bb_x_min, bb_x_max, bb_y_min, bb_y_max,
                                   bb_z_min, bb_z_max)

            print("Predicted Bounding Box 150:", final_bb)

            final_bb.append(sm.get_organ_label(organ))

            sm.make_bounding_box(final_bb, file, bb_path)

    print('done')
