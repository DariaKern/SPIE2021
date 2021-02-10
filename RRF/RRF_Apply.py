from timeit import default_timer as timer
import os
from sklearn.externals import joblib
import SharedMethods as sm
import nibabel as nib
import SimpleITK as sitk

# -------------------methods-----------------------
from RRF.mietzner_methods import nifti_image_affine_reader,\
    training_subset_generator_old, displacement_calc, \
    dummy, loop_apply, bb_finalize, make_bounding_box, training_subset_generator

# load model
def apply_RRF(ct_scan_path, rrf_path, bb_path,  organ):
    regressor = joblib.load('{}RRF_{}.pkl'.format(rrf_path, organ))

    for file in os.scandir(ct_scan_path):
        if file.is_file():
            print("")
            print(file)

            '''#TODO replace with sitk
            # load Nifti format image and save image in numpy array
            # load Nifti format
            img = nib.load(file)
            # save image in numpy array
            img_arr = img.get_fdata()
            '''
            img = nib.load(file)
            img_arr = img.get_fdata()
            spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)
            print("NIFTI")
            print(spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z)

            img1 = sitk.ReadImage(file.path) # x, y, z
            img_arr1 = sitk.GetArrayFromImage(img1) # z, y, x
            spacing = img1.GetSpacing()
            offset = img1.GetOrigin()
            print("SITK")
            print(spacing)
            print(offset)


            # get image affine from header
            training_xyz_min1, training_xyz_max1 = training_subset_generator(img_arr1, spacing, offset)

            training_xyz_min, training_xyz_max = training_subset_generator_old(img_arr,
                                                                           spacing_x, spacing_y, spacing_z,
                                                                           offset_x, offset_y, offset_z)
            exit()
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
            final_bb = bb_finalize(bb_x_min, bb_x_max, bb_y_min, bb_y_max,
                                     bb_z_min, bb_z_max)

            print("Predicted Bounding Box 150:", final_bb)

            final_bb.append(sm.get_organ_label(organ))

            make_bounding_box(final_bb, file, bb_path)

    print('done')