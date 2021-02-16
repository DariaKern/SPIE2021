from timeit import default_timer as timer
import os
import joblib
import SharedMethods as sm
import SimpleITK as sitk


def apply_RRF(ct_scan_path, rrf_path, bb_path, organ):
    regressor = joblib.load('{}RRF_{}.pkl'.format(rrf_path, organ))

    for file in os.scandir(ct_scan_path):
        if file.is_file():
            print("")
            print(file.name)
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
            bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max = sm.calc_bb_coordinates(img, predictions,
                                                                                                training_xyz_min,
                                                                                                training_xyz_max)
            # vote for the final wall position
            final_bb = sm.bb_finalize(bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max)
            final_bb.append(sm.get_organ_label(organ))
            print("Predicted Bounding Box :", final_bb)
            sm.make_bounding_box(final_bb, file, bb_path)
