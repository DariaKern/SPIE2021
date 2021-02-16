from sklearn.ensemble import RandomForestRegressor
import SimpleITK as sitk
import joblib
import SharedMethods as sm


def train_RRF(ct_scan_path, gt_bb_path, rrf_path, organ):
    '''
        trains a organ specific Random Regression Forest

               :param ct_scan_path: path to directory containing the CT scans
               :param gt_bb_path: path to directory containing the ground truth bounding boxes
               :param rrf_path: path to target directory where the RRF models will be saved
               :param organ: organ for which a RRF will be trained.
               Valid organ names are 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'

               Usage::
                   ct_scan_path = "/path to CT scans"
                   gt_bb_path = "/path to gt bb"
                   rrf_path = "/target path"
                   organ = "liver"

                   train_RRF(ct_scan_path, gt_bb_path, rrf_path, organ)
               '''
    scan_paths_dict = sm.get_dict_of_paths(ct_scan_path)
    bb_paths_dict = sm.get_dict_of_paths(gt_bb_path, organ)
    final_feature_vec = []
    final_offset_vec = []

    for key in sorted(scan_paths_dict.keys()):
        file = scan_paths_dict[key]
        bounding_box = bb_paths_dict[key]

        print("")
        print(file)
        img = sitk.ReadImage(file)  # x, y, z
        img_arr = sitk.GetArrayFromImage(img)  # z, y, x

        bb_coordinates = sm.get_bb_coordinates(bounding_box)
        #print(bb_coordinates)

        training_xyz_min, training_xyz_max = sm.training_subset_generator(img, img_arr)
        #print("daria p_training_xyz_min, training_xyz_max : ", training_xyz_min, training_xyz_max)

        displacement = sm.displacement_calc(img, training_xyz_min)
        #print("daria displacement: ", displacement)
        sm.get_final_vectors(img, img_arr, training_xyz_min, training_xyz_max,
                             bb_coordinates, displacement, final_feature_vec, final_offset_vec)

    # use files to train a random forest regressor
    print("start training RRF for {}".format(organ))
    regressor = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10)
    regressor.fit(final_feature_vec, final_offset_vec)

    # Output a pickle file for the model
    joblib.dump(regressor, '{}RRF_{}.pkl'.format(rrf_path, organ))
    print("{} done".format(organ))
