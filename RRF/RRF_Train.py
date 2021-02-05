from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.externals import joblib
from RRF.mietzner_methods import nifti_loader, nifti_image_affine_reader,\
    displacement_calc, training_subset_generator,\
    bounding_box_reader, get_final_vectors
from SharedMethods import get_dict_of_paths


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
    start = timer()
    # feature vector for one data-point: v(p) = (v1,..., vd)
    # voxel: p = (px, py, pz)

    #files = []
    #bb = []
    final_offset_vec = []
    final_feature_vec = []

    scan_paths_dict = get_dict_of_paths(ct_scan_path)
    bb_paths_dict_150 = get_dict_of_paths(gt_bb_path, organ)

    bb_counter = 0
    for key in sorted(scan_paths_dict.keys()):
        #files.append(scan_paths_dict[key])
        #bb.append(bb_paths_dict_150[key])
        file = scan_paths_dict[key]
        bounding_box = bb_paths_dict_150[key]



    #for file in files:
        print(file)
        #TODO: replace with SITK
        # load Nifti format image and save image in numpy array
        img, data = nifti_loader(file)

        # get image affine from header
        spacing_x, spacing_y, spacing_z, offset_x, offset_y, offset_z = nifti_image_affine_reader(img)

        # read bounding box for every organ
        # Bounding Box 6 vector bc=(bL, bR, bA, bP, bH, bF), position of BB walls

        bb_xmin, bb_ymin, bb_zmin, bb_xmax, bb_ymax, bb_zmax = bounding_box_reader(bounding_box, bb_counter)
        bb_coordinates = []
        bb_coordinates.append(bb_xmin)
        bb_coordinates.append(bb_xmax)
        bb_coordinates.append(bb_ymin)
        bb_coordinates.append(bb_ymax)
        bb_coordinates.append(bb_zmin)
        bb_coordinates.append(bb_zmax)

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
        final_feature_vec, final_offset_vec = get_final_vectors(
            data, training_xyz_min, training_xyz_max,
            spacing_x, spacing_y, spacing_z,
            offset_x, offset_y, offset_z,
            bb_coordinates,
            displacement_x, displacement_y, displacement_z, final_feature_vec,
            final_offset_vec)

        bb_counter += 1

    print("Elapsed time: ", (timer() - start) / 60)

    start = timer()
    # use files to train a random forest regressor
    print("start training RRF for {}".format(organ))
    regressor = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10)
    regressor.fit(final_feature_vec, final_offset_vec)
    # Output a pickle file for the model
    joblib.dump(regressor, '{}RRF_{}.pkl'.format(rrf_path, organ))
    print("{} done".format(organ))
    final_offset_vec = []

    print("Elapsed time: ", (timer() - start) / 60)
