'''
def calculate_surface_distance(gt_img_path, pred_img_path):
    # load images
    gt_img = sitk.ReadImage(gt_img_path)
    pred_img = sitk.ReadImage(pred_img_path)

    # calculate surface distance
    gt_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_img, squaredDistance=False, useImageSpacing=True))
    pred_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))

    gt_surface = sitk.LabelContour(gt_img)
    pred_surface = sitk.LabelContour(pred_img)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(gt_surface)
    num_gt_surface_pixels = int(statistics_image_filter.GetSum())
    statistics_image_filter.Execute(pred_surface)
    num_pred_surface_pixels = int(statistics_image_filter.GetSum())

    gt2pred_dist_map =pred_dist_map*sitk.Cast(gt_surface, sitk.sitkFloat32)
    pred2gt_dist_map =gt_dist_map*sitk.Cast(pred_surface, sitk.sitkFloat32)
    gt2pred_dist_map_arr = sitk.GetArrayViewFromImage(gt2pred_dist_map)
    pred2gt_dist_map_arr = sitk.GetArrayViewFromImage(pred2gt_dist_map)

    gt2pred_dist = list(gt2pred_dist_map_arr[gt2pred_dist_map_arr!=0])
    gt2pred_dist = gt2pred_dist + list(np.zeros(num_gt_surface_pixels - len(gt2pred_dist)))

    pred2gt_dist = list(pred2gt_dist_map_arr[pred2gt_dist_map_arr!=0])
    pred2gt_dist = pred2gt_dist + list(np.zeros(num_pred_surface_pixels - len(pred2gt_dist)))

    all_dist = pred2gt_dist + gt2pred_dist
    mean_surface_dist = np.mean(all_dist)
    max_surface_dist = np.max(all_dist)
    median_surface_dist = np.median(all_dist)
    std_surface_dist = np.std(all_dist)


    print("######xxxxxx######")
    print(mean_surface_dist)
    print(max_surface_dist)
    print(median_surface_dist)
    print(std_surface_dist)



    print("mean surface distance {}".format(mean_surface_dist))

    return mean_surface_dist
'''


'''
RRF TRAIN MIetzner old code

# funktioniert mit Mietzner Daten
GT_BB_PATH = "/Data/Daria/RRF/GT-BB/"
#GT_BB_PATH = "/Data/Daria/RRF/BB-TEST/"
#GT_BB_PATH = "/Data/Daria/RRF/_GT-MIETZNER/"
SCAN_PATH = "/Data/Daria/RRF/CT-SCANS/"
#SCAN_PATH= "/Data/Daria/RRF/SCANS-TEST/"
#SCAN_PATH = "/Data/Daria/RRF/_CT-SCANS-MIETZNER/"
SAVE_PATH = "/Data/Daria/RRF/"

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

BEFORE (Mietzner Code):
order or files in list didn't match
input:
print(bb_170)
print(files)
output:
['/Data/Daria/RRF/_GT-MIETZNER/seg0.nii_170_bb.vtk', '/Data/Daria/RRF/_GT-MIETZNER/seg1.nii_170_bb.vtk', ...
['/Data/Daria/RRF/_CT-SCANS-MIETZNER/26.nii.gz', '/Data/Daria/RRF/_CT-SCANS-MIETZNER/0.nii.gz', ...
'''

'''
bb_156 = []
bb_157 = []
bb_160 = []
bb_170 = []

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
# Dauer mit Mietzner Daten pro RandomForestRegressor ca 3h
# Dauer mit neuen Daten insgesamt für ALLE RRF nur 40 Minuten ( ?komisch?)
# -------------------------------------------------------------------------------------------
# use files to train a random forest regressor
print("start training RRF for 150")
regressor_150 = RandomForestRegressor(n_estimators=trees, random_state=0, max_depth=depth)
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


'''
RRF_apply Mietzner old code
regressor_150 = joblib.load('{}model_150.pkl'.format(rrf_path))
regressor_156 = joblib.load('{}model_156.pkl'.format(rrf_path))
regressor_157 = joblib.load('{}model_157.pkl'.format(rrf_path))
regressor_160 = joblib.load('{}model_160.pkl'.format(rrf_path))
regressor_170 = joblib.load('{}model_170.pkl'.format(rrf_path))

files = []

for entry in os.scandir(ct_scan_path):
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

    make_bounding_box(new_bb_150, file, bb_path)
    make_bounding_box(new_bb_156, file, bb_path)
    make_bounding_box(new_bb_157, file, bb_path)
    make_bounding_box(new_bb_160, file, bb_path)
    make_bounding_box(new_bb_170, file, bb_path)

print('done')
'''