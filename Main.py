"""
Execute:
1.Define needed variables:
    replace local standard paths,
    select organ to segment,
    define test split
2.GPU & Paths: leave as is
3.Load Data: leave as is
4.Preprocess Data: can be commented out after preprocessing was done once
5.Prepare Training & Test Data: leave as is
6.Train U-Net: can be commented out after training was done once
7.Apply U-Net: leave as is
8.Postprocess Data

Results folder creation timeline:
1. Cropped-CT-Scans & Cropped-SEG (after cropping out bb area of interest)
2. X train, y train, X test, y test (after resampling to input size of U-Net)
3. Result-SEG (after applying the U-Net on X test)
4. reverse resample (after resampling Result-SEG from output size of U-Net back to bb area of interest size)
5. reverse crop (after putting reverse resample (area of interest) back to where it was cropped out from)
"""

from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
from Data import get_dict_of_files, get_dict_of_paths, \
    check_if_all_files_are_complete, crop_out_bbs, resample_files, \
    get_training_data, get_segmentation_masks, split_train_test, \
    resample_files_reverse, crop_files_reverse
from UNet import generate_U_Net, train_U_Net, plot_history, generate_metrics


'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''_____________________________________________________________________________________________'''

# Darias local standard paths (NEEDED)
CT_SCANS_PATH = "/Data/Daria/DATA/CT-Scans"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB"
BB_PATH = "/Data/Daria/DATA/BB"
SAVE_PATH = "/Data/Daria/DATA/RESULTS/"

# organ to segment (NEEDED)
# choose from 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'
ORGAN = "liver"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
PERCENTAGE_TEST_SPLIT = 0.1

# define threshold for segmentation mask
THRESH = 0.5

# define validation split  (Default = 0.1)
# 0.00 (0%) - 1.00 (100%) percentage of validation files among Test files

# define batch size (Default = 15)

# define number of epochs (Default = 50)

'''_____________________________________________________________________________________________'''
'''|........................................GPU & PATHS........................................|'''
'''_____________________________________________________________________________________________'''

# GPU Use fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create missing folders
save_path_cropped_scans = "{}Cropped-CT-Scans/".format(SAVE_PATH)
save_path_cropped_seg = "{}Cropped-SEG/".format(SAVE_PATH)
save_path_X_train = "{}X train/".format(SAVE_PATH)
save_path_y_train = "{}y train/".format(SAVE_PATH)
save_path_X_test = "{}X test/".format(SAVE_PATH)
save_path_y_test = "{}y test/".format(SAVE_PATH)
save_path_results = "{}Result-SEG/".format(SAVE_PATH)
save_path_rr = "{}reverse resample/".format(SAVE_PATH)
save_path_rc = "{}reverse crop/".format(SAVE_PATH)
Path(save_path_cropped_scans).mkdir(parents=True, exist_ok=True)
Path(save_path_cropped_seg).mkdir(parents=True, exist_ok=True)
Path(save_path_X_train).mkdir(parents=True, exist_ok=True)
Path(save_path_y_train).mkdir(parents=True, exist_ok=True)
Path(save_path_X_test).mkdir(parents=True, exist_ok=True)
Path(save_path_y_test).mkdir(parents=True, exist_ok=True)
Path(save_path_results).mkdir(parents=True, exist_ok=True)
Path(save_path_rr).mkdir(parents=True, exist_ok=True)
Path(save_path_rc).mkdir(parents=True, exist_ok=True)


'''_____________________________________________________________________________________________'''
'''|.......................................LOAD DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# load data
dict_scan_files = get_dict_of_files(CT_SCANS_PATH)  # load all ct scans
dict_gt_seg_files = get_dict_of_files(GT_SEG_PATH)  # load all gt segmentations
dict_organ_gt_box_paths = get_dict_of_paths(GT_BB_PATH, ORGAN)  # load all paths to gt bbs of organ
check_if_all_files_are_complete(dict_scan_files, dict_gt_seg_files, dict_organ_gt_box_paths)


'''_____________________________________________________________________________________________'''
'''|.................................PREPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# crop out area of interest where the organ is
#crop_out_bbs(dict_scan_files, dict_organ_gt_box_paths, save_path_cropped_scans)
#crop_out_bbs(dict_gt_seg_files, dict_organ_gt_box_paths, save_path_cropped_seg, ORGAN)

# resample files to make them fit into the U-Net
# INFO: U-Net:(Width, Height, Depth) resample files:(Depth, Height, Width)
#resample_files(save_path_cropped_scans, save_path_X_train, 64, 64, 64)
#resample_files(save_path_cropped_seg, save_path_y_train, 64,  64, 64)

'''_____________________________________________________________________________________________'''
'''|...............................PREPARE TRAINING & TEST DATA................................|'''
'''_____________________________________________________________________________________________'''

split_train_test(save_path_X_train, save_path_X_test, PERCENTAGE_TEST_SPLIT)
split_train_test(save_path_y_train, save_path_y_test, PERCENTAGE_TEST_SPLIT)

# get training data in format Width, Height, Depth, Channels
X_train = get_training_data(save_path_X_train)
y_train = get_training_data(save_path_y_train, "y")

# get test data in format Width, Height, Depth, Channels
X_test = get_training_data(save_path_X_test)
y_test = get_training_data(save_path_y_test, "y")


'''_____________________________________________________________________________________________'''
'''|...................................TRAIN U-NET.............................................|'''
'''_____________________________________________________________________________________________'''

# generate the U-Net model (Width, Height, Depth, Channels)
#architecture = generate_U_Net(64, 64, 64, 1)

# train U-Net on training data and save it
#model, history = train_U_Net(architecture, X_train, y_train, SAVE_PATH)
#plot_history(history)


'''_____________________________________________________________________________________________'''
'''|...................................APPLY U-NET.............................................|'''
'''_____________________________________________________________________________________________'''

# load U-Net
model = load_model("{}U-Net.h5".format(SAVE_PATH))

# apply U-Net on test data and get results in format Width, Height, Depth, Channels
results = model.predict(X_test, verbose=1)

# generate segmentation masks from results
get_segmentation_masks(results, save_path_X_test, save_path_results, ORGAN, THRESH)

# Generate generalization metrics (evaluate the model)
generate_metrics(model, X_test, y_test)

'''_____________________________________________________________________________________________'''
'''|.................................POSTPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''
# go through bb files of results

# resample files to make them fit into the respective Bounding Box (??x??x??)
resample_files_reverse(save_path_results, save_path_rr, dict_organ_gt_box_paths, dict_scan_files)

# put area of interest back into original position
crop_files_reverse(save_path_rr, save_path_rc, dict_organ_gt_box_paths, dict_scan_files)
exit()





#TODO Tuesday: Zahlen noch rumprobieren und dann um segembtation mask teil kürzen da es den schon gibt
import nibabel as nib
import numpy as np
from Data import get_bb_coordinates
from helpers import nifti_image_affine_reader, bb_mm_to_vox

result_arr = results[0] # first results
curr_key = 45
orig_img = dict_gt_seg_files[curr_key]
orig_bb = dict_organ_gt_box_paths[curr_key]

# load original image
orig_img_arr = orig_img.get_fdata()

# transform bb from mm to vox
bb_coords = get_bb_coordinates(orig_bb)  # get bb coordinates
print(bb_coords)
spacing, offset = nifti_image_affine_reader(orig_img)
vox_170 = bb_mm_to_vox(bb_coords, spacing, offset)
print('Coordinates of area in vox: ', vox_170)

# make numpy int array
vox_170_int = np.asarray(vox_170)
vox_170_int = vox_170_int.astype(int)


#for x in range(result_arr.shape[0]):
pred_map_170 = np.zeros((orig_img_arr.shape[0], orig_img_arr.shape[1], orig_img_arr.shape[2]))
for x in range(64):
    for y in range(64):
        for z in range(64):
            if result_arr[x][y][z][0] > 0.3:
                x_real = x + region_170[0]
                y_real = y + region_170[2]
                z_real = z + region_170[4]

                pred_map_170[x_real, y_real,z_real] = 170

new_img = nib.Nifti1Image(pred_map_170, orig_img.affine, orig_img.header)
nib.save(new_img, '{}{}.nii.gz'.format(SAVE_PATH, "hehhe"))



'''
# save result #ERROR: Doesn't save the image spacing and stuff
import SimpleITK as sitk
result_img = sitk.GetImageFromArray(pred_map)
sitk.WriteImage(result_img, "{}{}".format(SAVE_PATH, "SITKresult.nii.gz"))

'''