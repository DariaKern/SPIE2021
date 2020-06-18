"""
Execute:
1.Define needed variables:
    replace local standard paths,
    select organ to segment,
    define test split
2.GPU & Paths: leave as is
3.Load Data: leave as is
4.Preprocess Data: can be commented out after pre-processing was done once
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
from UNet import generate_U_Net, train_U_Net, plot_history
from Evaluation import calculate_loss_and_accuracy, calculate_hausdorff_distance, \
    calculate_label_overlap_measures, evaluate_predictions, \
    create_excel_sheet, fill_excel_sheet


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
#get_segmentation_masks(results, save_path_X_test, save_path_results, ORGAN, THRESH)

'''_____________________________________________________________________________________________'''
'''|.................................POSTPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# resample files to make them fit into the respective Bounding Box (??x??x??)
#resample_files_reverse(save_path_results, save_path_rr, dict_organ_gt_box_paths, dict_scan_files)

# put area of interest back into original position
#crop_files_reverse(save_path_rr, save_path_rc, dict_organ_gt_box_paths, dict_scan_files)


'''_____________________________________________________________________________________________'''
'''|......................................EVALUATE..............................................|'''
'''_____________________________________________________________________________________________'''

calculate_loss_and_accuracy(model, X_test, y_test)
create_excel_sheet(ORGAN)
fill_excel_sheet(save_path_results, save_path_y_test, ORGAN)

'''
Perfect match:
average hausdorff distance 0.0
hausdorff distance 0.0
dice coefficient 1.0
mean overlap 1.0
volume similarity 0.0
'''