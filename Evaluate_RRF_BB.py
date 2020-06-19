'''
Evaluate using RRF BBs
use only after Model was trained with Main
'''

from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
from Data import get_dict_of_files, get_dict_of_paths, \
    crop_out_bbs, resample_files, \
    get_training_data, get_segmentation_masks, \
    resample_files_reverse, crop_files_reverse
from Evaluation import calculate_loss_and_accuracy, create_excel_sheet, \
    fill_excel_sheet, get_dict_of_test_data, get_segmentation_masks2

'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''|................................MUST BE SAME AS IN MAIN....................................|'''
'''_____________________________________________________________________________________________'''

# Darias local standard paths (NEEDED)
CT_SCANS_PATH = "/Data/Daria/DATA/CT-Scans"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB"
RRF_BB_PATH = "/Data/Daria/DATA/BB"
SAVE_PATH = "/Data/Daria/DATA/RRF BB EVALUATE/"
MODEL_SAVE_PATH = "/Data/Daria/DATA/RESULTS/"

# organ to segment (NEEDED)
# INFO: DELETE X train, X, test, y train and y test before switching to another organ
# choose from 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'
ORGAN = "spleen"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
PERCENTAGE_TEST_SPLIT = 0.1

# define threshold for segmentation mask
# recommended thresh: 0.5, for pancreas: 0.3
THRESH = 0.5

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
save_path_X_test = "{}X test/".format(SAVE_PATH)
save_path_results = "{}Result-SEG/".format(SAVE_PATH)
save_path_rr = "{}reverse resample/".format(SAVE_PATH)
save_path_rc = "{}reverse crop/".format(SAVE_PATH)
save_path_gt_seg = "{}filtered gt seg/".format(SAVE_PATH)
Path(save_path_cropped_scans).mkdir(parents=True, exist_ok=True)
Path(save_path_cropped_seg).mkdir(parents=True, exist_ok=True)
Path(save_path_X_test).mkdir(parents=True, exist_ok=True)
Path(save_path_results).mkdir(parents=True, exist_ok=True)
Path(save_path_rr).mkdir(parents=True, exist_ok=True)
Path(save_path_rc).mkdir(parents=True, exist_ok=True)
Path(save_path_gt_seg).mkdir(parents=True, exist_ok=True)



'''_____________________________________________________________________________________________'''
'''|.......................................LOAD DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# load data
dict_scan_files = get_dict_of_files(CT_SCANS_PATH)  # load all ct scans
dict_gt_seg_files = get_dict_of_files(GT_SEG_PATH)  # load all gt segmentations
dict_gt_organ_box_paths = get_dict_of_paths(GT_BB_PATH, ORGAN)  # load all paths to gt bbs of organ
dict_rrf_organ_box_paths = get_dict_of_paths(RRF_BB_PATH, ORGAN)   # load all paths to rrf bbs of organ

# take only test data
dict_scan_files = get_dict_of_test_data(dict_scan_files, PERCENTAGE_TEST_SPLIT)
dict_gt_seg_files = get_dict_of_test_data(dict_gt_seg_files, PERCENTAGE_TEST_SPLIT)
dict_gt_organ_box_paths = get_dict_of_test_data(dict_gt_organ_box_paths, PERCENTAGE_TEST_SPLIT)
dict_rrf_organ_box_paths = get_dict_of_test_data(dict_rrf_organ_box_paths, PERCENTAGE_TEST_SPLIT)

'''_____________________________________________________________________________________________'''
'''|.................................PREPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# crop out area of interest where the organ is
crop_out_bbs(dict_scan_files, dict_rrf_organ_box_paths, save_path_cropped_scans)    # X test

# resample files to make them fit into the U-Net
# INFO: U-Net:(Width, Height, Depth) resample files:(Depth, Height, Width)
resample_files(save_path_cropped_scans, save_path_X_test, 64, 64, 64)

get_segmentation_masks2(dict_gt_seg_files, CT_SCANS_PATH, save_path_gt_seg, ORGAN, dict_gt_organ_box_paths)

'''_____________________________________________________________________________________________'''
'''|......................................PREPARE TEST DATA....................................|'''
'''_____________________________________________________________________________________________'''

# get test data in format Width, Height, Depth, Channels
X_test = get_training_data(save_path_X_test)

'''_____________________________________________________________________________________________'''
'''|...................................APPLY U-NET.............................................|'''
'''_____________________________________________________________________________________________'''

# load U-Net
model = load_model("{}{}U-Net.h5".format(MODEL_SAVE_PATH, ORGAN))

# apply U-Net on test data and get results in format Width, Height, Depth, Channels
results = model.predict(X_test, verbose=1)

# generate segmentation masks from results
get_segmentation_masks(results, save_path_X_test, save_path_results, ORGAN, THRESH)

'''_____________________________________________________________________________________________'''
'''|.................................POSTPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# resample files to make them fit into the respective Bounding Box (??x??x??)
resample_files_reverse(save_path_results, save_path_rr, dict_rrf_organ_box_paths, dict_scan_files)

# put area of interest back into original position
crop_files_reverse(save_path_rr, save_path_rc, dict_rrf_organ_box_paths, dict_scan_files)


'''_____________________________________________________________________________________________'''
'''|......................................EVALUATE..............................................|'''
'''_____________________________________________________________________________________________'''

create_excel_sheet(ORGAN, SAVE_PATH)
fill_excel_sheet(save_path_rc, save_path_gt_seg, ORGAN, SAVE_PATH)
