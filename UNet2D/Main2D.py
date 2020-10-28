import tensorflow as tf
from Prepare2D import prepare2D, split_train_and_test2D
from Train2D import train2D
from Apply2D import apply2D
from Evaluate2D import evaluate2D, summarize_eval2D
from DATA.normalize import set_direction, set_origin, \
    set_voxeltype, set_spacing, check_all, change_segmentation_colorcode

'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''_____________________________________________________________________________________________'''

# Darias local standard paths (NEEDED)
SCAN_PATH = "/Data/Daria/DATA/CT-SCANS/"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG/"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB/"
RRF_BB_PATH = "/Data/Daria/DATA/GT-BB/"
#RRF_BB_PATH = "/Data/Daria/DATA/BB/"
SAVE_PATH = "/Data/Daria/DATA/"


# organ to segment (NEEDED)
# INFO: DELETE X train, X, test, y train and y test before switching to another organ
# choose from 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'
ORGAN = "liver"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
SPLIT = 0.2

# define threshold for segmentation mask
# recommended thresh: 0.5, for pancreas: 0.3
THRESH = 0.5

# Define input image size dimensions for preparation (actual dimensions are one less)
PREP_DIMENSIONS = [96, 96, 96, 1]


# define validation split  (Default = 0.1)
# 0.00 (0%) - 1.00 (100%) percentage of validation files among Test files
VAL_SPLIT = 0.1

# define batch size (Default = 15)
BATCH = 8

# define number of epochs (Default = 50)
EPOCHS = 100

#CUSTOM_TEST_SET = [7, 17, 15, 47, 22]
CUSTOM_TEST_SET = None

'''_____________________________________________________________________________________________'''
'''|........................................GPU................................................|'''
'''_____________________________________________________________________________________________'''

# GPU Use fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''_____________________________________________________________________________________________'''
'''|................................METHODS....................................|'''
'''_____________________________________________________________________________________________'''


def run_x_times(times):
    for x in range(0, times):
        number = x
        #custom_test_set = [19,63]
        #test_set, train_set = split_train_and_test2D(SCAN_PATH, SPLIT, custom_test_set)

        test_set, train_set = split_train_and_test2D(SCAN_PATH, SPLIT)
        for organ in ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']:
        #for organ in ['liver']:
            if organ == 'pancreas':
                thresh = 0.3
            else:
                thresh = 0.5
            prepare2D(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, PREP_DIMENSIONS, SPLIT, organ, test_set)
            train2D(SAVE_PATH, PREP_DIMENSIONS, organ, VAL_SPLIT, BATCH, EPOCHS)
            apply2D(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, PREP_DIMENSIONS, organ, thresh)
            evaluate2D(SAVE_PATH, organ, number)
        #exit()

run_x_times(10)

