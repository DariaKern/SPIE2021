import tensorflow as tf
from Prepare import prepare, split_train_and_test
from Train import train
from Apply import apply
from Evaluate import evaluate
import time
from KFoldCrossValidation import run_KfoldCV, summarize_eval, summarize_metrics
from DATA.normalize import set_direction, set_origin, \
    set_voxeltype, set_spacing, check_all, change_segmentation_colorcode

'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''_____________________________________________________________________________________________'''

# Darias local standard paths (NEEDED)
'''
SCAN_PATH = "/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/CT-SCANS/"
GT_SEG_PATH = "/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/GT-SEG"
GT_BB_PATH = "/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/GT-BB/"
RRF_BB_PATH = "/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/BB/"
SAVE_PATH = "/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/"
'''

SCAN_PATH = "/Data/Daria/DATA/CT-SCANS/"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG/"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB/"
RRF_BB_PATH = "/Data/Daria/DATA/GT-BB/"
#RRF_BB_PATH = "/Data/Daria/DATA/BB/"
SAVE_PATH = "/Data/Daria/DATA/"


# organ to segment (NEEDED)
# INFO: DELETE X train, X, test, y train and y test before switching to another organ
# choose from 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'
ORGAN = "pancreas"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
SPLIT = 0.2

# define threshold for segmentation mask
# recommended thresh: 0.5, for pancreas: 0.3
THRESH = 0.5

# Define input image size
DIMENSIONS = [96, 96, 96, 1]

# define validation split  (Default = 0.1)
# 0.00 (0%) - 1.00 (100%) percentage of validation files among Test files
VAL_SPLIT = 0.0

# define batch size (Default = 15)
BATCH = 8

# define number of epochs (Default = 50)
EPOCHS = 50

#CUSTOM_TEST_SET = [7, 17, 15, 47, 22]
#CUSTOM_TEST_SET = [19]
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
organs = ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']
direction= "axial"
#organs = ['liver']
#run_KfoldCV(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, BATCH, EPOCHS, organs, direction)
path = "/home/daria/Desktop/Data/Daria/EVAL/8/50 Epochs/2D axial/"
for organ in organs:
    summarize_eval(path, organ)
#summarize_metrics(path, "dice")
#summarize_metrics(path, "avd")
#summarize_metrics(path, "hd")



def run_x_times(times):
    for x in range(0, times):
        number = x + 5
        #custom_test_set = [19]
        #test_set, train_set = split_train_and_test(SCAN_PATH, SPLIT, custom_test_set)

        test_set, train_set = split_train_and_test(SCAN_PATH, SPLIT)
        #for organ in ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']:
        for organ in ['pancreas']:
            if organ == 'pancreas':
                thresh = 0.3
            else:
                thresh = 0.5
            prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, organ, train_set, test_set)
            start = time.time()
            train(SAVE_PATH, DIMENSIONS, organ, VAL_SPLIT, BATCH, EPOCHS)
            end = time.time()
            elapsed_time = end - start
            apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, organ, thresh)
            evaluate(SAVE_PATH, organ, number, elapsed_time)
        #exit()

#run_x_times(10)


#set_direction(in_dir1, out_dir)
#set_origin(in_dir, out_dir)
#set_voxeltype(in_dir, out_dir)
#set_spacing(in_dir, out_dir)
#check_all(out_dir)

#from RRF.RRF_Prepare import create_gt_bb_alternative
#in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step3 voxel type, spacing/GT-SEG/"
#out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/GT-BB/"
#create_gt_bb_alternative(in_path, out_path)

'''
in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step3 voxel type, spacing/GT-SEG/"
out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step4 segmentation color/"
organs = [6, 3, 2, 1, 11]
change_segmentation_colorcode(organs, in_path, out_path)
'''


#https://docs.python-guide.org/writing/documentation/