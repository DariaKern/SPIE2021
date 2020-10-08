import tensorflow as tf
from Prepare import prepare, split_train_and_test
from Train import train
from Apply import apply
from Evaluate import evaluate, summarize_eval
from DATA.normalize import set_direction, set_origin, \
    set_voxeltype, set_spacing, check_all

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
ORGAN = "liver"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
SPLIT = 0.4

# define threshold for segmentation mask
# recommended thresh: 0.5, for pancreas: 0.3
THRESH = 0.5

# Define input image size
DIMENSIONS = [64, 64, 64, 1]

# define validation split  (Default = 0.1)
# 0.00 (0%) - 1.00 (100%) percentage of validation files among Test files
VAL_SPLIT = 0.1

# define batch size (Default = 15)
BATCH = 5

# define number of epochs (Default = 50)
EPOCHS = 50

#CUSTOM_TEST_SET = [7, 17, 15, 47, 22]
#CUSTOM_TEST_SET = [7]
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

#prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, SPLIT, ORGAN, CUSTOM_TEST_SET)
#train(SAVE_PATH, DIMENSIONS, ORGAN, VAL_SPLIT, BATCH, EPOCHS)
#apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, ORGAN, THRESH)


def run_x_times(times):
    for x in range(0, times):
        number = x + 88
        #test_set = [1,7,8,9,12,13,15,16,17,25,26,28,29,31,33,35,41,44,45,46]
        test_set, train_set = split_train_and_test(SCAN_PATH, SPLIT)
        print(test_set)
        for organ in ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']:
        #for organ in ['liver']:
            if organ == 'pancreas':
                thresh = 0.3
            else:
                thresh = 0.5
            prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, SPLIT, organ, test_set)
            train(SAVE_PATH, DIMENSIONS, organ, VAL_SPLIT, BATCH, EPOCHS)
            apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, organ, thresh)
            evaluate(SAVE_PATH, organ, number)




#for organ in ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']:
    #summarize_eval(SAVE_PATH, organ)
#run_x_times(100)

out_dir = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step3 voxel type, spacing/CT-SCANS/"
in_dir = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/seg/"

#set_direction(in_dir1, out_dir)
#set_origin(in_dir, out_dir)
#set_voxeltype(in_dir, out_dir)
#set_spacing(in_dir, out_dir)
check_all(out_dir)