import tensorflow as tf
from Prepare import prepare
from Train import train
from Apply import apply
from Evaluate import evaluate

'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''_____________________________________________________________________________________________'''

# Darias local standard paths (NEEDED)
SCAN_PATH = "/Data/Daria/DATA/CT-Scans/"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG/"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB/"
RRF_BB_PATH = "/Data/Daria/DATA/BB/"
SAVE_PATH = "/Data/Daria/DATA/"

# organ to segment (NEEDED)
# INFO: DELETE X train, X, test, y train and y test before switching to another organ
# choose from 'liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas'
ORGAN = "liver"

# define train-test split (NEEDED)
# 0.00 (0%) - 1.00 (100%) percentage of test files among All files
SPLIT = 0.1

# define threshold for segmentation mask
# recommended thresh: 0.5, for pancreas: 0.3
THRESH = 0.5

# define validation split  (Default = 0.1)
# 0.00 (0%) - 1.00 (100%) percentage of validation files among Test files

# define batch size (Default = 15)

# define number of epochs (Default = 50)

'''_____________________________________________________________________________________________'''
'''|........................................GPU................................................|'''
'''_____________________________________________________________________________________________'''

# GPU Use fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


#prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, SPLIT, ORGAN)
#train(SAVE_PATH, ORGAN)
#apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, ORGAN, THRESH)
evaluate(SAVE_PATH, ORGAN)


# TODO:
"""
crop files reverse noch eine MEthode mit crop file reverse für einzelne Datei
"""