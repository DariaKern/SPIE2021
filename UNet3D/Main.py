#TODO write workflow into KFoldCrossValidation
import tensorflow as tf
from Prepare_3D import prepare, split_train_and_test
from Train import train
from Apply import apply
from Evaluate import evaluate
import time
import DATA.prepare_data as D
from KFoldCrossValidation import run_KfoldCV, summarize_eval, summarize_metrics
from DATA.prepare_data import set_direction, set_origin, \
    set_voxeltype, set_spacing, check_all, change_segmentation_colorcode

'''_____________________________________________________________________________________________'''
'''|................................DEFINE NEEDED VARIABLES....................................|'''
'''_____________________________________________________________________________________________'''

SCAN_PATH = "/Data/Daria/SPIE2021/CT-SCANS/"
GT_SEG_PATH = "/Data/Daria/SPIE2021/GT-SEG/"
GT_BB_PATH = "/Data/Daria/SPIE2021/GT-BB/"
RRF_BB_PATH = "/Data/Daria/SPIE2021/GT-BB/"
#RRF_BB_PATH = "/Data/Daria/SPIE2021/BB/"
SAVE_PATH = "/Data/Daria/SPIE2021/"


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
EPOCHS = 100

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

#organs = ['liver', 'left_kidney', 'right_kidney', 'spleen', 'pancreas']
direction= "axial"
organs = ['pancreas']
run_KfoldCV(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, BATCH, EPOCHS, organs, direction)
path = "/home/daria/Desktop/Data/Daria/SPIE2021/eval/"

#for organ in organs:
    #summarize_eval(path, organ)
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


'''
in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED PREP/Data2/step3 voxel type, spacing/GT-SEG/"
out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED PREP/Data2/step4 segmentation color/"
organs = [6, 3, 2, 1, 11]
change_segmentation_colorcode(organs, in_path, out_path)
'''


#https://docs.python-guide.org/writing/documentation/
'''
for organ in ['left_kidney', 'right_kidney', 'spleen', 'pancreas']:
    create_x_train(SCAN_PATH, GT_BB_PATH, SAVE_PATH, [96,96,96], None, organ)
    create_y_train(GT_SEG_PATH, GT_BB_PATH, SAVE_PATH, [96,96,96], None, organ)
    train_3DUNet(SAVE_PATH, [96,96,96,1], organ )
'''

'''
def get_full_bounds(img_path,bb_path):
    orig_img = sitk.ReadImage(img_path)
    width_mm= orig_img.GetWidth()*2 - 10
    height_mm = orig_img.GetHeight()*2 -10
    depth_mm = orig_img.GetDepth()*2 -10


    bounds = [ -width_mm, -100, -height_mm,10,10, depth_mm]

    # define bb as cube
    vtk_cube = vtk.vtkCubeSource()
    vtk_cube.SetBounds(bounds)
    vtk_cube.Update()
    output = vtk_cube.GetOutput()

    # save bounding box object to file
    bb_name = "{}_{}_bb.vtk".format(0, 160)
    save_path = "{}{}".format(bb_path, bb_name)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(output)
    writer.SetFileName(save_path)
    writer.Update()
'''