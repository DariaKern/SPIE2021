'''
- files need to be in .nii.gz format
- no mirrored files allowed
'''
from DATA.prepare_data import prepare_data
from RRF.Apply_RRF import apply_RRF
from UNet3D.Prepare_3D import create_x_test, split_train_and_test
from UNet3D.Apply_3D import apply_3DUnet


WF_PATH = "/home/daria/Desktop/Data/Daria/Workflow/WF/"
CT_PATH = "/home/daria/Desktop/Data/Daria/Workflow/WF/CT/"
BB_PATH = "/home/daria/Desktop/Data/Daria/Workflow/WF/BB/"
INPUT_DATA = "/home/daria/Desktop/Data/Daria/Workflow/INPUT DATA/"
TEMP_DIR = "/home/daria/Desktop/Data/Daria/Workflow/WF/temp/"

#prepare_data(INPUT_DATA, CT_PATH, TEMP_DIR)
#apply_RRF(CT_PATH, WF_PATH, BB_PATH, "liver")
#create_x_test(CT_PATH, BB_PATH, WF_PATH, [96, 96, 96], None, "liver")
#apply_3DUnet(CT_PATH, BB_PATH, WF_PATH, [96, 96, 96, 1], "liver", 0.5)


SCAN_PATH = "/Data/Daria/SPIE2021/CT-SCANS/"
GT_SEG_PATH = "/Data/Daria/SPIE2021/GT-SEG/"
GT_BB_PATH = "/Data/Daria/SPIE2021/GT-BB/"
RRF_BB_PATH = "/Data/Daria/SPIE2021/GT-BB/"
SAVE_PATH = "/Data/Daria/SPIE2021/"

