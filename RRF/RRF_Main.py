from DATA.prepare_data import create_gt_bb
from RRF.RRF_Train import train_RRF, train_RRF_old
from RRF.RRF_Apply import apply_RRF, apply_RRF_old
GT_SEG_PATH = "/Data/Daria/SPIE2021/GT-SEG/"
GT_BB_PATH = "/Data/Daria/SPIE2021/GT-BB/"
SCAN_PATH = "/Data/Daria/SPIE2021/CT-SCANS/"
SAVE_PATH = "/Data/Daria/SPIE2021/TESTRRF/"
BB_PATH = "/Data/Daria/SPIE2021/TESTRRF/"

in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step4 segmentation color/"
out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/GT-BB/"
#create_gt_bb(in_path, out_path)

train_RRF_old(SCAN_PATH, GT_BB_PATH, SAVE_PATH, "liver") # about 15minutes
#train_RRF(SCAN_PATH, GT_BB_PATH, SAVE_PATH, "liver") # about 15minutes
#apply_RRF(SCAN_PATH, SAVE_PATH, BB_PATH, "liver")
#apply_RRF_old(SCAN_PATH, SAVE_PATH, BB_PATH, "liver")