from RRF_Prepare import prepare, create_gt_bb
#from RRF_Evaluate import evaluate
from RRF_Train import train

GT_SEG_PATH = "/Data/Daria/RRF/GT-SEG/"
GT_BB_PATH = "/Data/Daria/RRF/GT-BB/"
SCAN_PATH = "/Data/Daria/RRF/CT-SCANS/"
SAVE_PATH = "/Data/Daria/RRF/"

#prepare(GT_SEG_PATH, GT_BB_PATH)
#evaluate()
in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step3 voxel type, spacing/GT-SEG/"
out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/GT-BB/"
create_gt_bb(in_path, out_path)