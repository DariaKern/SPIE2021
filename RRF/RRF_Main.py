from RRF.RRF_Prepare import create_gt_bb

GT_SEG_PATH = "/Data/Daria/RRF/GT-SEG/"
GT_BB_PATH = "/Data/Daria/RRF/GT-BB/"
SCAN_PATH = "/Data/Daria/RRF/CT-SCANS/"
SAVE_PATH = "/Data/Daria/RRF/"

in_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/step4 segmentation color/"
out_path = "/home/daria/Desktop/Data/Daria/NORMALIZED DATA/Data2/GT-BB/"
create_gt_bb(in_path, out_path)