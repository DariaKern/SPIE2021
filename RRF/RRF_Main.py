from RRF_Prepare import prepare
from RRF_Evaluate import evaluate

GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG/"
GT_BB_PATH = "/Data/Daria/new bb/"

prepare(GT_SEG_PATH, GT_BB_PATH)
#evaluate()