from helpers import compare, compare2, create_excel_sheet, write_into_sheet

GT_BB_PATH = "/Data/Daria/DATA/GT-BB/"
BB_PATH = "/Data/Daria/DATA/BB/"
SAVE_PATH = "/Data/Daria/"


create_excel_sheet(SAVE_PATH, "liver")

# compare liver bbs
dict_bb_diff_liver, liver_mean, liver_std  = compare(GT_BB_PATH, BB_PATH, "liver")
write_into_sheet(liver_mean, 3, SAVE_PATH)
write_into_sheet(liver_std, 4, SAVE_PATH)
liver_mean_all_dimensions = compare2(GT_BB_PATH, BB_PATH, "liver")

# compare left kidney bbs
dict_bb_diff_l_kidney, l_kidney_mean, l_kidney_std = compare(GT_BB_PATH, BB_PATH, "left_kidney")
write_into_sheet(l_kidney_mean, 8, SAVE_PATH)
write_into_sheet(l_kidney_std, 9, SAVE_PATH)
l_kidney_mean_all_dimensions = compare2(GT_BB_PATH, BB_PATH, "left_kidney")

# compare right kidney bbs
dict_bb_diff_r_kidney, r_kidney_mean, r_kidney_std = compare(GT_BB_PATH, BB_PATH, "right_kidney")
write_into_sheet(r_kidney_mean, 13, SAVE_PATH)
write_into_sheet(r_kidney_std, 14, SAVE_PATH)
r_kidney_mean_all_dimensions = compare2(GT_BB_PATH, BB_PATH, "right_kidney")

# compare spleen bbs
dict_bb_diff_spleen, spleen_mean, spleen_std = compare(GT_BB_PATH, BB_PATH, "spleen")
write_into_sheet(spleen_mean, 18, SAVE_PATH)
write_into_sheet(spleen_std, 19, SAVE_PATH)
spleen_mean_all_dimensions = compare2(GT_BB_PATH, BB_PATH, "spleen")

# compare pancreas bbs
dict_bb_diff_pancreas, pancreas_mean, pancreas_std = compare(GT_BB_PATH, BB_PATH, "pancreas")
write_into_sheet(pancreas_mean, 23, SAVE_PATH)
write_into_sheet(pancreas_std, 24, SAVE_PATH)
pancreas_mean_all_dimensions = compare2(GT_BB_PATH, BB_PATH, "pancreas")




