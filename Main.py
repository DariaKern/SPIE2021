"""
Execute:
1.Preprocess Data
2.Prepare Training Data
3.Train U-Net
4.Prepare Test Data
5.Apply U-Net
"""
from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path
from Data import get_dict_of_files, get_dict_of_paths, \
    check_if_all_files_are_complete, crop_out_bbs, resample_files, \
    get_training_data, get_segmentation_mask
from UNet import generate_U_Net, train_U_Net, plot_history, generate_metrics


# GPU Use fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Darias local standard paths (NEEDED)
CT_SCANS_PATH = "/Data/Daria/DATA/CT-Scans"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG"
GT_BB_PATH = "/Data/Daria/DATA/GT-BB"
BB_PATH = "/Data/Daria/DATA/BB"
SAVE_PATH = "/Data/Daria/DATA/RESULTS/"

# create missing folders
save_path_cropped_scans = "{}Cropped-CT-Scans/".format(SAVE_PATH)
save_path_cropped_seg = "{}Cropped-SEG/".format(SAVE_PATH)
save_path_X_train = "{}X train/".format(SAVE_PATH)
save_path_y_train = "{}y train/".format(SAVE_PATH)
save_path_X_test = "{}X test/".format(SAVE_PATH)
save_path_y_test = "{}y test/".format(SAVE_PATH)
save_path_results = "{}Result-SEG/".format(SAVE_PATH)
Path(save_path_cropped_scans).mkdir(parents=True, exist_ok=True)
Path(save_path_cropped_seg).mkdir(parents=True, exist_ok=True)
Path(save_path_X_train).mkdir(parents=True, exist_ok=True)
Path(save_path_y_train).mkdir(parents=True, exist_ok=True)
Path(save_path_X_test).mkdir(parents=True, exist_ok=True)
Path(save_path_y_test).mkdir(parents=True, exist_ok=True)
Path(save_path_results).mkdir(parents=True, exist_ok=True)

'''_____________________________________________________________________________________________'''
'''|.................................PREPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

# load data
dict_scan_files = get_dict_of_files(CT_SCANS_PATH)  # load all ct scans
dict_gt_seg_files = get_dict_of_files(GT_SEG_PATH)  # load all gt segmentations
dict_liver_gt_box_paths = get_dict_of_paths(GT_BB_PATH, "liver")  # load all paths to gt bbs of liver
number_of_patients = check_if_all_files_are_complete(dict_scan_files,
                                                     dict_gt_seg_files,
                                                     dict_liver_gt_box_paths)

# crop out area of interest where the organ is
#crop_out_bbs(dict_scan_files, dict_liver_gt_box_paths, save_path_cropped_scans)
#crop_out_bbs(dict_gt_seg_files, dict_liver_gt_box_paths, save_path_cropped_seg, "liver")

# resample files to make them fit into the U-Net (64x64x64)
#resample_files(save_path_cropped_scans, save_path_X_train, 64, 64, 64)
#resample_files(save_path_cropped_seg, save_path_y_train, 64, 64, 64)


'''_____________________________________________________________________________________________'''
'''|..................................PREPARE TRAINING DATA....................................|'''
'''_____________________________________________________________________________________________'''

X_train = get_training_data(save_path_X_train, number_of_patients)
y_train = get_training_data(save_path_y_train, number_of_patients, "y")


'''_____________________________________________________________________________________________'''
'''|...................................TRAIN U-NET.............................................|'''
'''_____________________________________________________________________________________________'''

# generate the U-Net model
#architecture = generate_U_Net(64, 64, 64, 1)

# train U-Net on training data and save it
#model, history = train_U_Net(architecture, X_train, y_train, SAVE_PATH)
#plot_history(history)

'''_____________________________________________________________________________________________'''
'''|.................................PREPARE TEST DATA.........................................|'''
'''_____________________________________________________________________________________________'''

X_test = get_training_data(save_path_X_test, number_of_patients)
y_test = get_training_data(save_path_y_test, number_of_patients, "y")

'''_____________________________________________________________________________________________'''
'''|...................................APPLY U-NET.............................................|'''
'''_____________________________________________________________________________________________'''

# load U-Net
model = load_model("{}U-Net.h5".format(SAVE_PATH))

# apply U-Net on test data
results = model.predict(X_test, verbose=1)
print(len(results))
threshold = 0.3
for i in range(0, len(results)):
    print(i)
    result = results[i]
    # check voxel values against treshold and get segmentationmask
    pred_map = get_segmentation_mask(result, "liver", threshold)

    import nibabel as nib

    # save cropped array as nifti file with patient number in name
    daria = nib.load("{}{}.nii.gz".format(save_path_X_test, i))
    new_img = nib.Nifti1Image(pred_map, daria.affine, daria.header)
    nib.save(new_img, '{}seg{}.nii.gz'.format(save_path_results, i))


#result_img_arr = results[0] # get first result




# Generate generalization metrics
generate_metrics(model, X_test, y_test)


#TODO:
# bei UNET.py train wird momentan traniert wie bei Mietzner im Code
# beim Plotten ist vall_acc immer 1. Wieso?
# bei UNet model save wirklich notwendig da Model Checkpointer glaub schon das beste speichert?

'''
# save result #ERROR: Doesn't save the image spacing and stuff
import SimpleITK as sitk
result_img = sitk.GetImageFromArray(pred_map)
sitk.WriteImage(result_img, "{}{}".format(SAVE_PATH, "SITKresult.nii.gz"))

'''