# tests everything with one single patient and respective patient files

'''_____________________________________________________________________________________________'''
'''|.................................!!!!DEPRECATED!!!!........................................|'''
'''_____________________________________________________________________________________________'''

import numpy as np
import SimpleITK as sitk  # https://simpleitk.readthedocs.io/en/master/index.html
from Methods import crop_out_bb, resample_file, get_segmentation_mask
from UNet import generate_U_Net
from Methods import bb_mm_to_vox, get_bb_coordinates, nifti_image_affine_reader
import nibabel as nib
import numpy as np


# map result to whole ct scan
def handle_region(vox_int, result_arr, img_arr, label):
    pred_map = np.zeros((img_arr.shape[0],img_arr.shape[1], img_arr.shape[2]))
    for x in range(result_arr.shape[0]):
        for y in range(result_arr.shape[1]):
            for z in range(result_arr.shape[2]):
                if result_arr[x][y][z] > 0.3:
                    x_real = x + vox_int[0]
                    y_real = y + vox_int[2]
                    z_real = z + vox_int[4]

                    pred_map[x_real, y_real, z_real] = label
    return pred_map


def do_sth(result_arr):

    # paths
    box_path = '/home/daria/Desktop/PycharmProjects/UNet/Data/Train/Box/seg0.nii_170_bb.vtk'
    img_path = '/home/daria/Desktop/PycharmProjects/UNet/Data/Train/Img/0.nii.gz'
    save_path = '/home/daria/Desktop/PycharmProjects/UNet/Temp/'

    # load image
    img = nib.load(img_path)
    img_arr = img.get_fdata()

    # transform bb from mm to vox
    bb_coords = get_bb_coordinates(box_path)    # get bb coordinates
    spacing, offset = nifti_image_affine_reader(img)
    vox_170 = bb_mm_to_vox(bb_coords, spacing, offset)
    print('Coordinates of area in vox: ', vox_170)

    # make numpy int array
    vox_170_int = np.asarray(vox_170)
    vox_170_int = vox_170_int.astype(int)

    '''    # crop out region
    region_170 = img_arr.copy()
    region_170 = region_170[vox_170_int[0]:vox_170_int[1],
                 vox_170_int[2]:vox_170_int[3],
                 vox_170_int[4]:vox_170_int[5]]'''

    # segment bb region with pre-trained model
    pred_map_170 = handle_region(vox_170_int,
                                 result_arr,
                                 img_arr)


    new_img = nib.Nifti1Image(pred_map_170, img.affine, img.header)
    nib.save(new_img, '{}{}.nii.gz'.format(save_path, "hehhe"))

# GPU Use fix
bug_fix()

# define paths and load the training data
patient0_box_path = '/home/daria/Desktop/PycharmProjects/DATA PATIENT 0/BB/seg0.nii_170_bb.vtk'
patient0_gt_seg_path = '/home/daria/Desktop/PycharmProjects/DATA PATIENT 0/GT-SEG/seg0.nii.gz'
patient0_scan_path = '/home/daria/Desktop/PycharmProjects/DATA PATIENT 0/CT-Scans/0.nii.gz'
patient0_save_path = '/home/daria/Desktop/PycharmProjects/UNet/Temp/'

'''_____________________________________________________________________________________________'''
'''|.................................PREPROCESS DATA...........................................|'''
'''_____________________________________________________________________________________________'''

## load training data
# load image
img = nib.load(patient0_scan_path)    # image
# load segemntation
seg = nib.load(patient0_gt_seg_path)    # segmentation

## crop out 3D volume of CT-img and seg
img_arr_cropped = crop_out_bb(img, patient0_box_path)
seg_arr_cropped = crop_out_bb(seg, patient0_box_path)

## filter out relevant segmentation
liver_label = 170
seg_arr_cropped[seg_arr_cropped < liver_label] = 0
seg_arr_cropped[seg_arr_cropped > liver_label] = 0

## save cropped CT-img and seg
save_as_nifti_file(img_arr_cropped, img, patient0_save_path, 'croppedImg')
save_as_nifti_file(seg_arr_cropped, seg, patient0_save_path, 'croppedSeg')

## load cropped training data
sitk_img = sitk.ReadImage("{}{}".format(patient0_save_path, 'croppedImg.nii'))
sitk_seg = sitk.ReadImage("{}{}".format(patient0_save_path, 'croppedSeg.nii'))
#print(sitk.GetArrayFromImage(sitk_img).shape)

# resample cropped training data to fit into the U-Net (64x64x64x1 Dim)
img_resampled = resample_file(sitk_img, 64, 64, 64)
seg_resampled = resample_file(sitk_seg, 64, 64, 64)

# save resampled training data
sitk.WriteImage(img_resampled, "{}{}".format(patient0_save_path, 'resampledImg.nii'))
sitk.WriteImage(seg_resampled, "{}{}".format(patient0_save_path, 'resampledSeg.nii'))

'''_____________________________________________________________________________________________'''
'''|..................................PREPARE TRAINING DATA....................................|'''
'''_____________________________________________________________________________________________'''
# load training data images into X_train
no_of_samples = 1
img = sitk.ReadImage("{}{}".format(patient0_save_path, 'resampledImg.nii'))  # load
img_arr = sitk.GetArrayFromImage(img)   # convert to numpy array
img_arr = np.expand_dims(img_arr, axis=3)   # add a fourth dimension
X_train = np.zeros((no_of_samples, 64, 64, 64, 1), dtype=np.uint8)  # define X_train array
X_train[0] = img_arr    # add samples

#TODO
# load training data segementations into y_train
seg = sitk.ReadImage("{}{}".format(patient0_save_path, 'resampledSeg.nii'))  # load
print(seg)
seg_arr = sitk.GetArrayFromImage(seg)   # convert to numpy array
seg_arr = np.expand_dims(seg_arr, axis=3)   # add a fourth dimension
y_train = np.zeros((no_of_samples, 64, 64, 64, 1), dtype=np.bool)   # define y_train array
y_train[0] = seg_arr    # add samples

'''_____________________________________________________________________________________________'''
'''|...................................TRAIN U-NET.............................................|'''
'''_____________________________________________________________________________________________'''
# generate the U-Net model
architecture = generate_U_Net(64, 64, 64, 1)

# train U-Net on training data
model, history = train_U_Net(architecture, X_train, y_train)
plot_history(history)
make_model_img(model, "unet")

# save U-Net
save_U_Net(model)

'''_____________________________________________________________________________________________'''
'''|.................................PREPARE TEST DATA.........................................|'''
'''_____________________________________________________________________________________________'''
# load the test data
X_test = X_train  # TODO

'''_____________________________________________________________________________________________'''
'''|...................................APPLY U-NET.............................................|'''
'''_____________________________________________________________________________________________'''
# load U-Net
model = load_U_Net()

# apply U-Net on test data
result = model.predict(X_test, verbose=1)
result_img_arr = result[0]

# check voxel values against treshold and get segmentationmask
pred_map = get_segmentation_mask(result_img_arr, 0.3, liver_label)

# save result
result_img = sitk.GetImageFromArray(pred_map)
sitk.WriteImage(result_img, "{}{}".format(patient0_save_path, "result.nii"))

#do_sth(result_img_arr)

