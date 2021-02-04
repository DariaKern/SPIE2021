'''
Preprocessing
'''

import SimpleITK as sitk
import os, re


def find_patient_no_in_file_name(file_name):
    '''
    finds the patient number (1 or more digits from 0-9) in a file name and returns it as integer

    :param file_name: name of the file

    Usage::
        path = "/path to files"

        for file in os.scandir(path):
            find_patient_no_in_file_name(file.name)
    '''
    regex = re.compile(r'\d+') # 1 or more digits (0-9)
    patient_no = int(regex.search(file_name).group(0)) #if not an integer 1 and 10 could cause problems

    return patient_no


def set_direction(in_dir, out_dir):
    '''
    changes the orientation to RAI using SITK.

    :param in_dir: path to directory containing images with wrong voxel type
    :param out_dir: path to target directory for result images

    Usage::
        in_dir = "/path to directory containing images with wrong orientation"

        out_dir = "/target path"

        set_direction(in_dir, out_dir)
    '''

    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))

        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_origin(in_dir, out_dir):
    '''
    changes the origin to 0 (x, y and z axis) using SITK.

    :param in_dir: path to directory containing images with wrong origin
    :param out_dir: path to target directory for result images

    Usage::
        in_dir = "/path to directory containing images with wrong origin"

        out_dir = "/target path"

        set_origin(in_dir, out_dir)
    '''
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetOrigin((0, 0, 0))
        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_spacing(in_dir, out_dir):
    '''
    changes the spacing between voxels to 2 (x, y and z axis) using SITK.

    :param in_dir: path to directory containing images with wrong spacing
    :param out_dir: path to target directory for result images

    Usage::
        in_dir = "/path to directory containing images with wrong spacing"

        out_dir = "/target path"

        set_spacing(in_dir, out_dir)
    '''
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetSpacing((2.0, 2.0, 2.0))

        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_voxeltype(in_dir, out_dir):
    '''
    changes the voxel type for segmentations to UInt16 and for CT Scans to Int16 using SITK.
    The segmentation names must be similar to "seg1.nii.gz" and the CT scan names must be similar to "1.nii.gz".

    :param in_dir: path to directory containing images with wrong voxel type
    :param out_dir: path to target directory for result images

    Usage::
        in_dir = "/path to directory containing images with wrong voxel type"

        out_dir = "/target path"

        set_voxeltype(in_dir, out_dir)
    '''
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))

        if "seg" in file.name:
            img = sitk.Cast(img, sitk.sitkUInt16)
            sitk.WriteImage(img, "{}{}".format(out_dir, file.name))
        else:
            img = sitk.Cast(img, sitk.sitkInt16)
            sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def change_segmentation_colorcode(organs, in_dir, out_dir):
    '''
    changes the color coding of the 5 target organs to 170(liver), 156(left kidney),
    157(right kidney), 160(spleen) and 150(pancreas)

    :param organs: list of 5 integer values
    :param in_dir: path to directory containing images with wrong color coding
    :param out_dir: path to target directory for result images

    Usage::
        in_dir = "/path to files with wrong color coding"

        out_dir = "/target path"

        organs = [6, 3, 2, 1, 11] # original color coding

        change_segmentation_colorcode(organs, in_dir, out_dir)
    '''
    switcher = {
        organs[0]: 170,     # liver
        organs[1]: 156,     # left kidney
        organs[2]: 157,     # right kidney
        organs[3]: 160,     # spleen
        organs[4]: 150,     # pancreas
    }
    for file in os.scandir(in_dir):
        print("{}".format(file.name))

        # load file and get array
        orig_img = sitk.ReadImage(file.path)
        orig_img_arr = sitk.GetArrayFromImage(orig_img)

        result_img_arr = orig_img_arr
        for colorcode in organs:

            # change colors
            result_img_arr[orig_img_arr == colorcode] = switcher.get(colorcode)
            #result_img = sitk.Mask(orig_img, sitk.Equal(organ_label, orig_img))  # procedural API of SimpleITK

        result_img = sitk.GetImageFromArray(result_img_arr)
        result_img.CopyInformation(orig_img)
        sitk.WriteImage(result_img, "{}{}".format(out_dir, file.name))


def check(file):
    '''
        prints patient number, origin, orientation, voxel spacing and voxel type of a file

        :param file: file to check

        Usage::
            path = "/path to files"

            for file in os.scandir(path):
                check(file)
        '''
    patient = find_patient_no_in_file_name(file.name)
    img = sitk.ReadImage("{}".format(file.path))
    o = img.GetOrigin()
    d = img.GetDirection()
    s = img.GetSpacing()
    t = img.GetPixelIDTypeAsString()
    print("patient #{}\n origin: {}\n direction: {}\n spacing: {}\n pixel: {}".format(patient, o, d, s, t))
    print("")


def check_all(path):
    '''
        prints patient number, origin, orientation, voxel spacing and voxel type of all files in a directory

        :param file: path to directory containing files to check

        Usage::
            path = "/path to files"

            check_all(path)
        '''
    for file in os.scandir(path):
        check(file)

