'''


'''
import SimpleITK as sitk
import os, re


def find_patient_no_in_file_name(file_name):
    regex = re.compile(r'\d+')
    patient_no = int(regex.search(file_name).group(0))

    return patient_no


def set_direction(in_dir, out_dir):
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetDirection((1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0))

        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_origin(in_dir, out_dir):
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetOrigin((0, 0, 0))
        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_spacing(in_dir, out_dir):
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))
        img.SetSpacing((2.0, 2.0, 2.0))

        sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def set_voxeltype(in_dir, out_dir):
    for file in os.scandir(in_dir):
        img = sitk.ReadImage("{}{}".format(in_dir, file.name))

        if "seg" in file.name:
            img = sitk.Cast(img, sitk.sitkUInt16)
            sitk.WriteImage(img, "{}{}".format(out_dir, file.name))
        else:
            img = sitk.Cast(img, sitk.sitkInt16)
            sitk.WriteImage(img, "{}{}".format(out_dir, file.name))


def change_segmentation_colorcode(organs, folder_path, target_folder_path):
    '''
    changes the color coding of the 5 target organs to 170(liver), 156(left kidney),
    157(right kidney), 160(spleen) and 150(pancreas)

    :param organs: list of 5 integer values
    :param folder_path: path to folder containing images with wrong color coding
    :param target_folder_path: path to target folder for result images

    Usage::
        in_path = "/path to files with wrong color coding"

        out_path = "/target path"

        organs = [6, 3, 2, 1, 11] # original color coding

        change_segmentation_colorcode(organs, in_path, out_path)
    '''
    switcher = {
        organs[0]: 170,     # liver
        organs[1]: 156,     # left kidney
        organs[2]: 157,     # right kidney
        organs[3]: 160,     # spleen
        organs[4]: 150,     # pancreas
    }
    for file in os.scandir(folder_path):
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
        sitk.WriteImage(result_img, "{}{}".format(target_folder_path, file.name))


def check(file):

    patient = find_patient_no_in_file_name(file.name)
    img = sitk.ReadImage("{}".format(file.path))
    o = img.GetOrigin()
    d = img.GetDirection()
    s = img.GetSpacing()
    t = img.GetPixelIDTypeAsString()
    print("patient #{}\n origin: {}\n direction: {}\n spacing: {}\n pixel: {}".format(patient, o, d, s, t))
    print("")


def check_all(path):
    for file in os.scandir(path):
        check(file)

