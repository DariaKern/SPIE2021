
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

