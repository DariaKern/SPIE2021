import SimpleITK as sitk
import vtk
import os
import SharedMethods as sm

OUT_DIR = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/RAI DIRECTION/'
OUT_ORIG = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/ZERO ORIGIN/'
OUT_SPAC = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/TARGET SPACING/'
FLIPPED = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/FLIPPED/'
OUT_VOXELID = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/VOXEL ID/'
OUT_SIZE = ''
INPUT_DIR = '/home/daria/Desktop/Data/Daria/Data old (Mietzner stuff)/GT-SEG/'
dar = '/home/daria/Desktop/Data/Daria/NORMALIZED DATA/step3 voxelspacing 222/CT-SCANS/'
def set_direction(input, output):
    for file in os.scandir(input):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(input, file.name))
        print(patient)
        print(orig_img.GetDirection())

        orig_img.SetDirection((1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0))

        print(orig_img.GetDirection())
        print("")

        sitk.WriteImage(orig_img, "{}{}".format(output, file.name))


def set_origin(input, output):
    for file in os.scandir(input):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(input, file.name))
        print(patient)
        print(orig_img.GetOrigin())
        orig_img.SetOrigin((0, 0, 0))
        print(orig_img.GetOrigin())
        sitk.WriteImage(orig_img, "{}{}".format(output, file.name))


def set_spacing(input, output):
    for file in os.scandir(input):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(input, file.name))
        print(patient)
        print(orig_img.GetSpacing())
        orig_img.SetSpacing((2.0, 2.0, 2.0))
        print(orig_img.GetSpacing())
        print("")
        sitk.WriteImage(orig_img, "{}{}".format(output, file.name))


def check(input):
    for file in os.scandir(input):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(input, file.name))
        print(patient)
        print(orig_img.GetOrigin())
        print(orig_img.GetDirection())
        print(orig_img.GetSpacing())
        #print(orig_img.GetDimension())
        #print(orig_img.GetSize())
        #print("({}, {}, {})".format(orig_img.GetWidth(), orig_img.GetHeight(), orig_img.GetDepth()))
        print(orig_img.GetPixelIDTypeAsString())
        #print(orig_img.GetPixelID())
        print("")


#set_direction(INPUT_DIR, OUT_DIR)
#set_origin(OUT_DIR, OUT_ORIG)
#set_spacing(FLIPPED, OUT_SPAC)
check(dar)


