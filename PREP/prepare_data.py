'''
Preprocessing
'''
import shutil
import SimpleITK as sitk
import SharedMethods as sm
import os
import vtk
from pathlib import Path


def rename_files(in_dir, out_dir):
    counter = 0
    for file in os.scandir(in_dir):
        shutil.copy(file.path, "{}{}".format(out_dir, "{}.nii.gz".format(counter)))
        counter += 1


def create_gt_bb(seg_path, bb_path):
    '''
    creates ground truth bounding boxes from segmentation files. Organ color coding must be 170(liver), 156(left kidney),
    157(right kidney), 160(spleen) and 150(pancreas). The segmentation names must be similar to "seg1.nii.gz".
    The output bounding box files will be named "1_170_bb.vtk" or similar.

        :param seg_path: path to directory containing the (ground truth) segmentation files
        :param bb_path: path to target bounding box directory

        Usage::
            seg_path = "/path to gt segmentation"
            bb_path = "/path to target directory"

            create_gt_bb(seg_path, bb_path)
        '''
    print("creating Ground Truth Bounding Boxes from segmentations in '{}'".format(seg_path))

    count = 0
    # get bb for every organ segmentation
    for file in os.scandir(seg_path):
        patient = sm.find_patient_no_in_file_name(file.name)

        # load original segmentation image
        orig_img = sitk.ReadImage("{}{}".format(seg_path, file.name))

        # for organ in [6, 3, 2, 1, 11]: # unused alternative for different color coding
        for organ in [150, 156, 157, 160, 170]:
            # get start index and size of organ
            lsi_filter = sitk.LabelShapeStatisticsImageFilter()
            lsi_filter.SetComputeOrientedBoundingBox(True)
            lsi_filter.Execute(orig_img)
            bb = lsi_filter.GetBoundingBox(organ)  # x1, y1, z1, w, h, d

            # other stuff
            bb_orig = lsi_filter.GetOrientedBoundingBoxOrigin(organ)
            bb_dir = lsi_filter.GetOrientedBoundingBoxDirection(organ)
            bb_vertices = lsi_filter.GetOrientedBoundingBoxVertices(organ)
            bb_size = lsi_filter.GetOrientedBoundingBoxSize(organ)

            # define for index slicing
            x_min = bb[0] - 1
            x_max = bb[0] + bb[3]
            y_min = bb[1] - 1
            y_max = bb[1] + bb[4]
            z_min = bb[2] - 1
            z_max = bb[2] + bb[5]

            # transform points to physical space
            p_min = orig_img.TransformIndexToPhysicalPoint((x_min, y_min, z_min))
            p_max = orig_img.TransformIndexToPhysicalPoint((x_max, y_max, z_max))

            '''
            NOTE: Nifti changes direction  ( 1,0,0   to    (-1,  0,  0
                                             0,1,0,          0, -1,  0      
                                             0,0,1 )         0,  0,  1

            that's why x and y have to be inverted when saving it to a VTK file
            '''
            bounds = [-p_max[0], -p_min[0], -p_max[1], -p_min[1], p_min[2], p_max[2]]

            # define bb as cube
            vtk_cube = vtk.vtkCubeSource()
            vtk_cube.SetBounds(bounds)
            vtk_cube.Update()
            output = vtk_cube.GetOutput()

            # switcher = { # unused alternative for different color coding
            #    6: 170,
            #    3: 156,
            #    2: 157,
            #    1: 160,
            #    10: 150
            # }
            # bb_name = "{}_{}_bb.vtk".format(patient, switcher.get(organ))

            # save bounding box object to file
            bb_name = "{}_{}_bb.vtk".format(patient, organ)
            save_path = "{}{}".format(bb_path, bb_name)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(output)
            writer.SetFileName(save_path)
            writer.Update()

    print("count {}".format(count))
    print("done. saved Ground Truth Bounding Boxes to '{}'".format(bb_path))


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
            path = "/path to file"

            for file in os.scandir(path):
                check(file)
        '''
    patient = sm.find_patient_no_in_file_name(file.name)
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


def copy_files_to_folder(in_dir, out_dir):
    # delete all files in output directory
    shutil.rmtree(out_dir, ignore_errors=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # move files
    for file in os.scandir(in_dir):
        shutil.copy(file.path, "{}{}".format(out_dir, file.name))

    # delete all files in input directory
    shutil.rmtree(in_dir, ignore_errors=True)
    Path(in_dir).mkdir(parents=True, exist_ok=True)


def prepare_data(in_dir, out_dir, temp_dir):
    # delete all files in output directory
    shutil.rmtree(out_dir, ignore_errors=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    rename_files(in_dir, out_dir)
    print("renamed files: ")
    check_all(out_dir)

    set_voxeltype(out_dir, temp_dir)
    copy_files_to_folder(temp_dir, out_dir)

    set_origin(out_dir, temp_dir)
    copy_files_to_folder(temp_dir, out_dir)

    set_direction(out_dir, temp_dir)
    copy_files_to_folder(temp_dir, out_dir)

    set_spacing(out_dir, temp_dir)
    copy_files_to_folder(temp_dir, out_dir)
    print("prepared files: ")
    check_all(out_dir)

