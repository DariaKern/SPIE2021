import SimpleITK as sitk
import vtk
import os
import SharedMethods as sm


def create_gt_bb(GT_SEG_PATH, GT_BB_PATH):
    print("creating Ground Truth Bounding Boxes from segmentations in '{}'".format(GT_SEG_PATH))

    count = 0
    # get bb for every organ segmentation
    for file in os.scandir(GT_SEG_PATH):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(GT_SEG_PATH, file.name))

        for organ in [150, 156, 157, 160, 170]:
            # get start index and size of organ
            lsi_filter = sitk.LabelShapeStatisticsImageFilter()
            lsi_filter.SetComputeOrientedBoundingBox(True)
            lsi_filter.Execute(orig_img)
            bb = lsi_filter.GetBoundingBox(organ)  # x1, y1, z1, w, h, d
            bb_orig = lsi_filter.GetOrientedBoundingBoxOrigin(organ)
            bb_dir = lsi_filter.GetOrientedBoundingBoxDirection(organ)
            bb_vertices = lsi_filter.GetOrientedBoundingBoxVertices(organ)
            bb_size = lsi_filter.GetOrientedBoundingBoxSize(organ)

            # define for index slicing
            x_min = bb[0]-1
            x_max = bb[0] + bb[3]
            y_min = bb[1]-1
            y_max = bb[1] + bb[4]
            z_min = bb[2]-1
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

            # save bounding box object to file
            bb_name = "{}_{}_bb.vtk".format(patient, organ)
            save_path = "{}{}".format(GT_BB_PATH, bb_name)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(output)
            writer.SetFileName(save_path)
            writer.Update()

    print("count {}".format(count))
    print("done. saved Ground Truth Bounding Boxes to '{}'".format(GT_BB_PATH))


def create_gt_bb_alternative(GT_SEG_PATH, GT_BB_PATH):
    print("creating Ground Truth Bounding Boxes from segmentations in '{}'".format(GT_SEG_PATH))

    count = 0
    # get bb for every organ segmentation
    for file in os.scandir(GT_SEG_PATH):
        patient = sm.find_patient_no_in_file_name(file.name)
        # load original image
        orig_img = sitk.ReadImage("{}{}".format(GT_SEG_PATH, file.name))

        for organ in [6, 3, 2, 1, 10]:
            # get start index and size of organ
            lsi_filter = sitk.LabelShapeStatisticsImageFilter()
            lsi_filter.SetComputeOrientedBoundingBox(True)
            lsi_filter.Execute(orig_img)
            bb = lsi_filter.GetBoundingBox(organ)  # x1, y1, z1, w, h, d
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

            # save bounding box object to file
            switcher = {
                6: 170,
                3: 156,
                2: 157,
                1: 160,
                10: 150
            }
            bb_name = "{}_{}_bb.vtk".format(patient, switcher.get(organ))
            save_path = "{}{}".format(GT_BB_PATH, bb_name)
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(output)
            writer.SetFileName(save_path)
            writer.Update()

    print("count {}".format(count))
    print("done. saved Ground Truth Bounding Boxes to '{}'".format(GT_BB_PATH))


def prepare(GT_SEG_PATH, GT_BB_PATH):
    create_gt_bb(GT_SEG_PATH, GT_BB_PATH)