import SimpleITK as sitk
import vtk
import os
from SharedMethods import get_dict_of_paths, find_patient_no_in_file_name


#TODO
SCAN_PATH = "/Data/Daria/DATA/CT-SCANS/"
GT_SEG_PATH = "/Data/Daria/DATA/GT-SEG/"
GT_BB_PATH = "/Data/Daria/new bb/"

scan_paths = get_dict_of_paths(SCAN_PATH)

for file in os.scandir(GT_SEG_PATH):
    patient_number = find_patient_no_in_file_name(file.name)
    scan_path = scan_paths[patient_number]
    img_scan = sitk.ReadImage(scan_path)
    img_seg = sitk.ReadImage(file.path)

    worldOrig = img_seg.GetOrigin()
    worldSpacg = img_seg.GetSpacing()

    for organ in [150, 156, 157, 160, 170]:
        lsi_filter = sitk.LabelStatisticsImageFilter()

        lsi_filter.Execute(img_scan, img_seg)
        v = lsi_filter.GetBoundingBox(organ)

        # convert image coordinates to world
        worldV = [((worldOrig[0] + v[0]) * worldSpacg[0])-1,
                  ((worldOrig[0] + v[1]) * worldSpacg[0])+1,
                  ((worldOrig[1] + v[2]) * worldSpacg[1])-1,
                  ((worldOrig[1] + v[3]) * worldSpacg[1])+1,
                  ((worldOrig[2] + v[4]) * worldSpacg[2])-1,
                  ((worldOrig[2] + v[5]) * worldSpacg[2])+1]

        if(patient_number == 7):
            print(worldV)
        bb = vtk.vtkCubeSource()
        bb.SetBounds(worldV)
        bb.Update()

        #print(bb)

        bb_name = "{}_{}_bb.vtk".format(patient_number, organ)
        save_path = "{}{}".format(GT_BB_PATH, bb_name)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(bb.GetOutput())
        writer.SetFileName(save_path)
        writer.Update()
