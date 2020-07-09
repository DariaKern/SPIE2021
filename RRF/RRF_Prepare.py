import os
import numpy as np
import nibabel as nib
from RRF_SharedMethods import find_patient_no_in_file_name


# search non-zero values in array
def find_organ_min_max_bounds(organ, img_arr):
        # Test whether any array element along a given axis evaluates to True.
        x = np.any(img_arr==organ, axis=(1, 2))
        y = np.any(img_arr==organ, axis=(0, 2))
        z = np.any(img_arr==organ, axis=(0, 1))

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return xmin, xmax, ymin, ymax, zmin, zmax


def write_into_box_file(file_path, x1, x2, y1, y2, z1, z2):
    boxfile = open(file_path, 'w')

    boxfile.write('# vtk DataFile Version 2.0 \n')
    boxfile.write('Cube example \n')
    boxfile.write('ASCII \n')
    boxfile.write('DATASET POLYDATA \n')
    boxfile.write('POINTS 8 float \n')
    boxfile.write(str(x1))
    boxfile.write(' ')
    boxfile.write(str(y2))
    boxfile.write(' ')
    boxfile.write(str(z1))
    boxfile.write('\n')
    boxfile.write(str(x1))
    boxfile.write(' ')
    boxfile.write(str(y1))
    boxfile.write(' ')
    boxfile.write(str(z1))
    boxfile.write('\n')
    boxfile.write(str(x2))
    boxfile.write(' ')
    boxfile.write(str(y1))
    boxfile.write(' ')
    boxfile.write(str(z1))
    boxfile.write('\n')
    boxfile.write(str(x2))
    boxfile.write(' ')
    boxfile.write(str(y2))
    boxfile.write(' ')
    boxfile.write(str(z1))
    boxfile.write('\n')
    boxfile.write(str(x1))
    boxfile.write(' ')
    boxfile.write(str(y2))
    boxfile.write(' ')
    boxfile.write(str(z2))
    boxfile.write('\n')
    boxfile.write(str(x1))
    boxfile.write(' ')
    boxfile.write(str(y1))
    boxfile.write(' ')
    boxfile.write(str(z2))
    boxfile.write('\n')
    boxfile.write(str(x2))
    boxfile.write(' ')
    boxfile.write(str(y1))
    boxfile.write(' ')
    boxfile.write(str(z2))
    boxfile.write('\n')
    boxfile.write(str(x2))
    boxfile.write(' ')
    boxfile.write(str(y2))
    boxfile.write(' ')
    boxfile.write(str(z2))
    boxfile.write('\n')
    boxfile.write('POLYGONS 6 30 \n')
    boxfile.write('4 0 1 2 3 \n')
    boxfile.write('4 4 5 6 7 \n')
    boxfile.write('4 0 1 5 4 \n')
    boxfile.write('4 2 3 7 6 \n')
    boxfile.write('4 0 4 7 3 \n')
    boxfile.write('4 1 2 6 5 \n')
    boxfile.write('CELL_DATA 6 \n')
    boxfile.write('SCALARS cell_scalars int 1 \n')
    boxfile.write('LOOKUP_TABLE default \n')
    boxfile.write('0 \n')
    boxfile.write('1 \n')
    boxfile.write('2 \n')
    boxfile.write('3 \n')
    boxfile.write('4 \n')
    boxfile.write('5 \n')
    boxfile.write('NORMALS cell_normals float \n')
    boxfile.write('0 0 -1 \n')
    boxfile.write('0 0 1 \n')
    boxfile.write('0 -1 0 \n')
    boxfile.write('0 1 0 \n')
    boxfile.write('-1 0 0 \n')
    boxfile.write('1 0 0 \n')
    boxfile.write('FIELD FieldData 2 \n')
    boxfile.write('cellIds 1 6 int \n')
    boxfile.write('0 1 2 3 4 5 \n')
    boxfile.write('faceAttributes 2 6 float \n')
    boxfile.write('0.0 1.0 1.0 2.0 2.0 3.0 3.0 4.0 4.0 5.0 5.0 6.0 \n')
    boxfile.write('POINT_DATA 8 \n')
    boxfile.write('SCALARS sample_scalars float 1 \n')
    boxfile.write('LOOKUP_TABLE my_table \n')
    boxfile.write('0.0 \n')
    boxfile.write('1.0 \n')
    boxfile.write('2.0 \n')
    boxfile.write('3.0 \n')
    boxfile.write('4.0 \n')
    boxfile.write('5.0 \n')
    boxfile.write('6.0 \n')
    boxfile.write('7.0 \n')
    boxfile.write('LOOKUP_TABLE my_table 8 \n')
    boxfile.write('0.0 0.0 0.0 1.0 \n')
    boxfile.write('1.0 0.0 0.0 1.0 \n')
    boxfile.write('0.0 1.0 0.0 1.0 \n')
    boxfile.write('1.0 1.0 0.0 1.0 \n')
    boxfile.write('0.0 0.0 1.0 1.0 \n')
    boxfile.write('1.0 0.0 1.0 1.0 \n')
    boxfile.write('0.0 1.0 1.0 1.0 \n')
    boxfile.write('1.0 1.0 1.0 1.0 \n')

    boxfile.close()


def create_GT_BB(GT_SEG_PATH, GT_BB_PATH):
    organs = [150, 156, 157, 160, 170]

    for file in os.scandir(GT_SEG_PATH):
        # load Nifti format and save image in numpy array
        img = nib.load(file)
        img_arr = img.get_data()
        img_hdr = img.header

        # get affine from header
        spacing_x = img.affine[0][0]
        spacing_y = img.affine[1][1]
        spacing_z = img.affine[2][2]
        qoffset_x = img.affine[0][3]
        qoffset_y = img.affine[1][3]
        qoffset_z = img.affine[2][3]

        patient_number = find_patient_no_in_file_name(file.name)

        # for every organ (liver, kidneys, pancreas, spleen)
        for organ in organs:
            x1, x2, y1, y2, z1, z2 = find_organ_min_max_bounds(organ, img_arr)

            x1 = (x1 * spacing_x) + qoffset_x
            x2 = (x2 * spacing_x) + qoffset_x
            y1 = (y1 * spacing_y) + qoffset_y
            y2 = (y2 * spacing_y) + qoffset_y
            z1 = (z1 * spacing_z) + qoffset_z
            z2 = (z2 * spacing_z) + qoffset_z

            # if negative spacing switch max and min
            if spacing_x < 0:
                tempx = x1
                x1 = x2
                x2 = tempx

            if spacing_y < 0:
                tempy = y1
                y1 = y2
                y2 = tempy

            if spacing_z < 0:
                tempz = z1
                z1 = z2
                z2 = tempz

            file_name = "%s%s%s%s" % (patient_number, "_", organ, "_bb.vtk")
            file_path = "{}{}".format(GT_BB_PATH, file_name)
            write_into_box_file(file_path, x1, x2, y1, y2, z1, z2)
