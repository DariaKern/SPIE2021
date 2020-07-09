import argparse
import textwrap
import numpy as np
import nibabel as nib
import os, re


def find_patient_no_in_file_name(file_name):
    # find patient number in file name
    regex = re.compile(r'\d+')
    patient_no = int(regex.search(file_name).group(0))

    return patient_no

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''\
                    Generate Bounding Boxes from Segmentation Data
                    ----------------------------------------------
                    This script is used to find the bounding boxes of specfic abdominal organs in CT segmentation data.
                    
                    Preparation:
                        The CT data should be saved in the NIfTI format (.nii or .nii.gz).
                        The segmentation data has to be saved in its own directory.
                        It is essential, to assign the standard IMI organ-values to the segmented data.
                        Organ standard values:
                            pancreas: 150
                            kidney (left): 156
                            kidney (right): 157
                            spleen: 160
                            liver: 170
                        Make sure that the segmentation data has a meaningful name. Preferably the name is similar to
                        the name of the original CT data. 
                        Example:
                            Original data: datasetname_1.nii
                            Segmentation data: datasetname_1_seg.nii
                    
                    Execution:
                        This script has one input argument. The segmentation_path is the path that points to the
                        directory containing the segmentation data. 
                        By default the bounding boxes of all 5 organs will be found. If you wish to search for specific 
                        organs, you can add optional arguments. The short and long versions of these arguments are
                        listed below.
                        The bounding boxes will be saved as .vtk objects in the same directory as the segmentation data. 
                        The name of the bounding boxes, will be generated as: segmentationname_organ.vtk
                    '''))
parser.add_argument("segmentation_path", help = "find bounding boxes for all segmentations in given path")
parser.add_argument("-p", "--pancreas", help="create Bounding Box for pancreas", action="store_true")
parser.add_argument("-lk", "--leftkidney", help="create Bounding Box for left kidney", action="store_true")
parser.add_argument("-rk", "--rightkidney", help="create Bounding Box for right kidney", action="store_true")
parser.add_argument("-s", "--spleen", help="create Bounding Box for spleen", action="store_true")
parser.add_argument("-l", "--liver", help="create Bounding Box for liver", action="store_true")
args = parser.parse_args()
print(args.segmentation_path)


input1 = args.segmentation_path
print(input1)

files = []

for entry in os.scandir(input1):
    if entry.is_file():
        files.append(entry.path)

print(files)

if args.pancreas or args.leftkidney or args.rightkidney or args.spleen or args.liver:
    organs = []
    if args.pancreas:
        organs.append(150)
    if args.leftkidney:
        organs.append(156)
    if args.rightkidney:
        organs.append(157)
    if args.spleen:
        organs.append(160)
    if args.liver:
        organs.append(170)
else:
    organs = [150, 156, 157, 160, 170]

print(organs)

for file in files:

    print(file)

    for organ in organs:

        # load Nifti format
        img = nib.load(file)

        # save image in numpy array
        data = img.get_data()

        print(data.shape)
        print(organ)

        hdr = img.header

        # print(hdr)
        filePrefix = os.path.splitext(file)[0]
        patient_number = find_patient_no_in_file_name(filePrefix)
        input2 = "%s%s%s%s"%(patient_number, "_", organ, "_bb.vtk")

        # first method
        # set everything to zero except the organ
        data[data > organ] = 0
        data[data < organ] = 0
        print(data.shape)

        # check if organ is in segmentation data
        if organ in data:
            ocheck = 1
        else:
            print('Organ', organ, 'not found')
            continue

        # search non-zero values in array
        def bbox2_3D(data):

            x = np.any(data, axis=(1, 2))
            y = np.any(data, axis=(0, 2))
            z = np.any(data, axis=(0, 1))

            xmin, xmax = np.where(x)[0][[0, -1]]
            ymin, ymax = np.where(y)[0][[0, -1]]
            zmin, zmax = np.where(z)[0][[0, -1]]

            return xmin, xmax, ymin, ymax, zmin, zmax


        result = bbox2_3D(data)

        print(result)

        # Ergebnisse von newbb
        y = result[2]
        x = result[0]
        y2 = result[3]
        x2 = result[1]
        zmin = result[4]
        zmax = result[5]

        # get affine from header
        spacing_x = img.affine[0][0]
        spacing_y = img.affine[1][1]
        spacing_z = img.affine[2][2]
        qoffset_x = img.affine[0][3]
        qoffset_y = img.affine[1][3]
        qoffset_z = img.affine[2][3]
        print('spacing_x = ', spacing_x)
        print('spacing_y = ', spacing_y)
        print('spacing_z = ', spacing_z)
        print('offset_x = ', qoffset_x)
        print('offset_y = ', qoffset_y)
        print('offset_z = ', qoffset_z)


        x = x * spacing_x
        y = y * spacing_y
        x2 = x2 * spacing_x
        y2 = y2 * spacing_y
        zmin = zmin * spacing_z
        zmax = zmax * spacing_z

        print('Erster Schritt')
        print('x = ', x, 'y = ', y, 'x2 = ', x2, 'y2 = ', y2, 'zmin = ', zmin, 'zmax = ', zmax)

        x = x + qoffset_x
        y = y + qoffset_y
        x2 = x2 + qoffset_x
        y2 = y2 + qoffset_y
        zmin = zmin + qoffset_z
        zmax = zmax + qoffset_z

        print('Zweiter Schritt')
        print('x = ', x, 'y = ', y, 'x2 = ', x2, 'y2 = ', y2, 'zmin = ', zmin, 'zmax = ', zmax)

        if spacing_x < 0:
            tempx = x
            x = x2
            x2 = tempx

        if spacing_y < 0:
            tempy = y
            y = y2
            y2 = tempy

        print('x = ', x, 'y = ', y, 'x2 = ', x2, 'y2 = ', y2, 'zmin = ', zmin, 'zmax = ', zmax)

        boxfile = open(input2, 'w')

        boxfile.write('# vtk DataFile Version 2.0 \n')
        boxfile.write('Cube example \n')
        boxfile.write('ASCII \n')
        boxfile.write('DATASET POLYDATA \n')
        boxfile.write('POINTS 8 float \n')
        boxfile.write(str(x))
        boxfile.write(' ')
        boxfile.write(str(y2))
        boxfile.write(' ')
        boxfile.write(str(zmin))
        boxfile.write('\n')
        boxfile.write(str(x))
        boxfile.write(' ')
        boxfile.write(str(y))
        boxfile.write(' ')
        boxfile.write(str(zmin))
        boxfile.write('\n')
        boxfile.write(str(x2))
        boxfile.write(' ')
        boxfile.write(str(y))
        boxfile.write(' ')
        boxfile.write(str(zmin))
        boxfile.write('\n')
        boxfile.write(str(x2))
        boxfile.write(' ')
        boxfile.write(str(y2))
        boxfile.write(' ')
        boxfile.write(str(zmin))
        boxfile.write('\n')
        boxfile.write(str(x))
        boxfile.write(' ')
        boxfile.write(str(y2))
        boxfile.write(' ')
        boxfile.write(str(zmax))
        boxfile.write('\n')
        boxfile.write(str(x))
        boxfile.write(' ')
        boxfile.write(str(y))
        boxfile.write(' ')
        boxfile.write(str(zmax))
        boxfile.write('\n')
        boxfile.write(str(x2))
        boxfile.write(' ')
        boxfile.write(str(y))
        boxfile.write(' ')
        boxfile.write(str(zmax))
        boxfile.write('\n')
        boxfile.write(str(x2))
        boxfile.write(' ')
        boxfile.write(str(y2))
        boxfile.write(' ')
        boxfile.write(str(zmax))
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