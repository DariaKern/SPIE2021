"""
HELPING METHODS
"""


'''_____________________________________________________________________________________________'''
'''|.................................Helping Methods...........................................|'''
'''_____________________________________________________________________________________________'''


# given the name of an organ it returns the organs label number
def get_organ_label(organ):

    # define dictionary (to simulate switch-case)
    switcher = {
        "liver": 170,
        "left_kidney": 156,
        "right_kidney": 157,
        "spleen": 160,
        "pancreas": 150
    }

    # if given organ isn't a defined key, return "no valid organ"
    organ_label = switcher.get(organ, "no valid organ")

    # raise error message if no valid organ name was given
    if organ_label == "no valid organ":
        raise ValueError("'{}' is no valid organ name. Valid names are: "
                         "'liver', 'left_kidney', 'right_kidney', 'spleen', "
                         "'pancreas'".format(organ))
    else:
        return organ_label


# TODO: when creating VTK files bb Coordinates switch y and y2
# read bounding box coordinates
def get_bb_coordinates(box_path):

    # open vtk file and get coordinates
    bb_file = open(box_path, 'r')
    lines = bb_file.readlines()

    # get coordinates
    numbers1 = lines[5].split()
    x = float(numbers1[0])
    y2 = float(numbers1[1])  # TODO
    z = float(numbers1[2])

    # get coordinates
    numbers2 = lines[11].split()
    x2 = float(numbers2[0])
    y = float(numbers2[1])  # TODO
    z2 = float(numbers2[2])

    # close file
    bb_file.close()

    # add coordinates to array
    bb_coords = [x, x2, y, y2, z, z2]

    return bb_coords


# transform coordinate list (x,y,z) from mm-space to voxelspace
# return new coordinate list
def mm_to_vox(coord_list, spacing, offset):

    # calculate coordinates from mm-space to voxelspace
    x_vox = (coord_list[0] - offset[0]) / spacing[0]
    y_vox = (coord_list[1] - offset[1]) / spacing[1]
    z_vox = (coord_list[2] - offset[2]) / spacing[2]
    coord_vox = [x_vox, y_vox, z_vox]

    return coord_vox


# transform overlap of bounding box and image/segmentation to vox coordinates
def bb_mm_to_vox(bb_coords, spacing, offset):

    # split coordinates into min and max coordinate values of bounding box
    bb_coords_min_mm = [bb_coords[0], bb_coords[2], bb_coords[4]]  # x1, y1, z1
    bb_coords_max_mm = [bb_coords[1], bb_coords[3], bb_coords[5]]  # x2, y2, z2

    # transform to vox coordinates
    bb_coords_min_vox = mm_to_vox(bb_coords_min_mm, spacing, offset)
    bb_coords_max_vox = mm_to_vox(bb_coords_max_mm, spacing, offset)

    # merge min and max coordinates again
    bb_coords_vox = []
    bb_coords_vox.append(bb_coords_min_vox[0])
    bb_coords_vox.append(bb_coords_max_vox[0])
    bb_coords_vox.append(bb_coords_min_vox[1])
    bb_coords_vox.append(bb_coords_max_vox[1])
    bb_coords_vox.append(bb_coords_min_vox[2])
    bb_coords_vox.append(bb_coords_max_vox[2])

    # if negative x spacing, switch x1 and x2
    if spacing[0] < 0:
        temp_space_x = bb_coords_vox[0]
        bb_coords_vox[0] = bb_coords_vox[1]
        bb_coords_vox[1] = temp_space_x

    # if negative y spacing, switch y1 and y2
    if spacing[1] < 0:
        temp_space_y = bb_coords_vox[2]
        bb_coords_vox[2] = bb_coords_vox[3]
        bb_coords_vox[3] = temp_space_y

    # if negative z spacing, switch z1 and z2
    if spacing[2] < 0:
        temp_space_z = bb_coords_vox[4]
        bb_coords_vox[4] = bb_coords_vox[5]
        bb_coords_vox[5] = temp_space_z

    return bb_coords_vox


# get image affine from header
# for coordinate system handling
# return spacing and offset
def nifti_image_affine_reader(img):

    # read spacing
    spacing_x = img.affine[0][0]
    spacing_y = img.affine[1][1]
    spacing_z = img.affine[2][2]
    spacing = [spacing_x, spacing_y, spacing_z]

    # read offset
    offset_x = img.affine[0][3]
    offset_y = img.affine[1][3]
    offset_z = img.affine[2][3]
    offset = [offset_x, offset_y, offset_z]

    return spacing, offset
