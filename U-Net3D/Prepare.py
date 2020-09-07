
from SharedMethods import create_paths, get_organ_label, find_patient_no_in_file_name, \
    get_dict_of_paths, resample_file, get_bb_coordinates
from openpyxl.styles import Alignment, NamedStyle, Font
from openpyxl import Workbook
import os, random, shutil
import SimpleITK as sitk


# takes the numbers in the names of the patient files in SCAN_PATH
# and splits all patient numbers into 2 random subsets of a certain size defined by SPLIT
def split_train_and_test(SCAN_PATH, SPLIT, CUSTOM_TEST_SET=None):
    # count how many files are in SCAN_PATH
    all_patient_numbers = set()
    for file in os.scandir(SCAN_PATH):
        patient_no = find_patient_no_in_file_name(file.name)
        all_patient_numbers.add(patient_no)
    amount_patients = len(all_patient_numbers)

    # determine size of test split and split test patient numbers
    # if no custom test set is given, random patient numbers will be picked
    if CUSTOM_TEST_SET is None:
        test_split_size = int(amount_patients * SPLIT)
        test_split = set(random.sample(all_patient_numbers, test_split_size))
    else:
        test_split_size = len(CUSTOM_TEST_SET)
        test_split = set(CUSTOM_TEST_SET)

    # determine size of train split and split train patient numbers
    train_split = all_patient_numbers - test_split
    train_split_size = int(amount_patients - test_split_size)
    print("splitting {} files into {} TRAIN and {} TEST files".format(amount_patients, train_split_size, test_split_size))

    return test_split, train_split


def create_excel_sheet(SAVE_PATH, ORGAN, test_split, train_split):
    print("creating excel sheet for {} in {}".format(ORGAN, SAVE_PATH))

    # create excel sheet
    wb = Workbook()
    sheet = wb.active
    sheet.title = ORGAN

    # write down test split and train split
    test_split_arr = []
    test_split_arr.append("test split patient#")
    for patient_no in test_split:
        test_split_arr.append(patient_no)
    sheet.append(test_split_arr)

    train_split_arr = []
    train_split_arr.append("train split patient#")
    for patient_no in train_split:
        train_split_arr.append(patient_no)
    sheet.append(train_split_arr)

    # create headings and apply style
    headings_style = NamedStyle(
        name="daria",
        font=Font(color='000000', bold=True),
        alignment=Alignment(horizontal='left')
    )
    headings_row = '3'
    headings = ["patient #", "hausdorff dist",
                "Ø hausdorff dist",
                "dice coeff","-"]
    sheet.append(headings)
    for cell in sheet[headings_row]:
        cell.style = headings_style

    # make cells wider
    column_width = 20
    sheet.column_dimensions['A'].width = column_width
    sheet.column_dimensions['B'].width = column_width
    sheet.column_dimensions['C'].width = column_width
    sheet.column_dimensions['D'].width = column_width
    sheet.column_dimensions['E'].width = column_width
    sheet.column_dimensions['F'].width = column_width

    wb.save("{}Evaluation {}.xlsx".format(SAVE_PATH, ORGAN))


def copy_files_to_folder(folder_path, target_folder_path, patient_number_set):
    for file in os.scandir(folder_path):
        patient_no = find_patient_no_in_file_name(file.name)
        if patient_no in patient_number_set:
            shutil.copy(file.path, "{}{}".format(target_folder_path, file.name))


def filter_out_relevant_segmentations(folder_path, target_folder_path, ORGAN):
    print("filtering out relevant segmentations in {}".format(folder_path))
    organ_label = get_organ_label(ORGAN)

    for file in os.scandir(folder_path):
        # load file and filter out segmentation
        orig_img = sitk.ReadImage(file.path)
        result_img = sitk.Mask(orig_img, sitk.Equal(organ_label, orig_img))  # procedural API of SimpleITK
        sitk.WriteImage(result_img, "{}{}".format(target_folder_path, file.name))

    print("done. saved filtered files to '{}'".format(target_folder_path))


# crops out the bounding box volume of the given CT-image or segmentation
def crop_out_bb(orig_img, box_path):
    x_min, x_max, y_min, y_max, z_min, z_max = get_bb_coordinates(box_path)

    # transform physical space to index points
    p_min = orig_img.TransformPhysicalPointToIndex((x_min, y_min, z_min))
    p_max = orig_img.TransformPhysicalPointToIndex((x_max, y_max, z_max))

    # x and y are negative because of nifti file orientation -x, -y, x
    # -> swap min and max and multiply by -1 to make positive
    # alson substract 1 from all max...only god knows why
    x_min = (p_max[0]-1)*-1
    x_max = (p_min[0])*-1
    y_min = (p_max[1]-1)*-1
    y_max = (p_min[1])*-1
    z_min = p_min[2]
    z_max = p_max[2]-1

    # cut out ROI with slicing operator
    result_img = orig_img[x_min:x_max, y_min:y_max, z_min:z_max]

    return result_img


# crops out the areas of interest (where to organ is supposed to be) defined by the given bounding boxes
def crop_out_bbs(folder_path, bb_folder_path, target_folder_path, patient_number_set, organ=None, isSegmentation=False):
    # organize bbs in bb_folder_path by patient number
    bb_path_dict = get_dict_of_paths(bb_folder_path, organ)

    print("cropping out bounding boxes (volume of interest)")
    # go through all files and if they are in patient_number_set, crop out the bb
    for file in os.scandir(folder_path):
        patient_number = find_patient_no_in_file_name(file.name)

        if patient_number in patient_number_set:
            # access relevant patient files
            bb_path = bb_path_dict[patient_number]
            orig_img = sitk.ReadImage(file.path)

            # crop out box area
            result_img = crop_out_bb(orig_img, bb_path)


            # extract segmentation of given organ (only necessary for segmentations)
            # (filter out overlapping segmentations of other organs)
            if isSegmentation:
                organ_label = get_organ_label(organ)
                result_img = sitk.Mask(result_img, sitk.Equal(organ_label, result_img))  # procedural API of SimpleITK

            # save cropped file with patient number in name
            sitk.WriteImage(result_img, "{}{}".format(target_folder_path, file.name))

    print("done. saved cropped files to '{}'".format(target_folder_path))


# resamples all files in a folder to a given size and saves it to the given path
def resample_files(path, target_path, target_depth, target_height, target_width):
    print("resampling files in '{}'".format(path))
    for file in os.scandir(path):
        orig_img = sitk.ReadImage(file.path)
        result_img = resample_file(orig_img, target_depth, target_height, target_width)
        sitk.WriteImage(result_img, "{}{}".format(target_path, file.name))

    print("done. saved resampled files to '{}'".format(target_path))


def create_x_train(SCAN_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN):
    print("")
    print("CREATING X TRAIN")
    path_x_train, path_x_train_cropped, path_x_train_resampled, path_x_train_orig = create_paths(SAVE_PATH, "Xtrain")
    copy_files_to_folder(SCAN_PATH, path_x_train_orig, train_split)

    # crop GT BBs out of SCANs
    crop_out_bbs(SCAN_PATH, GT_BB_PATH, path_x_train_cropped, train_split, ORGAN)

    # resample cropped out area
    resample_files(path_x_train_cropped, path_x_train_resampled, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2])


def create_y_train(GT_SEG_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN):
    print("")
    print("CREATING Y TRAIN")
    path_y_train, path_y_train_cropped, path_y_train_resampled, path_y_train_orig = create_paths(SAVE_PATH, "ytrain")
    copy_files_to_folder(GT_SEG_PATH, path_y_train_orig, train_split)

    # crop GT BBs out of GT SEGs
    crop_out_bbs(GT_SEG_PATH, GT_BB_PATH, path_y_train_cropped, train_split, ORGAN, isSegmentation=True)

    # resample cropped out area
    resample_files(path_y_train_cropped, path_y_train_resampled, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2])


def create_x_test(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN):
    print("")
    print("CREATING X TEST")
    path_x_test, path_x_test_cropped, path_x_test_resampled, path_x_test_orig = create_paths(SAVE_PATH, "Xtest")
    copy_files_to_folder(SCAN_PATH, path_x_test_orig, test_split)

    # crop RRF BBs out of SCANs
    crop_out_bbs(SCAN_PATH, RRF_BB_PATH, path_x_test_cropped, test_split, ORGAN)

    # resample cropped out area
    resample_files(path_x_test_cropped, path_x_test_resampled, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2])


def create_y_test(GT_SEG_PATH, SAVE_PATH, ORGAN, test_split):
    print("")
    print("CREATING Y TEST")
    path_y_test, path_y_test_cropped, path_y_test_resampled, path_y_test_orig = create_paths(SAVE_PATH, "ytest")
    copy_files_to_folder(GT_SEG_PATH, path_y_test_orig, test_split)

    # filter out relevant segmentation
    filter_out_relevant_segmentations(path_y_test_orig, path_y_test_orig, ORGAN)


def prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, SPLIT, ORGAN, CUSTOM_TEST_SET=None):
    # get training data
    test_split, train_split = split_train_and_test(SCAN_PATH, SPLIT, CUSTOM_TEST_SET)

    create_excel_sheet(SAVE_PATH, ORGAN, test_split, train_split)

    create_x_train(SCAN_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_y_train(GT_SEG_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_x_test(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN)
    create_y_test(GT_SEG_PATH, SAVE_PATH, ORGAN, test_split)
