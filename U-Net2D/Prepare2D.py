import os, random, shutil
import SimpleITK as sitk
from SharedMethods import find_patient_no_in_file_name
from openpyxl.styles import Alignment, NamedStyle, Font
from openpyxl import Workbook

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

    wb.save("{}2DEvaluation {}.xlsx".format(SAVE_PATH, ORGAN))