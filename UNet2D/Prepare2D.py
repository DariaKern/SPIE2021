from UNet3D.Prepare import split_train_and_test, create_x_train, create_y_train, create_x_test, create_y_test, create_excel_sheet
from openpyxl.styles import Alignment, NamedStyle, Font
from openpyxl import Workbook

def create_excel_sheet2D(SAVE_PATH, ORGAN, test_split, train_split):
    create_excel_sheet(SAVE_PATH, ORGAN, test_split, train_split, True)


def split_train_and_test2D(SCAN_PATH, SPLIT, CUSTOM_TEST_SET=None):
    test_split, train_split = split_train_and_test(SCAN_PATH, SPLIT, CUSTOM_TEST_SET)
    return test_split, train_split


def prepare2D(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, SPLIT, ORGAN, CUSTOM_TEST_SET=None):
    # get training data
    test_split, train_split = split_train_and_test2D(SCAN_PATH, SPLIT, CUSTOM_TEST_SET)

    create_excel_sheet2D(SAVE_PATH, ORGAN, test_split, train_split)

    create_x_train(SCAN_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_y_train(GT_SEG_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_x_test(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN)
    create_y_test(GT_SEG_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN)
