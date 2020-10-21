from U-Net3D/Prepare import split_train_and_test, create_x_train, create_y_train, create_x_test, create_y_test
from openpyxl.styles import Alignment, NamedStyle, Font
from openpyxl import Workbook

def create_excel_sheet2D(SAVE_PATH, ORGAN, test_split, train_split):
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



def prepare2D(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, SPLIT, ORGAN, CUSTOM_TEST_SET=None):
    # get training data
    test_split, train_split = split_train_and_test(SCAN_PATH, SPLIT, CUSTOM_TEST_SET)

    create_excel_sheet2D(SAVE_PATH, ORGAN, test_split, train_split)

    create_x_train(SCAN_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_y_train(GT_SEG_PATH, GT_BB_PATH, SAVE_PATH, DIMENSIONS, train_split, ORGAN)
    create_x_test(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN)
    create_y_test(GT_SEG_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, test_split, ORGAN)
