import numpy as np
from sklearn.model_selection import KFold
from SharedMethods import find_patient_no_in_file_name
import os
import tensorflow as tf
from UNet3D.Prepare import prepare
from UNet3D.Train import train
from UNet3D.Apply import apply
from UNet3D.Evaluate import evaluate
from UNet2D.Prepare2D import create_excel_sheet2D
from UNet2D.Train2D import train2D
from UNet2D.Apply2D import apply2D
from UNet2D.Evaluate2D import evaluate2D
import time

from openpyxl import load_workbook
import numpy as np
from openpyxl.styles import Alignment, NamedStyle, Font
from openpyxl import Workbook
import openpyxl as op

def get_files_in_path(path):
    # count how many files are in SCAN_PATH
    all_patient_numbers = set()
    for file in os.scandir(path):
        patient_no = find_patient_no_in_file_name(file.name)
        all_patient_numbers.add(patient_no)
    amount_patients = len(all_patient_numbers)
    all_patient_numbers_arr = np.zeros(amount_patients)
    i = 0
    for val in all_patient_numbers:
        all_patient_numbers_arr[i] = val
        i = i + 1

    return all_patient_numbers_arr, amount_patients


def run_KfoldCV(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, BATCH, EPOCHS, organs):
    kfold = KFold(5, True, 1)
    data, amount_patients = get_files_in_path(SCAN_PATH)
    parts = kfold.split(data)

    number = 0
    for train_set, test_set in parts:
        print("#{} train set: {} files".format(number, len(train_set)))
        print(train_set)
        print("#{} test set: {} files".format(number, len(test_set)))
        print(test_set)
        print("")

        number = number + 1
        for organ in organs:
            if organ == 'pancreas':
                thresh = 0.3
            else:
                thresh = 0.5

            prepare(SCAN_PATH, GT_BB_PATH, RRF_BB_PATH, GT_SEG_PATH, SAVE_PATH, DIMENSIONS, organ, train_set, test_set)
            create_excel_sheet2D(SAVE_PATH, organ, train_set, test_set)

            # 3D
            start = time.time()
            train(SAVE_PATH, DIMENSIONS, organ, 0.0, BATCH, EPOCHS)
            end = time.time()
            elapsed_time = end - start

            apply(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, organ, thresh)
            evaluate(SAVE_PATH, organ, number, elapsed_time)

            # 2D
            start = time.time()
            train2D(SAVE_PATH, DIMENSIONS, organ, 0.0, BATCH, EPOCHS)
            end = time.time()
            elapsed_time = end - start

            apply2D(SCAN_PATH, RRF_BB_PATH, SAVE_PATH, DIMENSIONS, organ, thresh)
            evaluate2D(SAVE_PATH, organ, number, elapsed_time)


def summarize_eval(SAVE_PATH, ORGAN):
    # create excel sheet
    eval_wb = Workbook()
    eval_sheet = eval_wb.active
    eval_sheet.title = "summarize eval"

    # create headings and apply style
    headings_style = NamedStyle(
        name="daria",
        font=Font(color='000000', bold=True),
        alignment=Alignment(horizontal='left')
    )
    headings_row = '1'
    headings = ["file #", "mean hd",
                "std", "mean avd",
                "std", "mean dice",
                "std", "time"]
    eval_sheet.append(headings)
    for cell in eval_sheet[headings_row]:
        cell.style = headings_style

    # make cells wider
    eval_sheet.column_dimensions['A'].width = 50
    eval_sheet.column_dimensions['B'].width = 20
    eval_sheet.column_dimensions['C'].width = 20
    eval_sheet.column_dimensions['D'].width = 20
    eval_sheet.column_dimensions['E'].width = 20
    eval_sheet.column_dimensions['F'].width = 20
    eval_sheet.column_dimensions['G'].width = 20
    eval_sheet.column_dimensions['H'].width = 20

    mean_dice_row = 20
    mean_standard_d_row = 21
    mean_avd_row = 20
    mean_standard_avd_row = 21
    mean_hd_row = 20
    mean_standard_hd_row = 21
    #mean_asd_row = 20
    #mean_standard_asd_row = 21
    mean_time_row = 22

    relevant_col = 2

    curr_row = 2
    curr_col = 1

    path = SAVE_PATH
    for file in os.scandir(path):
        found = file.name.find(ORGAN)
        if found is not -1:
            # open
            wb_obj = op.load_workbook(file)
            # get active sheet
            sheet_obj = wb_obj.active

            # read cells
            #DICE
            cell_mean_dice = sheet_obj.cell(row=mean_dice_row, column=relevant_col)
            cell_standard_d = sheet_obj.cell(row=mean_standard_d_row, column=relevant_col)
            #AVD
            cell_mean_avd = sheet_obj.cell(row=mean_avd_row, column=relevant_col+1)
            cell_standard_avd = sheet_obj.cell(row=mean_standard_avd_row, column=relevant_col+1)
            #HD
            cell_mean_hd = sheet_obj.cell(row=mean_hd_row, column=relevant_col+2)
            cell_standard_hd = sheet_obj.cell(row=mean_standard_hd_row, column=relevant_col+2)
            #ASD
            #cell_mean_asd = sheet_obj.cell(row=mean_asd_row, column=relevant_col+3)
            #cell_standard_asd = sheet_obj.cell(row=mean_standard_asd_row, column=relevant_col+3)
            #TIME
            cell_mean_time = sheet_obj.cell(row=mean_time_row, column=relevant_col)


            # get values
            mean_dice = cell_mean_dice.value
            standard_d = cell_standard_d.value
            mean_avd = cell_mean_avd.value
            standard_avd = cell_standard_avd.value
            mean_hd = cell_mean_hd.value
            standard_hd = cell_standard_hd.value
            #mean_asd = cell_mean_asd.value
            #standard_asd = cell_standard_asd.value
            mean_time = cell_mean_time.value


            # write into evaluation summary sheet
            patient_no = find_patient_no_in_file_name(file.name)
            #DICE
            eval_sheet.cell(column=curr_col, row=curr_row, value=patient_no)
            eval_sheet.cell(column=curr_col+1, row=curr_row, value=mean_dice)
            eval_sheet.cell(column=curr_col+2, row=curr_row, value=standard_d)
            #AVD
            eval_sheet.cell(column=curr_col+3, row=curr_row, value=mean_avd)
            eval_sheet.cell(column=curr_col+4, row=curr_row, value=standard_avd)
            #HD
            eval_sheet.cell(column=curr_col+5, row=curr_row, value=mean_hd)
            eval_sheet.cell(column=curr_col+6, row=curr_row, value=standard_hd)
            #ASD
            #eval_sheet.cell(column=curr_col+7, row=curr_row, value=mean_asd)
            #eval_sheet.cell(column=curr_col+8, row=curr_row, value=standard_asd)
            #TIME
            eval_sheet.cell(column=curr_col+7, row=curr_row, value=mean_time)

            curr_row = curr_row+1

    eval_wb.save("{}Evaluation Summary {}.xlsx".format(SAVE_PATH, ORGAN))