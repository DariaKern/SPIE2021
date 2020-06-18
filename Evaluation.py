'''
Metrics for medical image processing evaluation:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/

Semantic Segmentation:
https://www.jeremyjordan.me/semantic-segmentation/

How to use sitk filter:
https://www.programmersought.com/article/5062239478/

About the Filter:
https://mevislabdownloads.mevis.de/docs/current/FMEwork/ITK/Documentation/Publish/ModuleReference/itkHausdorffDistanceImageFilter.html

'''
from Data import get_dict_of_paths
import SimpleITK as sitk
import os
import re
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, NamedStyle, Font


def calculate_loss_and_accuracy(model, X_test, y_test):
    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# returns dice coefficent, mean overlap and volume similarity measurements
def calculate_label_overlap_measures(gt_img_path, pred_img_path):
    # load images as int since float is not supported by GetDiceCoefficient()
    gt_img = sitk.ReadImage(gt_img_path, sitk.sitkInt32)
    pred_img = sitk.ReadImage(pred_img_path, sitk.sitkInt32)

    # calculate dice
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter.Execute(gt_img, pred_img)
    dice = dice_filter.GetDiceCoefficient()
    mean_overlap = dice_filter.GetMeanOverlap()
    volume_similarity = dice_filter.GetVolumeSimilarity()

    print("dice coefficient {}".format(dice))
    print("mean overlap {}".format(mean_overlap))
    print("volume similarity {}".format(volume_similarity))

    return dice, mean_overlap, volume_similarity


# returns hausdorff distance and average hausdorff distance measurements
def calculate_hausdorff_distance(gt_img_path, pred_img_path):
    # load images
    gt_img = sitk.ReadImage(gt_img_path)
    pred_img = sitk.ReadImage(pred_img_path)

    # calculate hausdorff
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(gt_img, pred_img)
    avg_hd = hd_filter.GetAverageHausdorffDistance()
    hd = hd_filter.GetHausdorffDistance()

    print("hausdorff distance {}".format(hd))
    print("average hausdorff distance {}".format(avg_hd))

    return hd, avg_hd


def evaluate_predictions(pred_path, gt_path):
    results = []

    dict_gt_paths = get_dict_of_paths(gt_path)

    # evaluate every prediction
    for prediction in os.scandir(pred_path):
        # find patient number in file name
        regex = re.compile(r'\d+')
        patient_no = int(regex.search(prediction.name).group(0))
        gt_img_path = dict_gt_paths[patient_no]

        print("Patient Number : {}".format(patient_no))
        hd, avg_hd = calculate_hausdorff_distance(gt_img_path, prediction.path)
        dice, avg_ol, vs = calculate_label_overlap_measures(gt_img_path, prediction.path)
        print("")

        results.append([patient_no, hd, avg_hd, dice, avg_ol, vs])

    return results


'''
https://realpython.com/openpyxl-excel-spreadsheets-python/
'''
def create_excel_sheet(organ):
    # create excel sheet
    wb = Workbook()
    sheet = wb.active
    sheet.title = organ

    # create headings and apply style
    headings_style = NamedStyle(
        name = "daria",
        font=Font(color='000000',bold=True),
        alignment=Alignment(horizontal='left')
    )
    headings_row = '1'
    headings = ["patient #", "hausdorff dist",
                "Ø hausdorff dist",
                "dice coeff", "Ø overlap",
                "volume similarity"]
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

    wb.save("Evaluation {}.xlsx".format(organ))


def fill_excel_sheet(save_path_results, save_path_y_test, organ):
    # open excel sheet
    wb = load_workbook(filename="Evaluation {}.xlsx".format(organ))
    sheet = wb.active

    # get metrics
    results = evaluate_predictions(save_path_results, save_path_y_test)

    # write metrics into excel
    number_of_results = len(results)
    number_of_metrics = len(results[0])
    start_row = 2
    start_col = 1
    for i in range(0, number_of_results):
        for j in range(0, number_of_metrics):
            row = start_row + i # start a new row for every result
            col = start_col + j # put metrcis into the right column
            # cut all values to 2 decimal places. Except patient number
            metric_value = "{:.2f}".format(results[i][j])
            if j == 0:
                metric_value = "{}".format(results[i][j])
            sheet.cell(column=col, row=row, value=metric_value)

    wb.save("Evaluation {}.xlsx".format(organ))