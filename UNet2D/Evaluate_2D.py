from UNet3D.Evaluate_3D import evaluate_predictions, evaluate
from openpyxl import load_workbook


def evaluate2D(SAVE_PATH, ORGAN, ROUND, elapsed_time):
    evaluate(SAVE_PATH, ORGAN, ROUND, elapsed_time, True)
