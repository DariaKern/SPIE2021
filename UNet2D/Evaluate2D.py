from UNet3D.Evaluate import evaluate_predictions, evaluate
from openpyxl import load_workbook


def evaluate2D(SAVE_PATH, ORGAN, ROUND, elapsed_time):
    evaluate(SAVE_PATH, ORGAN, ROUND, elapsed_time, True)
