from UNet3D.Evaluate import evaluate_predictions, summarize_eval, evaluate
from openpyxl import load_workbook

def summarize_eval2D(SAVE_PATH, ORGAN):
    summarize_eval(SAVE_PATH, ORGAN)

def evaluate2D(SAVE_PATH, ORGAN, ROUND, elapsed_time):
    evaluate(SAVE_PATH, ORGAN, ROUND, elapsed_time, True)
