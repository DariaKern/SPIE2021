"""
"""

import shutil
import os, re
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from pathlib import Path

'''_____________________________________________________________________________________________'''
'''|.................................Helping Methods...........................................|'''
'''_____________________________________________________________________________________________'''


def find_patient_no_in_file_name(file_name):
    # find patient number in file name
    regex = re.compile(r'\d+')
    patient_no = int(regex.search(file_name).group(0))

    return patient_no