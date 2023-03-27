"""
Removes rows with no labels. Performs 80/10/10 split of dataset into train/val/test sets, splitting by note. 
"""

import pandas as pd
import random 
from utils import MyLogger, checkfornomatches 

inputfilepath = "vatokenlabels-v2.csv" # remove "v2" for final?
outputfilepath = "v2-"

dftokenlabels = pd.read_csv(inputfilepath)
idx = list(dftokenlabels["note_deid"].unique())

import random
random.seed(1)
random.shuffle(idx) 
valsize = round(0.1 * len(idx))
testsize = round(0.1 * len(idx))
validx = idx[0:valsize]
testidx = idx[valsize:valsize + testsize]
trainidx = idx[valsize + testsize:]

dftokenlabels[dftokenlabels["note_deid"].isin(trainidx)].to_csv(outputfilepath + "train.csv", index = False)
dftokenlabels[dftokenlabels["note_deid"].isin(testidx)].to_csv(outputfilepath + "test.csv", index = False)
dftokenlabels[dftokenlabels["note_deid"].isin(validx)].to_csv(outputfilepath + "val.csv", index = False)
