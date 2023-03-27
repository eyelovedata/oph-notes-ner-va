"""Baseline regular expression algorithm for identifying and labeling visual acuities."""

import sys

import re
import pandas as pd

from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import string 
import math
import time
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer 
from utils import gettokenspanlist, findlabel, findmatchingtoken, returnlabels, initializetokenlistlabels, gettokenlistlength, checkfornomatches, remove_unicode_specials, MyLogger, select_longest_regex_finding



# PART 1.1: Formatting input to regex  
# Test set does not yet have the raw text notes so we need to find the test notes from the original dataset and add that to the dataframe

inputdata = pd.read_csv("gs://stanfordoptimagroup/STRIDE/oph-notes-ner/data/va/valabels.csv") # the entire va dataset
inputnotes = inputdata['note'] # extract the notes 
inputdeids = inputdata['note_deid'] # extract the deids  

testset = pd.read_csv("v2-test.csv") # the test set for all our models 
testsetdeids = testset['note_deid'] # the deids of all test data 


# PART 1.2: Since note_deids are in order for both datasets, we need to find the corresponding note to each deid in the test set
testnotes = [] 
for testID in testsetdeids:
    row = inputdata.loc[inputdata['note_deid'] == testID] 
    testnotes.append(row.iloc[0,14]) # make a list of notes we need to incorporate in our test set; these are in order by deid 


    
# PART 1.3: creating the csv file to use as input to the regex
testset_with_notes = testset 
testset_with_notes['note'] = testnotes # this will be the input to the regex model
testset_with_notes.to_csv('varegextestset.csv', index = False) 


# Part 2.1: Initialize variables, specific to our data format

labelset = 'va'
inputfilepath = "varegextestset.csv"
outputfilepath = 'vatestregex-outset12.csv'
samplesize = 2793 # size of test set
skiprows = 0 
header = 0 


# Part 2.1 (Free text): Initialize variables
# Use these arguments for running the algorithm on a dataset of just notes/ free text 
# labelset = 'va'
# inputfilepath = 'consult_notes_300distance.csv'
# outputfilepath = 'consultnotesdist-outset.csv'
# samplesize = 300
# skiprows = 0
# header = 0


def getregextokenlabels(labelset, inputfilepath, outputfilepath, samplesize, skiprows, header):
    
    #read the input csv file
    if labelset == 'va': 
        from valabelnames import labelnames 
        df=pd.read_csv(inputfilepath, nrows=samplesize, skiprows=skiprows, header=0, 
        low_memory=True)
        
    
    df["note"]=df["note"].apply(remove_unicode_specials)   # remove unicode special characters from note text
    print("length of input dataframe:", str(len(df)))
    
    print("tokenizing...")
    #tokenize
    t = TreebankWordTokenizer()
    df["tokens"]=df["note"].apply(t.tokenize)
    df["tokenspanlist"]=df["note"].apply(gettokenspanlist)
    df["doclength"]=df["tokenspanlist"].apply(len)
    
    #initialize tokenlist labels to all 'O'
    df["tokenlistlabels"]=df["tokenspanlist"].apply(initializetokenlistlabels)
    
    return df  # return the new dataframe with the cleaned note column so we can experiment with regex expessions with the notes
  
    
    
df = getregextokenlabels(labelset, inputfilepath, outputfilepath, samplesize, skiprows, header)


# List of all notes in test set in deid order
notes = df.note


# Create output dataframe
outputdf = pd.DataFrame()
outputdf['note_deid'] = df['note_deid'] # comment out for consult notes since consult notes does not include note_deid

outputdf['doclength'] = df['doclength']
outputdf['tokens'] = df['tokens']


def findVAs(notes):   
    """
    Use regular expressions to identify VA exam sections and label corresponding VAs.

    Parameters:
        notes (list): A list of notes to be processed, where each element is a string.

    Returns:
        None

    Notes:
        This function performs various string matching and labeling operations to identify patterns related to VA in the input notes. The function uses regular expressions to match specific patterns and extracts the corresponding tokens from the notes using the 'tokenspanlist' and 'tokenlistlabels' dictionaries. The function then modifies the 'labeldict' dictionary to label the corresponding tokens with the approprate VA label. 
    """

    # Regular expressions 
    regex1 = r"(?i)(\s?dist)?\s?(cc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))"  # separate instances of 'cc 20/20  cc 20/25'
    regex2 = r"(?i)(\s?dist)?\s?(with correction(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))"  # separate instances of 'With correction 20/20...'
    regex3 = r"(?i)(\s?dist)?\s?(sc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))" # separate instances of 'sc' 
    regex4 = r"(?i)(\s?dist)?\s?(without correction(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))" # separate instances of 'without correction'
    regex5 = r"(?i)(\s?dist)?\s?(ph sc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))" # separate instances of ph sc with or without measurements
    regex6 = r"(?i)(with pin-hole(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))" # separate instances of with-pinhole with or without a measurement for use with 'with correction'
    regex7 = r"(?i)(\s?dist)?\s?(ph cc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))" # separate instances of ph cc with or without measurement
    
    # ie. "with correction 20/20  20/20"  (back to back acuities)
    regex8 = r"(?i)(\s?dist)?\s?(with correction(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    # ie. "without correction 20/20  20/20"  (back to back acuities)
    regex9 = r"(?i)(\s?dist)?\s?(without correction(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    # with pin-hole back to back
    regex10 = r"(?i)(with pin-hole(:)?)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    
    regex11 = r"(?i)(\s?dist)?\s?(cc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    regex12 = r"(?i)(\s?dist)?\s?(sc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    regex13 = r"(?i)(\s?dist)?\s?(ph cc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    regex14 = r"(?i)(\s?dist)?\s?(ph sc)([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})(-?\+?\d?)?))([ \t]+)?((hm( at \d(\\)?')?) | (hand motion( at \d(\\)?')?)| (cf( at \d(\\)?')?) | (count fingers( at \d(\\)?')?) | lp | nlp | ni | (20)\/((\d{2,3})([ \t]+)?(-?\+?\d?)?))"
    
    matchcount = 0 # variable to keep track of how many notes a VA was found in 
    
    tklabels = []
    
    for i in range(0, notes.size):
        
        # Turn tokenspanlist and tokenlistlabels into a dictionary
        labeldict = dict(zip(df['tokenspanlist'][i], df['tokenlistlabels'][i]))
        tokenspanlist = df['tokenspanlist'][i]
        
        regex1match = re.search(regex1, notes[i])
        regex2match = re.search(regex2, notes[i])
        regex3match = re.search(regex3, notes[i])
        regex4match = re.search(regex4, notes[i])
        regex6match = re.search(regex6, notes[i]) 
         
        count = 0 # Count variable to keep track of how many acuities (we only want to take the first two) 
        
        if(regex1match):  
            matchcount = matchcount + 1
            # If more than two matches, just consider the first two 
            consecutivematch = re.search(regex11, notes[i])
            if(consecutivematch):
                listofspans = extractdoublevaspan(tokenspanlist, consecutivematch)  # This should be 2 lists of lists of tuples
                for j,spanlist in enumerate(listofspans):
                    if j == 0: # vaoddistcc (right eye)
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaoddistcc'
                            else:
                                labeldict[span] = 'I-vaoddistcc'
                    else: # vaosdistcc (left eye)
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaosdistcc'
                            else:
                                labeldict[span] = 'I-vaosdistcc'
            else:
                for match in re.finditer(regex1, notes[i], re.S):
                    count = count + 1 
                    if count > 2:
                        break
                    else: 
                        vaspanlist = extractvaspan(tokenspanlist, match)
                        if count == 1:  
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaoddistcc'
                                else:
                                    labeldict[span] = 'I-vaoddistcc'
                        else:  
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaosdistcc'
                                else:
                                    labeldict[span] = 'I-vaosdistcc'
            phmatch = re.search(regex7, notes[i])

            if(phmatch): 
                phconsecutivematch = re.search(regex13, notes[i])
                if(phconsecutivematch):
                    listofspans = extractdoublevaspan(tokenspanlist, phconsecutivematch)  
                    for j,spanlist in enumerate(listofspans):
                        if j == 0: 
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaoddistccph'
                                else:
                                    labeldict[span] = 'I-vaoddistccph'
                        else: 
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaosdistccph'
                                else:
                                    labeldict[span] = 'I-vaosdistccph'
                else:
                    count = 0 
                    for match in re.finditer(regex7, notes[i], re.S):
                        count = count + 1 
                        if count > 2:
                            break
                        else:
                            vaspanlist = extractvaspan(tokenspanlist,match)
                            if (vaspanlist) and (count == 1):  
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaoddistccph'
                                    else:
                                        labeldict[span] = 'I-vaoddistccph'
                            elif (vaspanlist) and (count == 2):
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaosdistccph'
                                    else:
                                        labeldict[span] = 'I-vaosdistccph'
                            else:
                                continue
        elif(regex2match):   
            matchcount = matchcount + 1
            
            consecutivematch = re.search(regex8, notes[i])
            if(consecutivematch):
                listofspans = extractdoublevaspan(tokenspanlist, consecutivematch)  
                for j,spanlist in enumerate(listofspans):
                    if j == 0: 
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaoddistcc'
                            else:
                                labeldict[span] = 'I-vaoddistcc'
                    else: 
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaosdistcc'
                            else:
                                labeldict[span] = 'I-vaosdistcc'
            else:
                for match in re.finditer(regex2, notes[i], re.S):
                    count = count + 1 
                    if count > 2:
                        break
                    else:  
                        vaspanlist = extractvaspan(tokenspanlist,match) 
                        if count == 1:  
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaoddistcc'
                                else:
                                    labeldict[span] = 'I-vaoddistcc'
                        else: 
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaosdistcc'
                                else:
                                    labeldict[span] = 'I-vaosdistcc'
            phmatch = re.search(regex6, notes[i])
            if(phmatch): 
                phconsecutivematch = re.search(regex10, notes[i])
                if(phconsecutivematch):
                    listofspans = extractdoublevaspan(tokenspanlist, phconsecutivematch)  
                    for j,spanlist in enumerate(listofspans):
                        if j == 0: # right eye
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaoddistccph'
                                else:
                                    labeldict[span] = 'I-vaoddistccph'
                        else: # left eye
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaosdistccph'
                                else:
                                    labeldict[span] = 'I-vaosdistccph'
                else:
                    count = 0 
                    for match in re.finditer(regex6, notes[i], re.S):
                        count = count + 1 
                        if count > 2:
                            break
                        else:
                            vaspanlist = extractvaspan(tokenspanlist,match)
                            if (vaspanlist) and (count == 1): 
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaoddistccph'
                                    else:
                                        labeldict[span] = 'I-vaoddistccph'
                            elif (vaspanlist) and (count == 2):
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaosdistccph'
                                    else:
                                        labeldict[span] = 'I-vaosdistccph'
                            else:
                                continue
        elif(regex3match):  
            matchcount = matchcount + 1
            consecutivematch = re.search(regex12, notes[i])
            if(consecutivematch):
                listofspans = extractdoublevaspan(tokenspanlist, consecutivematch)  
                for j,spanlist in enumerate(listofspans):
                    if j == 0: 
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaoddistsc'
                            else:
                                labeldict[span] = 'I-vaoddistsc'
                    else: 
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaosdistsc'
                            else:
                                labeldict[span] = 'I-vaosdistsc'
            else:
                for match in re.finditer(regex3, notes[i], re.S):
                    count = count + 1 
                    if count > 2:
                        break
                    else:  
                        vaspanlist = extractvaspan(tokenspanlist,match) 
                        if count == 1:  
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaoddistsc'
                                else:
                                    labeldict[span] = 'I-vaoddistsc'

                        else:
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaosdistsc'
                                else:
                                    labeldict[span] = 'I-vaosdistsc'
            phmatch = re.search(regex5, notes[i])
            if(phmatch): 
                phconsecutivematch = re.search(regex14, notes[i])
                if(phconsecutivematch):
                    listofspans = extractdoublevaspan(tokenspanlist, phconsecutivematch)  
                    for j,spanlist in enumerate(listofspans):
                        if j == 0: 
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaoddistscph'
                                else:
                                    labeldict[span] = 'I-vaoddistscph'
                        else: 
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaosdistscph'
                                else:
                                    labeldict[span] = 'I-vaosdistscph'
                else:
                    count = 0 
                    for match in re.finditer(regex5, notes[i], re.S):
                        count = count + 1 
                        if count > 2:
                            break
                        else:
                            vaspanlist = extractvaspan(tokenspanlist,match)
                            if (vaspanlist) and (count == 1): 
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaoddistscph'
                                    else:
                                        labeldict[span] = 'I-vaoddistscph'

                            elif (vaspanlist) and (count == 2):
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaosdistscph'
                                    else:
                                        labeldict[span] = 'I-vaosdistscph'

                            else:
                                continue
        elif(regex4match):  
            matchcount = matchcount + 1
            consecutivematch = re.search(regex9, notes[i])
            if(consecutivematch):
                listofspans = extractdoublevaspan(tokenspanlist, consecutivematch) 
                for j,spanlist in enumerate(listofspans):
                    if j == 0: # right eye
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaoddistsc'
                            else:
                                labeldict[span] = 'I-vaoddistsc'
                    else: # left eye
                        for k,span in enumerate(spanlist):
                            if k == 0:
                                labeldict[span] = 'B-vaosdistsc'
                            else:
                                labeldict[span] = 'I-vaosdistsc'
            else:
                for match in re.finditer(regex4, notes[i], re.S):
                    count = count + 1 
                    if count > 2:
                        break
                    else: # consolidate the below into a method 
                        vaspanlist = extractvaspan(tokenspanlist,match) 
                        if count == 1:  # vaoddistcc (right eye)
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaoddistsc'
                                else:
                                    labeldict[span] = 'I-vaoddistsc'
                        else: # vaosdistcc (left eye)
                            for ind, span in enumerate(vaspanlist):
                                if ind == 0:
                                    labeldict[span] = 'B-vaosdistsc'
                                else:
                                    labeldict[span] = 'I-vaosdistsc'
            phmatch = re.search(regex6, notes[i])
            if(phmatch): 
                # Check to see if this is a consecutive "With Pinhole 20/20  20/20" case
                # If yes run the specific methods otherwise, 
                phconsecutivematch = re.search(regex10, notes[i])
                if(phconsecutivematch):
                    listofspans = extractdoublevaspan(tokenspanlist, phconsecutivematch) 
                    for j,spanlist in enumerate(listofspans):
                        if j == 0: # right eye
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaoddistscph'
                                else:
                                    labeldict[span] = 'I-vaoddistscph'
                        else: # left eye
                            for k,span in enumerate(spanlist):
                                if k == 0:
                                    labeldict[span] = 'B-vaosdistscph'
                                else:
                                    labeldict[span] = 'I-vaosdistscph'
                else:
                    count = 0 
                    for match in re.finditer(regex6, notes[i], re.S):
                        count = count + 1 
                        if count > 2:
                            break
                        else:
                            vaspanlist = extractvaspan(tokenspanlist,match)
                            if (vaspanlist) and (count == 1): 
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaoddistscph'
                                    else:
                                        labeldict[span] = 'I-vaoddistscph'
                            elif (vaspanlist) and (count == 2):
                                for ind, span in enumerate(vaspanlist):
                                    if ind == 0:
                                        labeldict[span] = 'B-vaosdistscph'
                                    else:
                                        labeldict[span] = 'I-vaosdistscph'

                            else:
                                continue

        tklabels.append(list(labeldict.values()))
        
    
    outputdf['tokenlistlabels'] = tklabels
    

# The match contains both acuities 

def extractdoublevaspan(tokenspanlist, match):
    """
    Finds corresponding span for each visual acuity contained in the regular expression match when acuities are listed back to back.
    
    Parameters:
        tokenspanlist (list): list of token spans for a specific note
        match (regex match): regex match containing header words and VA

    Returns:
        spanlists (list): list of spans corresponding to only VAs
    """
    
    varegex = r"(?i)(((\d{2,3})\/((\d{2,3})(-?\+?\d?)?))|(hand motion( at \d(\\)?')?)|(hm( at \d(\\)?')?)|hm|(count fingers( at \d(\\)?')?)|(cf( at \d(\\)?')?)|cf|lp|nlp|ni)" # adding the ()
    iregex = r"(?i)((hand motion at \d(\\)?'?)|(hm at \d(\\)?'?)|(count fingers at \d(\\)?'?)|(cf at \d(\\)?'?))"

    spanlists = [] # a list of tuples
    totalstr = match.group(0)
    spanstart = match.span()[0]
    
    imatch = re.search(iregex, totalstr)
    
    for casematch in re.finditer(varegex, totalstr, re.S): 
        vaspanlist = []
        vastart = spanstart + casematch.span()[0] 
        vaend = spanstart + casematch.span()[1] 
        vaspan = (vastart, vaend)  # Find span in context of the whole note
        if(imatch) :
            vaspanlist = getspanlist(tokenspanlist, vastart, vaend)
        else :
            vaspanlist.append(vaspan)
        spanlists.append(vaspanlist)
    
    return spanlists

def extractvaspan(tokenspanlist, match):
    """
    Finds corresponding span for each visual acuity contained in the regular expression match.
    
    Parameters:
        tokenspanlist (list): list of token spans for a specific note
        match (regex match): regex match containing header words and VA

    Returns:
        vaspanlist (list): list of spans corresponding to only VAs
    """
    
    varegex = r"(?i)(((\d{2,3})\/((\d{2,3})(-?\+?\d?)?))|(hand motion( at \d(\\)?')?)|(hm( at \d(\\)?')?)|hm|(count fingers( at \d(\\)?')?)|(cf( at \d(\\)?')?)|cf|lp|nlp|ni)" # adding the ()
    
    #The i regex is to flag down multiword acuities (variations of count finger or hand motion)
    iregex = r"(?i)((hand motion at \d(\\)?'?)|(hm at \d(\\)?'?)|(count fingers at \d(\\)?'?)|(cf at \d(\\)?'?))"
    
    vaspanlist = []
    
    totalstr = match.group(0)   # string match (ie. ' cc  20/25')
    spanstart = match.span()[0] # start of the span of the entire match in context of the entire note 
    vamatch = re.search(varegex, totalstr)  # match of the actual visual acuity ('20/25')
    imatch = re.search(iregex, totalstr)

    if(vamatch):
        vastart = vamatch.span()[0]  # start of va (not in relation to entire note)
        vaend = vamatch.span()[1]  # end of va span (not in relation to entire note)
        vaspanstart = spanstart + vastart  # start of va in context of entire note
        vaspanend = spanstart + vaend  # end of va in context of entire note
        vaspan = (vaspanstart, vaspanend)  # tuple of the span we want for the acuity
        vaspanlist.append(vaspan)
        if(imatch):
            vaspanlist = getspanlist(tokenspanlist, vaspanstart, vaspanend)
            return vaspanlist
        return vaspanlist
    else:
        return None 

    
def getspanlist(tokenspanlist, vastart, vaend):
    """
    Function to identify corresponding spans for multiword VAs. 

    
    Parameters:
        tokenspanlist (list): A list of sample notes to be processed, where each element is a string.
        vastart (int): The start index of the multiword VA span
        vaend (int): The end index of the multiword VA span

    Returns:
        tokenspanlist (list): A subset of spans from tokenspanlist
    """
    startind = endind  = 0
    for i, span in enumerate(tokenspanlist):
        if span[0] == vastart:
            startind = i
        if span[1] == vaend:
            endind = i
            break
    return tokenspanlist[startind : endind+1]



findVAs(notes) # Run the main function 


outputdf.to_csv(outputfilepath, index = False) # Save output to a csv file; use this to run evals
