import numpy as np
from utils import OphNERDataset, MyLogger, defaultlabels
import json
import random 
import pandas as pd 
import ast 

'''
Separate script to export predictions to prodigy for regular expression algorithm output. Essentially converts regex model predictions from a CSV to a JSONL file so that we can open them in Prodigy and perform hand corrections. 
'''

labelset = "va" 
if labelset == "va": 
    from valabelnames import labelnames, tag2id, id2tag, inferenceid2tag

val_dataset_path = 'varegexpreds300.csv'
outputregexpath = 'varegex300-predictions.jsonl'
samplesize = 300

dftokenlabels = pd.read_csv(val_dataset_path)
dftokenlabels.sort_values(by = ['note_deid'], inplace = True)  # comment out for data with just notes/ if there is no note_deid col 
print('length of tokenlabels', len(dftokenlabels)) 

def get_entity_token_spans(newdoctokens, newdoclabels): 
    '''
    Outputs lines of json which include the entity span (start and end) 
    for the training labels 
    '''
    entitylist = [] 
    # turn newdoctokens into a list of token spans 
    tokenspans = [(0,len(newdoctokens[0]))]
    for i in range(1,len(newdoctokens)): 
        token = newdoctokens[i]
        start = tokenspans[i - 1][1] + 1
        end = start + len(token)
        tokenspans.append((start,end))
        label = newdoclabels[i]
        if label != 'O':
            entitylist.append({"start":start, "end":end, "label":label})
    jsonline = {}
    jsonline["text"] = ' '.join([x for x in newdoctokens])
    jsonline["spans"] = entitylist
    return jsonline

def save_labels_to_prodigy(outputfilepath, df): 
    jsonlinelist = []
    doclist = []
    for idx in range(len(df)): 
        # get a list of tokens - aligns with the labels 
        wordpiecetokens = ast.literal_eval(df.iloc[idx]["tokens"])
        wordpiecetags = [label.replace('B-','').replace('I-', '') for label in ast.literal_eval(df.iloc[idx]["tokenlistlabels"])]
        #wordpiecetags = [[label for label in ast.literal_eval(doc)] for doc in predictions]
        jsonline = get_entity_token_spans(wordpiecetokens, wordpiecetags)
        jsonlinelist.append(jsonline)
        docstring = jsonline["text"] 
        doclist.append(docstring)
    
    print('saving labels file to ', outputfilepath)
    with open(outputfilepath, 'w') as f:
        for jsonline in jsonlinelist:
            f.write(json.dumps(jsonline) + "\n")
    return doclist 

print('process regex file...') 
save_labels_to_prodigy(outputregexpath, dftokenlabels)    

