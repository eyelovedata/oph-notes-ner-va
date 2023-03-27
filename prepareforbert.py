import pandas as pd
import re 
import numpy as np
import ast
import string 
import math
import torch
import time 

from utils import splitdoc, encode_tags, fill_labels, OphNERDataset, MyLogger 

##logger=MyLogger("data/logs/prepareforbert.log") 

'''
#creates function that takes in tokenized notes, 
#calls splitdoc to makes them into shorter documents 
#runs bert tokenizer over the documents 
#propagates labels as appropriate over the word pieces 
#puts the dataset into a PyTorch appropriate dataset class 
#torch.saves the data, ready to load into model 
'''

# create encoding of tags 
# odd numbers are begin tags
# even numbers are continuation tags for the label that immediately precedes it in number

# specify which model we want to preprocess for. One of four choices ("distilbert", "biobert", "clinicalbert", "bluebert_pm")
modeltype = "bluebert_pm"  
labelset = "va" 

def create_torch_data(labelset, inputfilepath, outputfilepath, modeltype): 
    if labelset == "va": 
        from valabelnames import tag2id 
        
    start = time.time()
    if modeltype == "distilbert": 
        subdocsize = 256
        sequencelength = 512
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    
    elif modeltype == "biobert": 
        subdocsize = 256
        sequencelength = 512
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-v1.1")

    elif modeltype == "clinicalbert":
        subdocsize = 64
        sequencelength = 128
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    elif modeltype == "bluebert_pm": 
        subdocsize = 64
        sequencelength = 128
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
        
    ##logger.debug("preparing data for input into " + modeltype)
    ##logger.debug("reading input dataframe from " + inputfilepath)
    
    dftokenlabels = pd.read_csv(inputfilepath)

    # turn the df into a list of lists for tokens and tokenlabels 
    tokenlist = dftokenlabels["tokens"].apply(ast.literal_eval).tolist()
    tokenlistlabels = dftokenlabels["tokenlistlabels"].apply(ast.literal_eval).tolist() 
    
    # run helper function which splits lists of lists into smaller lists 
    print("splitting documents into subdocuments...") 
    tokens, tags = splitdoc(tokenlist, tokenlistlabels, size = subdocsize)

    # run wordpiece tokenization
    print("Word Piece Tokenizing...") 
    encodings = tokenizer(tokens, 
                          is_split_into_words=True, 
                          return_offsets_mapping=True, 
                          padding=True, 
                          truncation=True, 
                          max_length=sequencelength)

    # propagate labels through wordpieces correctly 
    print("Propagating labels to all word pieces...") 
    labels = encode_tags(tags, encodings, tag2id)

    #prepare dataset for PyTorch, takes a few minutes due to removing the offset mapping from the encodings 
    print("Creating PyTorch dataset...") 
    encodings.pop("offset_mapping") # we don't want to pass this to the model
    dataset = OphNERDataset(encodings, labels)
    ##logger.debug("output dataset encodings are of length " + str(len(dataset)))
    ##logger.debug("saving output pytorch dataset to" + outputfilepath)

    torch.save(dataset, outputfilepath)
    end = time.time()
    ##logger.debug("elapsed time: " + str( end - start) + " seconds")    
    return 

###inputfilepath ='/home/jupyter/ib_folder/data/va/v2-train.csv'
###outputfilepath='/home/jupyter/ib_folder/data/'+labelset+'/preprocessed/train-'+modeltype+'.pt'
###create_torch_data(labelset, inputfilepath, outputfilepath, modeltype)

###inputfilepath ='/home/jupyter/ib_folder/data/va/v2-test.csv'
###outputfilepath='/home/jupyter/ib_folder/data/'+labelset+'/preprocessed/test-'+modeltype+'.pt'
###create_torch_data(labelset, inputfilepath, outputfilepath, modeltype)

###inputfilepath ='/home/jupyter/ib_folder/data/va/v2-val.csv'
###outputfilepath='/home/jupyter/ib_folder/data/'+labelset+'/preprocessed/val-'+modeltype+'.pt'
###create_torch_data(labelset, inputfilepath, outputfilepath, modeltype)


# prodigy review
###inputfilepath ='/home/jupyter/ib_folder/data/va/testsample500.csv'
###outputfilepath='/home/jupyter/ib_folder/data/'+labelset+'/preprocessed/testsample500-'+modeltype+'.pt'
###create_torch_data(labelset, inputfilepath, outputfilepath, modeltype)




