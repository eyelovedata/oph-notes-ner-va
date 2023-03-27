import torch 
import numpy as np
from utils import OphNERDataset, MyLogger, singledocinference, multipledocinference
from transformers import DistilBertForTokenClassification, pipeline
import json
import random 

##logger=MyLogger("data/logs/ib_va-exporttoprodigy.log") 


'''
This script selects a sample of validation examples, runs inference through a selected model, 
and outputs the ground truth and model predictions to .jsonl files that can be loaded to Prodigy for review
'''

labelset = "va" 
if labelset == "va": 
    from valabelnames import labelnames, tag2id, id2tag, inferenceid2tag
    
# Specify which model to run inference. One of three choices ("distilbert", "biobert", "clinicalbert", "bluebert_pm") 
modeltype = "bluebert_pm"
# Designate paths for model, val set, and outputs for predictions and ground truth
inputmodelpath = '/home/jupyter/ib_folder/data//models/' + labelset + '/' + modeltype + '_bestparams_retrain.model'
val_dataset_path = '/home/jupyter/ib_folder/data/' + labelset + '/preprocessed/testsample500-' + modeltype + '.pt'
outputjsonpredictionspath = '/home/jupyter/ib_folder/data/prodigyfiles/' + labelset + '/' + modeltype + '-predictions.jsonl'
outputgroundtruthfilepath = '/home/jupyter/ib_folder/data/prodigyfiles/' + labelset + '/' + modeltype + '-groundtruthlabels.jsonl'

def propagate_wordpiece_labels_to_tokens(wordpiecetokens, wordpiecetags): 
    '''
    Training labels are saved at the word piece label. 
    To export to prodigy we propagate them back up to the token level
    '''
    newdoctokens = []
    newdoclabels = []
    for i in range(len(wordpiecetokens)):
        wordpiece = wordpiecetokens[i]
        tag = wordpiecetags[i]
        if wordpiece[0:2] != "##": 
            newdoctokens.append(wordpiece)
            newdoclabels.append(tag)
        else: 
            newdoctokens[-1] = newdoctokens[-1] + wordpiece[2:]
    return newdoctokens, newdoclabels 

def get_entity_token_spans(newdoctokens, newdoclabels): 
    '''
    Outputs lines of json which include the entity span (start and end) 
    for the training labels 
    '''
    entitylist = [] 
    tokenspans = [(0,len(newdoctokens[0]))]
    for i in range(1,len(newdoctokens)): 
        token = newdoctokens[i]
        start = tokenspans[i-1][1] + 1
        end = start + len(token)
        tokenspans.append((start,end))
        label = newdoclabels[i]
        if label != 'O':
            entitylist.append({"start":start, "end":end, "label":label})
    jsonline = {}
    jsonline["text"] = ' '.join([x for x in newdoctokens])
    jsonline["spans"] = entitylist
    return jsonline  

def predictions_to_prodigy(modeltype, inputmodelpath, val_dataset_path, outputgroundtruthfilepath, outputjsonpredictionspath, inferenceid2tag, samplesize=300): 
    '''
    - performs predictions using loaded model on specified validation dataset, of random samplesize 
    - save the predictions to a jsonl file which can then be loaded into prodigy for manual annotation/correction 
    - also saves the model's training labels to a separate jsonl file 
    '''    
    ##logger.debug('loading model from '+ inputmodelpath)
    from transformers import pipeline
    if modeltype == "distilbert": 
        from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
        model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    if modeltype == "clinicalbert": 
        from transformers import BertForTokenClassification, BertTokenizerFast 
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
    if modeltype == "biobert": 
        from transformers import BertForTokenClassification, BertTokenizerFast
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-v1.1")
        
    if modeltype == "bluebert_pm":
        from transformers import BertForTokenClassification, BertTokenizerFast
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        tokenizer = BertTokenizerFast.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
    
    classifier = pipeline('ner', model = model, tokenizer = tokenizer, grouped_entities = False, ignore_labels = ['LABEL_0'])
    
    ##logger.debug('loading dataset from '+ val_dataset_path)
    val_dataset = torch.load(val_dataset_path)
    
    # determine which validation set actually has ground truth labels, to reduce dataset size  
    docidxwithlabels = []
    for i in range(len(val_dataset)):
        if val_dataset.labels[i] != [0] * len(val_dataset.labels[i]): 
            docidxwithlabels.append(i)
    ##logger.debug('number of documents with entities in them'+ str(len(docidxwithlabels)))
    
    # only perform inference over a sample of subdocuments 
    # sample the indices in docidxwithlabels 
    
    ##logger.debug("sampling "+ str(samplesize)+ " documents with labels")
    random.seed(1) 
    sampleidx = random.sample(docidxwithlabels, samplesize) 
    
    print("process ground truth labels") 
    jsonlinelist = []
    doclist = []
    #for idx in sampleidx:   # for random sampling
    for idx in docidxwithlabels:   # use docidxwithlabels to remove the random sampling  
        # get a list of tokens - aligns with the labels 
        wordpiecetokens = tokenizer.convert_ids_to_tokens(val_dataset.encodings['input_ids'][idx])
        wordpiecetags = [id2tag[id].replace('B-','').replace('I-', '') for id in val_dataset.labels[idx]]
        newdoctokens, newdoclabels = propagate_wordpiece_labels_to_tokens(wordpiecetokens, wordpiecetags)
        jsonline = get_entity_token_spans(newdoctokens, newdoclabels)
        jsonlinelist.append(jsonline)
        docstring = jsonline["text"] 
        doclist.append(docstring)
    
    print('saving ground truth labels file to ', outputgroundtruthfilepath)
    with open(outputgroundtruthfilepath, 'w') as f:
        for jsonline in jsonlinelist:
            f.write(json.dumps(jsonline) + "\n")

    ##logger.debug("performing inference") 
    multipledocinference(doclist, classifier, outputjsonpredictionspath, inferenceid2tag)
    
predictions_to_prodigy(modeltype, inputmodelpath, val_dataset_path, outputgroundtruthfilepath, outputjsonpredictionspath, inferenceid2tag)