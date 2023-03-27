# note: this script was used for training base models as well as training models with the best hyperparameters found using the finetune_model.py script. 
# code for training with best hparams has been commented out.

import sys
import pandas as pd
import re 
import nltk 
import numpy as np
import ast
import string 
import math
import torch
from tqdm.notebook import tqdm
from utils import OphNERDataset, MyLogger 
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging
from seqeval.metrics import f1_score, accuracy_score, classification_report 

# specify which model we want to train for. One of four choices ("distilbert", "biobert", "clinicalbert", "bluebert_pm") 
modeltype = "bluebert_pm"
labelset = "va" 

outputmodelpath = '/home/jupyter/ib_folder/data//models/' + labelset + '/' + modeltype + '.model' # training base models
#outputmodelpath = '/home/jupyter/ib_folder/data//models/' + labelset + '/' + modeltype + '_bestparams_retrain.model' # training with best hyperparams 

inputmodelpath = False # training base models
#inputmodelpath = '/home/jupyter/ib_folder/data//models/' + labelset + '/' + modeltype + '_bestparams.model' # training with best hyperparams  

train_dataset_path = '/home/jupyter/ib_folder/data/' + labelset + '/preprocessed/train-' + modeltype + '.pt'
val_dataset_path = '/home/jupyter/ib_folder/data/' + labelset + '/preprocessed/val-' + modeltype + '.pt'

##logger = MyLogger("data/logs/trainmodel-" + labelset + "-" + modeltype + ".log")  # normal training 
##logger = MyLogger("data/logs/trainmodel-" + labelset + "-" + modeltype + "_bestparams_retrain.log") # training with best hyperparams 

logger.debug(torch.cuda.get_device_name(0))

def trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, outputmodelpath, inputmodelpath=None): 
    logger.debug("training model type: "+modeltype)
    logger.debug("labels we are training for: "+labelset) 
    if labelset == "va": 
        from valabelnames import labelnames, tag2id, id2tag   

    print("loading datasets...") 

    logger.debug("train dataset: " + train_dataset_path) 
    logger.debug("val dataset: " + val_dataset_path) 

    train_dataset=torch.load(train_dataset_path)
    val_dataset=torch.load(val_dataset_path)

    logger.debug("length of training dataset: "+str(len(train_dataset.labels)))
    logger.debug("length of validation dataset: "+str(len(val_dataset.labels)))


    torch.cuda.empty_cache()
    
    
    # training base models
    training_args = TrainingArguments(
        output_dir='/home/jupyter/ib_folder/data/output',         
        num_train_epochs=num_epochs = 10,  # total number of training epochs
        per_device_train_batch_size=32,    # batch size per device during training
        per_device_eval_batch_size=32,     # batch size for evaluation
        warmup_steps=500,                  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                 # strength of weight decay
        disable_tqdm=False, 
        no_cuda=False,
        load_best_model_at_end=True, 
        evaluation_strategy="epoch",
        save_strategy="epoch"

    )
    
    ''' # training args using best hyperparameters from finetune_model.py
        training_args = TrainingArguments(
        output_dir='/home/jupyter/ib_folder/data/output',          
        num_train_epochs= 15,             
        per_device_train_batch_size=32,  
        per_device_eval_batch_size=32,   
        warmup_steps= 707,                
        weight_decay= 0.0767,               
        disable_tqdm=False, 
        no_cuda=False,
        load_best_model_at_end=True, 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate = 2.998e-05
    )
    '''
   
    if modeltype == "distilbert": 
        from transformers import DistilBertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else: 
            logger.debug("loading pretrained model...") 
            model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels = len(tag2id))

    elif modeltype == "clinicalbert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else: 
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels = len(tag2id))

    elif modeltype == "biobert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=len(tag2id))
            
    elif modeltype == "bluebert_pm": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", num_labels = len(tag2id))

    trainer = Trainer(
        model = model,                         
        args = training_args,                  
        train_dataset = train_dataset,         
        eval_dataset = val_dataset,             
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)] # do not use early stopping training with best hyperparams 
    )

    trainer.train() # the resume_from_checkpoint arg was helpful if training crashes

    #save model 
    logger.debug("Saving model to: "+ outputmodelpath) 
    trainer.save_model(outputmodelpath)
    return 

trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, outputmodelpath, inputmodelpath)
