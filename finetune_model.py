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
from seqeval.metrics import f1_score, accuracy_score, classification_report
import optuna

'''
This script takes the input model (MODELNAME.model) and conducts a hyperparameter search using the BERT-model-specific prepared train and val sets (train/val-MODELNAME.pt) to identify best possible num_train_epochs, learning_rate, weight_decay, and warmup_steps hyperparameters, and saves model (MODELNAME_bestparams.model). 
'''

# specify which model we want to train for. One of three choices ("distilbert", "biobert", "clinicalbert", "bluebert_pm") 
modeltype = "bluebert_pm"
labelset = 'va'
n_trials = 5

##logger=MyLogger("data/logs/tunemodel-" + labelset + "-" + modeltype + ".log") 
##logger.debug(torch.cuda.get_device_name(0))

outputmodelpath = 'data/models/' + labelset + '/' + modeltype + '_bestparams.model'

inputmodelpath = False # set to False when performing fine tuning of base models 
#inputmodelpath = 'data/models/' + labelset + '/' + modeltype + '.model'

train_dataset_path='data/' + labelset + '/preprocessed/train-' + modeltype + '.pt'
val_dataset_path='data/' + labelset + '/preprocessed/val-' + modeltype + '.pt'


def trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, n_trials, inputmodelpath = None): 
    ##logger.debug("training model type: " + modeltype)
    ##logger.debug("labels we are training for: " + labelset) 
    
    print("training model type: " + modeltype)
    print("labels we are training for: " + labelset)
    
    if labelset == "va": 
        from valabelnames import labelnames, tag2id, id2tag 
 
    print("loading datasets...") 

    ##logger.debug("train dataset: " + train_dataset_path) 
    ##logger.debug("val dataset: " + val_dataset_path) 

    train_dataset = torch.load(train_dataset_path)
    val_dataset = torch.load(val_dataset_path)

    ##logger.debug("length of training dataset: "+str(len(train_dataset.labels)))
    ##logger.debug("length of validation dataset: "+str(len(val_dataset.labels)))

    torch.cuda.empty_cache() # release cached memory so GPU application can use it
 
    training_args = TrainingArguments(
        output_dir = 'data/output/fine_tune_results_optuna', # dir to store checkpoints
        eval_steps = 500,  
        disable_tqdm = False, 
        load_best_model_at_end = True, 
        evaluation_strategy = "steps",
        metric_for_best_model = 'eval_loss',
        save_total_limit = 3, # only saves latest 3 checkpoints. Less load on memory.
        per_device_train_batch_size = 32,  # batch size per device during training
        per_device_eval_batch_size = 32,   # batch size for evaluation
        #resume_from_checkpoint = True # helpful if it crashes
    )

    if modeltype == "distilbert": 
        from transformers import DistilBertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else: 
            logger.debug("loading pretrained model...") 
            print('loading pretrained model...')
            model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels = len(tag2id))

    elif modeltype == "clinicalbert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else: 
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels = len(tag2id))

    elif modeltype == "biobert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels = len(tag2id))
            
    elif modeltype == "bluebert_pm": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from " + inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", num_labels = len(tag2id))

    def model_init():
        return model
    
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20),
            "warmup_steps":trial.suggest_int("warmup_steps", 100, 1000)
        }

    trainer = Trainer(
        model_init = model_init,                       
        args = training_args,                      
        train_dataset = train_dataset,              
        eval_dataset = val_dataset,                 
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], # early stopping 
    )

    best_trail = trainer.hyperparameter_search(
           direction = "minimize",         # default search direction by searching the minimum validation loss
           hp_space = hp_space,            
           backend = "optuna",             
           n_trials = n_trials,            
    )

    trainer.save_model(outputmodelpath)
    logger.debug("Best hyperparameters are: " + str(best_trail.hyperparameters))
    print("Best hyperparameters are: ", str(best_trail.hyperparameters)) # add print here just in case 
    return best_trail.hyperparameters 

trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, n_trials, inputmodelpath)