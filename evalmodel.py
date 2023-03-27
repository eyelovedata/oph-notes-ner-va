import torch 
import numpy as np
from utils import OphNERDataset, MyLogger
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments, DistilBertConfig
import pandas as pd
from seqeval.metrics import f1_score, accuracy_score, classification_report 

'''
This script takes as inputs the model, modeltype, and validation set path. 
It runs the model on the validation set and displays seqeval classification metrics. 
'''

#specify which model we want to evaluate for. One of four choices ("distilbert", "biobert", "clinicalbert", "bluebert_pm") 
modeltype = "bluebert_pm"
labelset = "va" 

inputmodelpath = '/home/jupyter/ib_folder/data/models/va/' + modeltype + '.model' # base trained model
#inputmodelpath = '/home/jupyter/ib_folder/data/models/va/' + modeltype + '_bestparams.model' # fine-tuned model
#inputmodelpath = '/home/jupyter/ib_folder/data/models/va/' + modeltype + '_bestparams_retrain.model' # fine-tuned model trained with best hparams

val_dataset_path='/home/jupyter/ib_folder/data/' + labelset + '/preprocessed/val-' + modeltype + '.pt'

output_predictions_path='/home/jupyter/ib_folder/data/' + labelset + '/predictions/' + modeltype + '.csv'

##logger=MyLogger('/home/jupyter/ib_folder/data/logs/evalmodel-' + labelset + '-' + modeltype + '.log') # base trained model
##logger=MyLogger('/home/jupyter/ib_folder/data/logs/evalmodel-' + labelset + '-' + modeltype + '_bestparams.log') # fine-tuned model
##logger=MyLogger('/home/jupyter/ib_folder/data/logs/evalmodel-' + labelset + '-' + modeltype + '_bestparams_retrain.log') # fine-tuned model trained with best hparams

def evalmodel(labelset, modeltype, inputmodelpath, val_dataset_path, output_predictions_path): 
    if labelset == "va": 
        from valabelnames import labelnames, tag2id, id2tag

    ##logger.debug("now evaluating " + inputmodelpath + " on " + "labelset" + " labels") 
    ##logger.debug("val dataset path: " + val_dataset_path) 
    
    val_dataset=torch.load(val_dataset_path)

    ##logger.debug("length of validation dataset: "+str(len(val_dataset.labels)))

    if modeltype == "distilbert": 
        from transformers import DistilBertForTokenClassification
        model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
    if modeltype == "clinicalbert": 
        from transformers import BertForTokenClassification 
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
    if modeltype == "biobert": 
        from transformers import BertForTokenClassification
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)
    if modeltype =="bluebert_pm": 
        from transformers import BertForTokenClassification
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only = True)    

    model.eval()

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = True)

    #device = torch.device('cuda')
    #model.to(device)
    
    eval_loss, eval_accuracy = 0, 0  # reset the validation loss for this epoch.
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    num_batches = len(test_loader)
    for batch_num, batch in enumerate(test_loader):
        print(str(batch_num ) + '/' + str(num_batches))
        b_input_ids = batch['input_ids']#.to(device)
        b_input_mask = batch['attention_mask']#.to(device)
        b_labels = batch['labels']#.to(device)

        # do not compute or store gradients to save memory and speed up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids,
                            attention_mask = b_input_mask, labels = b_labels)
            
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis = 2)])
        true_labels.extend(label_ids)
        
    validation_loss_values = []
    eval_loss = eval_loss / len(test_loader)
    validation_loss_values.append(eval_loss)
    logger.debug("Validation loss: " + str(eval_loss)) 
    #print("Validation loss: {}".format(eval_loss))

    # outputs lists of lists which works for seqeval
    pred_tags = [[id2tag[label] for label in doc] for doc in predictions]
    valid_tags = [[id2tag[label] for label in doc] for doc in true_labels]
    
	# saves the outputs to a pandas dataframe for bootstrapping later on 
    df = pd.DataFrame({'pred_tags':pred_tags, 'valid_tags':valid_tags})
    df.to_csv(output_predictions_path, index = False)

    logger.debug("Validation Accuracy: " + str(accuracy_score(pred_tags, valid_tags)))
    logger.debug("Validation F1-Score: " + str(f1_score(pred_tags, valid_tags)))
    print()

    #seqeval version takes a list of lists of tags, corresponding to each document 
    logger.debug(classification_report(valid_tags, pred_tags))
    return 
    
    
evalmodel(labelset, modeltype, inputmodelpath, val_dataset_path, output_predictions_path)