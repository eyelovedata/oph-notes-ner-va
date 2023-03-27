# Named Entity Recognition to Recognize Visual Acuity Types in Ophthalmology Clinical Notes 

# Requirements 
Please use Anaconda to manage environments and use the included environment_clean.yml file to recreate the environment.
Notably, however, these scripts make use of the huggingface transformers library and train using PyTorch. 

# Purpose 
- To recognize 4 visual acuity (VA) types from the eye exam components of free-text ophthalmology progress notes derived from Stanford electronic health records.
- We target the following VA entities to recognize for left (OS) and right (OD) eyes: 
    - Visual acuity at distance with glasses (vaoddistcc and vaosdistcc) 
    - Visual acuity at distance without glasses (vaoddistsc and vaosdistsc)
    - Visual acuity at distance with glasses and with pinhole occluder (vaoddistccph and vaosdistccph)
    - Visual acuity at distance without glasses and with pinhole occluder (vaoddistscph and vaosdistscph)

# General Approach 
- We use use pre-trained DistilBERT, BioBERT, ClinicalBERT, and BlueBERT-PM as base models which we finetune for our specific purpose. 
- We utilize the huggingface transformers library.  

# Scripts 
## Preprocessing Data 
- tokenizeandlabel_bybatch.py: Takes raw data (valabels.csv) and performs preprocessing to propagate document-level labels to token-level labels. To do this, this script tokenizes the notes, does text search for the visual acuity measurements over those tokens, and assigns tokens to labels using a heuristic. The outputs are token-level labels saved to a csv file that is one row per note (vatokenlabels-v2.csv). This script only needs to be run once on the raw data. In our study, we tokenized and labeled in batches of 10000.
- traintestsplitbynote.ipynb: Performs a 80:10:10 train/val/test split, splitting at the document (note) level.
- prepareforbert.py: Script which prepares the data (train/val/test.csv) for input into PyTorch models, including splitting long documents into shorter subdocuments, and performing WordPieceTokenization. Each model has different input length requirements and different tokenization requirements, so this script takes a parameter which specifies which model type you want to preprocess for. Must be run separately for each different model. Outputs are preprocessed datasets stored as custom PyTorch dataset classes with extention ".pt" (train/val/test-MODELNAME.pt) which are ready to load into the model using torch.load(). 

## Model Training and Evaluation Statistics 
- trainmodel.py: Takes the BERT model-specific prepared data (train/val/test-MODELNAME.pt) and trains the desired model, outputting the saved model to desired location (MODELNAME.model). Also used to train models with best hyperparameters found using finetune_model.py. 
- evalmodel.py: Loads a saved model (MODELNAME.model), runs and saves predictions over a validation set of your choosing (predictions/MODELNAME.csv RENAME THIS FILE TO MODELNAME_predictions.csv), and computes performance metrics of validation loss, accuracy, F1-score, and classification report per class using the seqeval package. 
- finetune_model.py: For the input model (MODELNAME.model), conducts a hyperparameter search using the BERT-model-specific prepared train and val sets (train/val-MODELNAME.pt) to identify best possible num_train_epochs, learning_rate, weight_decay, and warmup_steps hyperparameters, and saves model (MODELNAME_bestparams.model). 
- varegex.py: Regular expression algorithm for identifying and labeling visual acuities, acting as a baseline model of comparison against our deep learning models. Runs the algorithm on each note of the dataset and produces a CSV file of the labeled notes that can then be evaluated manually with Prodigy using exporttoprodigy_regex.py. 


## Model Error Analysis with Prodigy 
The purpose of this set of scripts is to be able to use a user-friendly interface for annotating notes (Prodigy, https://prodi.gy/), to visualize, evaluate, and correct our models' output predictions. The goal is to take a sample of the model's predictions on the validation set, load into Prodigy, view which words are highlighted as entities, and correct these predictions. Then we can evaluate how close the model's predictions are to the human-level "ground truth". This extra step is important because the model isn't trained on human-level ground truth labels, but rather on this weakly-supervised labels as described above in preprocessing steps. 
- exporttoprodigy_bert.py: Takes in desired BERT model (MODELNAME.model) and BERT model-formatted val set (val-MODELNAME.pt), performs inference, and outputs a file with predictions (MODELNAME-predictions.jsonl) and a file with ground truths (MODELNAME-groundtruthlabels.jsonl) for downstream annotation in Prodigy.
- exporttoprodigy_regex.py: Separate script to export predictions to prodigy for regular expression algorithm output. Essentially converts regex model predictions from a .csv to a .jsonl file so that they can open be opened in Prodigy to perform hand corrections.  
- prodigycustom.py: Custom recipe for Prodigy to display the VA annotations correctly 
- prodigyexamplecommands.txt: Example commands for calling Prodigy from command line to start the interface, delete the data (start over), export the corrected data as a .jsonl when ready to analyze. 
- importfromprodigy.py: Prodigy saves annotations into .jsonl files. This script converts the .jsonl files back into a format which can be input into seqeval and can run classification metric on them. This script is labelset agnostic, assuming that the two files being compared have the same labelset. The purpose of this script is to calculate metrics comparing two .jsonl files: the original model predictions (the output from exporttoprodigy) and the hand corrections produced using prodigy.


## Utilities 
- utils.py: A collection of utility functions called by other functions in above scripts. Also contains definitions for OphNERDataset class (PyTorch dataset) 
- valabelnames.py: a variety of dictionaries of labels, label ids, etc. specific to identifying visual acuity entities 

## Useful references 
https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities 
