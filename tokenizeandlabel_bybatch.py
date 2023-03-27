import pandas as pd
import re 
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import ast
import string 
import math
import time
from nltk.tokenize.treebank import TreebankWordTokenizer
from utils import gettokenspanlist, findlabel, findvaheader, findsleheader, findextheader, findllheader, findmatchingtoken, returnlabels, initializetokenlistlabels, gettokenlistlength, countmatches, remove_unicode_specials, MyLogger 

##logger = MyLogger("logs/vatokenizeandlabel.log") 

'''
Total entries in valabels.csv is 319756. We tokenize and label in batches of 10000 = 32 total batches (31 batches of 10000 and 1 batch of 9756). A for loop running through all 32 batches resulted in the environment going dead so I ran the script three times: first iterating through batches 0 through 12, then 13 through 23, and then 24 through 31. 

Once you run the script 3 times, combine the outputs into a complete csv that we can use as input for traintestsplitbynote.py
'''

# First run iterates through a total of 13 batches
#run = 1
#startbatch = 0
#endbatch = 13

# Second run iterates through a total of 11 batches
#run = 2
#startbatch = 13
#endbatch = 24

# Third run iterates through a total of 8 batches
run = 3
startbatch = 24
endbatch = 32


t = TreebankWordTokenizer() 

inputfilepath = "gs://stanfordoptimagroup/STRIDE/oph-notes-ner/data/va/valabels.csv" # KEEP THIS PATH?
outputfilepath = "va-tokenlabels"+str(run)+".csv"
labelset = 'va' 

def gettokenlabels(labelset, inputfilepath, outputfilepath, samplesize, startat):   
    start = time.time() 
    ##logger.debug("reading input dataframe from "+ inputfilepath) 
    
    if labelset == 'va': 
        from valabelnames import labelnames 
        df=pd.read_csv(inputfilepath, nrows=samplesize, skiprows=startat, header=None, 
        low_memory=True, names=['smartformid', 'vaoddistcc', 'vaoddistccph', 'vaoddistsc',
        'vaoddistscph', 'vaodnearcc', 'vaodnearsc', 'vaosdistcc', 'vaosdistccph', 'vaosdistsc',
        'vaosdistscph', 'vaosnearcc', 'vaosnearsc', 'note_deid', 'note'])

    df["note"]=df["note"].apply(remove_unicode_specials)
    ##logger.debug("length of input dataframe:" +str(len(df)))
    
    df=df[df[labelnames].notnull().any(1)] #keep rows where any labelnames are not null

    outputdf = pd.DataFrame() # initiate an empty pandas dataframe to add the output from each batch to
    
    # For loop to process batches more efficiently
    # Can successfully process through about 12 batches of size 10,000 each in the for loop without crashing
    for i in range(startbatch, endbatch):
        print("tokenizing and labeling batch " + str(i) + "/32")
        if i != 31:
            batch = df.iloc[i*10000 : i*10000 + 10000]
        else:
            batch = df.iloc[i*10000 : i*10000 + 9757]
        print("tokenizing batch: " + str(i))
        #tokenize
        t = TreebankWordTokenizer()
        batch["tokens"]=batch["note"].apply(t.tokenize)
        batch["tokenspanlist"]=batch["note"].apply(gettokenspanlist)
        batch["doclength"]=batch["tokenspanlist"].apply(len)
    
        #initialize tokenlist labels to all 'O'
        batch["tokenlistlabels"]=batch["tokenspanlist"].apply(initializetokenlistlabels)
    
        #for each label, find the spans, match-index, and update the token list labels
        print("finding the matches and updating the token list labels for batch: " + str(i))
        for name in labelnames: 
            batch[name+'_spanlist']=batch[[name,"note"]].apply(lambda x: findlabel(*x), axis=1)
            batch[name+"_matchindexlist"]=batch[["tokenspanlist",name+"_spanlist"]].apply(lambda x: findmatchingtoken(*x), axis=1) 
            batch["tokenlistlabels"]=batch[[name+"_matchindexlist","tokenlistlabels"]].apply(lambda x: returnlabels(*x, name), axis=1)
    
        ##logger.debug("length of resulting dataframe: "+ str(len(df)))
    
        batch['labelcount']=batch[labelnames].count(axis=1) # count how many smartform labels there are
        
        outputdf = outputdf.append(batch)
    
    ##logger.debug("saving output dataframe to "+ outputfilepath)
    outputdf[["note_deid", "doclength", "tokens", "tokenlistlabels", "labelcount"]].to_csv(outputfilepath, index=False)
    
    end = time.time()
    ##logger.debug("elapsed time: "+ str(end - start)+ " seconds") 
    return 

gettokenlabels(labelset, inputfilepath, outputfilepath, samplesize=319756, startat=1)
##logger.debug("reading back in the full dataset...") 

df=pd.read_csv(outputfilepath)

##logger.debug("calculating ratio of matches to labels...") 
df["detectedlabelcount"]=df["tokenlistlabels"].apply(countmatches)
df["ratiodetectedlabels"]=df["detectedlabelcount"]/df["labelcount"]

##logger.debug("length dataframe after discarding notes without complete label matches: "+ str(len(df[df["ratiodetectedlabels"]>=1]) )) 
##logger.debug("length dataframe after discarding notes without >80% label matches: "+ str(len(df[df["ratiodetectedlabels"]>=0.8]) ))
##logger.debug("length dataframe after discarding notes without >50% label matches: "+ str(len(df[df["ratiodetectedlabels"]>=0.5]) ))
##logger.debug("length dataframe after discarding notes without >20% label matches: "+ str(len(df[df["ratiodetectedlabels"]>=0.2]) ))

df[df["ratiodetectedlabels"]>=0.2].to_csv(outputfilepath[0:-4]+"-gt20pctlabelmatches.csv", index=False)
df[df["ratiodetectedlabels"]>=0.5].to_csv(outputfilepath[0:-4]+"-gt50pctlabelmatches.csv", index=False)
df[df["ratiodetectedlabels"]>=0.8].to_csv(outputfilepath[0:-4]+"-gt80pctlabelmatches.csv", index=False)
df[df["ratiodetectedlabels"]>=1].to_csv(outputfilepath[0:-4]+"-100pctlabelmatches.csv", index=False)