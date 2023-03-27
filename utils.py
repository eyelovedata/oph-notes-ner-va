import pandas as pd
import re 
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import ast
import string 
import math
import torch 
import json 

#logging function 
def MyLogger(output_file_path): 
    import logging 
    import sys
    logger=logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    output_file_handler=logging.FileHandler(output_file_path, mode='w')
    stdout_handler=logging.StreamHandler(sys.stdout) 

    logger.addHandler(output_file_handler) 
    logger.addHandler(stdout_handler) 
    return logger 

##turns the oph-ner-preprocess02-tokenizeandlabel.ipynb util functiosn into a script

class OphNERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

##Preprocessing Utils

def gettokenspanlist(note): 
    t = TreebankWordTokenizer()
    tokenspanlist=list(t.span_tokenize(note))
    return tokenspanlist
    
#find the visual acuity portion of the note 
def findvaheader(note): 
    regex=re.compile(r"(?i)Visual\sAcuity[:]?")
    vaspanlist=[]
    for m in regex.finditer(note): 
        vaspanlist.append((m.start(), m.end()))
    try: return vaspanlist[0][0]
    except: return 0
    
def findsleheader(note): 
    regex=re.compile(r"(?i)S(lit)?(\s|-)?L(amp)?(\s?E(xam)?)?(\s|:|-)")
    slespanlist=[]
    for m in regex.finditer(note): 
        slespanlist.append((m.start(), m.end()))
    try: return slespanlist[-1][0]
    except: return 0
    
def findextheader(note): 
    regex=re.compile(r"(?i)External")
    extspanlist=[]
    for m in regex.finditer(note): 
        extspanlist.append((m.start(), m.end()))
    try: return extspanlist[0][0]
    except: return 0

def findllheader(note): 
    regex=re.compile(r"(?i)((l(ids)?\s?(\/|&|(\sand\s))\s?l(ashes)?))") 
    llspanlist=[]
    for m in regex.finditer(note): 
        llspanlist.append((m.start(), m.end()))
    try: return llspanlist[0][0]
    except: return 0
    
#build function that turns cell into a regex and searches text for it, returning the span of the match 
def findlabel(regextext, note): 
    spanlist=[]
    sleposition = findsleheader(note) 
    vaposition = findvaheader(note) 
    llposition = findllheader(note)
    startposition = max(sleposition, vaposition, llposition)
    try: 
        regextext=str.strip(regextext)
        regex=re.compile(re.escape(" "+regextext)) #regex escape? 
    except: 
        return []
    for m in regex.finditer(note): 
        if m.start()+1 >= startposition: 
            spanlist.append((m.start()+1, m.end()))
    return spanlist 
    

#now we have to match the spans to actual locations of the tokens 
#what do we want to return? the index of the token which starts the match, and the indices of the subsequent tokens
#version 2, what is the beginning and end of the spanlist doesn't match the begin and end of a token
def findmatchingtoken(tokenspanlist, spanlist): 
    '''
    inputs: 
        tokenlist: a list of span tuples identifying start and ends of tokens
        spanlist: a list of span tuples identifying starts and ends of matches. 
            Each tuple could span multiple tokens. Could be multiple tuples indicating multiple matches. 
        
    returns: 
        a list of tuples indicating start and end indices for matches?
    '''
    
    matchindexlist=[]
    if len(spanlist)==0: 
        return []
    else: 
        for match in spanlist: 
            matchbeg=match[0]
            matchend=match[1]
            #search for match beginning 
            for i in range(len(tokenspanlist)): 
                tokenspan=tokenspanlist[i]
                if tokenspan[0] <= matchbeg <= tokenspan[1]: 
                    #save the beginning of the token here 
                    begindex=i
                    #search for match end (starting with match beginning): 
                    j=begindex
                    while j < len(tokenspanlist): 
                        tokenspan=tokenspanlist[j]
                        if tokenspan[0] <= matchend <= tokenspan[1]: 
                            #save the end of the token here 
                            endindex=j
                            break #stop searching if beg and end found 
                        j=j+1
                    try: 
                        matchindexlist.append((begindex,endindex))
                    except: 
                        print(match, tokenspanlist)
        if len(matchindexlist)==0: 
            return []
        else: return matchindexlist
    

#then given a series of indices which have matches, return the whole sequence of labels
#how to resolve conflicts? one token span might match more than one label 
#simple version may not do any significant conflict resolution - i.e., pick a random version that works 
def returnlabels(matchindexlist, tokenlistlabels, name): 
    '''
    inputs: 
        matchindexlist: a list of indices which match the named entity 
        tokenlistlabels: a list of labels (in progress) which need to be labeled with the match 
        name: the name of the entity type 
    '''
    if len(matchindexlist)>0: 
        for match in matchindexlist:
            matchstart=match[0]
            matchend=match[1]
            if tokenlistlabels[matchstart]=='O': #if first available match is "free" or unassigned
                tokenlistlabels[matchstart]='B-'+name #label entity start 
                if matchend>matchstart: #for multiple token matches, label entity continuation 
                    for i in range(matchstart+1, matchend+1): 
                        tokenlistlabels[i]='I-'+name
                break 
            else: #if first available match is already assigned, go to the next one 
                continue 
    return tokenlistlabels 

def initializetokenlistlabels(tokenspanlist): 
    tokenlistlabels=['O']*len(tokenspanlist)
    return tokenlistlabels 
    
def gettokenlistlength(tokenspanlist):
    tokenspanlist=ast.literal_eval(tokenspanlist)
    return len(tokenspanlist)
    
def checkfornomatches(doclength, tokenlistlabels): 
    tokenlistlabels=ast.literal_eval(tokenlistlabels)
    if tokenlistlabels == ["O"]*doclength: 
        return True 
    else: 
        return False 
        
def countmatches(tokenlistlabels): 
    tokenlistlabels=ast.literal_eval(tokenlistlabels)
    return sum('B-' in s for s in tokenlistlabels)
    
#split long documents into shorter ones of specified size 
#ready to input into sklearn train_test_split for dev splitting off 
def splitdoc(tokenlist, tokenlistlabels, size=256):
    '''
    inputs: 
        tokenlist: a list of list of tokens for the documents
        tokenlistlabels: a corresponding list of list of token labels for the documents 
    
    returns: 
        list of lists containing the tokens, as well as a list of lists containing the tags
        ready         
        n.b. In the end we will treat each subdocument as an independent document 

    '''
    #master lists of lists of tokens and tags 
    tokens=[]
    tags=[]
    
    #iterate through each document 
    for i in range(len(tokenlist)): 
        doc=tokenlist[i]
        labels=tokenlistlabels[i]
        #split up each doc into a list of lists of tokens and tags 
        tokenlistbroken=[doc[x:x+size] for x in range(0, len(doc), size)]
        taglistbroken=[labels[x:x+size] for x in range(0, len(labels), size)]
        #tack on the lists of lists into the master lists of lists
        tokens.extend(tokenlistbroken)
        tags.extend(taglistbroken)
    
    return tokens, tags

#AK: rewrite splitdoc for consult_notes? so no tokenlist labels...?

#propagate labels through a document through its word pieces        
def fill_labels(doc_labels, doc_offset): 
    '''
    inputs: 
        encoding offset array for a document (after it passed through Bert WordPieceTokenizer)
        original labels for that document  
    outputs: 
        new set of labels for the WordPieced document, 
        consisting of original labels propagated from head of word to rest of word as appropriate
    '''
    arr_offset = np.array(doc_offset) #turns the list of tuples into an array, 512 rows by 2 columns
    df=pd.DataFrame(data=arr_offset, columns=["offset0", "offset1"])

    # set labels whose first offset position is 0 and the second is not 0, which means its the first subpart of a word
    selected_label_idx=(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)
    num_selected_labels=selected_label_idx.sum()
    #sometimes i have too many doc_labels for token heads, which I think happens when the doc gets truncated 
    #thus num_selected_labels tells me how many labels i need from the doc_labels 

    #initialize labels with Nan's 
    df["labels"]=np.nan
    #fill selected labels corresponding to wordpiece heads with the original labels 
    df.loc[selected_label_idx, "labels"] = doc_labels[0:num_selected_labels]
    #special tokens are filled in as 0  
    df.loc[(df["offset0"]==0) & (df["offset1"]==0), "labels"]=0
    #the only remaining tokens without labels are word pieces that don't start words
    #we fill those from the previous labels 
    #select rows that are odd (start tokens for entities) 
    startlabelidx=df.loc[(df["labels"] % 2 == 1)].index
    #select the following row but only if it needs filling 
    requiresfillidx=df.loc[startlabelidx+1, "labels"][df["labels"].isnull()].index
    #fill it with the prior row's values +1 (because it will be a "continue" token corresponding to prior start token)
    df.loc[requiresfillidx, "labels"]=list(df.loc[requiresfillidx-1, 'labels']+1)
    #now the only missing labels follow even numbers, so use usual ffill method.
    df.ffill(inplace=True)
    return list(df["labels"].astype(int))
    
#iterate through documents to propagate labels 
def encode_tags(tags, encodings, tag2id):
    #turns the labels into a list of lists of tag-id's 
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    #iterate through each document's labels and offset mapping
    i=0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        if i%10000==0: 
            print("processing subdocument", i) 
        encoded_labels.append(fill_labels(doc_labels, doc_offset))
        i=i+1
    return encoded_labels
    
def remove_unicode_specials(inputstring):
    newstring=inputstring.replace("\uFFFD", "")
    return newstring


#helper functions to do inference and export results to prodigy
def singledocinference(doc, classifier, inferenceid2tag): 
    '''runs a single document through inference and processes the output dictionary to work for prodigy''' 
    pipelineoutput=classifier(doc)
    docdict={} 
    docdict["text"]=doc
    spanlist=[] 
    for dictionary in pipelineoutput:
        dictionary["start"]=int(dictionary["start"])
        dictionary["end"]=int(dictionary["end"])
        #dictionary.pop('score', None)
        #dictionary.pop('word', None)
        dictionary["label"]=inferenceid2tag[dictionary['entity']]
        #dictionary.pop("entity")
        spanlist.append(dictionary)
    docdict["spans"]=spanlist
    return docdict 
    
def multipledocinference(doclist, classifier, outputjsonpath, inferenceid2tag): 
    '''Calls singledoc inference multiple times, to do multiple doc inference, 
    and saves multiline jsonl filepath for prodigy'''
    docdictlist=[]
    print(doclist)
    for doc in doclist: 
        docdict=singledocinference(doc, classifier, inferenceid2tag)
        #print(docdict)
        docdictlist.append(docdict)
    print('saving jsonl file to ', outputjsonpath)
    with open(outputjsonpath, 'w') as f:
        for item in docdictlist:
            f.write(json.dumps(item) + "\n")

def select_longest_regex_finding(regex_one, regex_two, note):
    """
    Return the longest regex match of the first finding of regex in note.

    Parameters
        regex_one: str, regular expression to search through note.
        regex_two: str, regular expression to search through note if we don't find any matches for regex_one.
        note: str, string to be searched through with regex.

    Returns
        str, longest regex match of the first finding of regex in note. If no
            regex matches are found, returns None.
    """
    if (regex_one != None and note != None):
        findings = re.findall(regex_one, note, flags=re.IGNORECASE)
        if (len(findings) > 0):
            return max(findings[0], key=len)
        else:
            if (regex_two != None):
                findings = re.findall(regex_two, note, flags=re.IGNORECASE)
                if (len(findings) > 0):
                    return max(findings[0], key=len)
                else:
                    return None
    else:
        return None


def find_sandwiched_regex_finding_indices(regex_curr, regex_next, note, final_entity_width):
    """
    Searches 'note' for the longest regex match of the regular expression
    regex_curr, and the longest regex match of regex_next that occurs after
    regex_curr's match. Returns the end index of regex_curr's match and
    the start index of regex_next's match. If 
    regex_next is set to None, returns the content after regex_curr's match
    to the end of note.
    
    Parameters
        regex_curr: str, a regular expression to search through the note
        regex_next: str, a regular expression to search through the note, or None
            if we want to search to the end of the note.
        note: str, a string that will be searched using the two regexes
        final_entity_width: int, specifying how many characters should be in the final entity

    Returns:
        int, int: index of the end of the longset regex match of regex_curr,
            index of the start of the longest regex match of regex_next after
            the end of regex_curr's match.
    """
    if (regex_curr != None and note != None):
        curr_findings = re.findall(regex_curr, note, flags=re.IGNORECASE)

        # Only proceed if we find regex matches
        if (len(curr_findings) > 0):
            # re.findall sometimes returns lists of strings, sometimes lists of tuples
            if (isinstance(curr_findings[0], str)):
                curr_heading = max(curr_findings, key=len)
            else:
                curr_heading = max(curr_findings[0], key=len)
            curr_heading_end_ind = note.index(curr_heading) + len(curr_heading)

            updated_note = remove_longest_regex_finding(regex_curr, note)

            # If regex_next != None, we are going to find the sandwiched finding
            if (regex_next != None):
                next_findings = re.findall(regex_next, updated_note, flags=re.IGNORECASE)

                if (len(next_findings) > 0):
                    if (isinstance(next_findings[0], str)):
                        next_heading = max(next_findings, key=len)
                    else:
                        next_heading = max(next_findings[0], key=len)
                    next_heading_start_ind = curr_heading_end_ind + updated_note.index(next_heading)
                else:
                    next_heading_start_ind = len(note)
            # if regex_next == None, we return the text up to the end of the note
            else:
                next_heading_start_ind = curr_heading_end_ind + final_entity_width

            return curr_heading_end_ind, next_heading_start_ind
                
    return None, None


def select_sandwiched_regex_finding(regex_curr, regex_next, note, final_entity_width):
    """
    Searches 'note' for the longest regex match of the regular expression
    regex_curr, and the lognest regex match of regex_next that occurs after
    regex_curr's match. Returns the content between the two matches. If 
    regex_next is set to None, returns the content after regex_curr's match
    to the end of note.
    
    Parameters
        regex_curr: str, a regular expression to search through the note
        regex_next: str, a regular expression to search through the note
        note: str, a string that will be searched using the two regexes
        final_entity_width: int, specifying how many characters should be in the final entity

    Returns:
        str, text between the longest regex matches of regex_curr and regex_next.
            If regex_next is None, returns the text from the end of regex_curr's match
            to the end of the string.
    """
    curr_heading_end_ind, next_heading_start_ind = find_sandwiched_regex_finding_indices(regex_curr, regex_next, note, final_entity_width)
    if (curr_heading_end_ind != None and next_heading_start_ind != None):
        finding = note[curr_heading_end_ind:next_heading_start_ind]
        if (finding.isspace()):
            return None
        return finding 
    return None

def remove_sandwiched_regex_finding(regex_curr, regex_next, note, final_entity_width):
    """
    Searches 'note' for the longest regex match of the regular expression
    regex_curr, and the lognest regex match of regex_next that occurs after
    regex_curr's match. Returns the content from the start of regex_next's match
    onwards.
    
    Parameters
        regex_curr: str, a regular expression to search through the note
        regex_next: str, a regular expression to search through the note
        note: str, a string that will be searched using the two regexes
        final_entity_width: int, specifying how many characters should be in the final entity

    Returns:
        str, text after longest regex match of regex_next
    """
    curr_heading_end_ind, next_heading_start_ind = find_sandwiched_regex_finding_indices(regex_curr, regex_next, note, final_entity_width)
    if (curr_heading_end_ind != None and next_heading_start_ind != None):
        finding = note[next_heading_start_ind:]
        return finding 
    return None

def remove_longest_regex_finding(regex, note):
    """
    Finds the longest match of 'regex' within 'note', and returns a modified
    version of 'note' with the sequence before and including the longest match
    removed.

    Parameters:
    - regex: str, a regular expression to search through note
    - note: str, a string that will be searched using regex

    Returns:
    - str, version of note with text up to and including the longest regex match
    removed. If no regex match is found, this returns note.
    """
    if (regex != None and note != None):
        findings = re.findall(regex, note, flags=re.IGNORECASE)
        if (len(findings) > 0):
            if (isinstance(findings[0], str)):
                longest_finding = max(findings, key=len)
            else:
                longest_finding = max(findings[0], key=len)                
            longest_match_len = len(longest_finding)
            if (longest_match_len > 0):
                start_ind = note.index(longest_finding)
                end_ind = start_ind + longest_match_len
                return note[end_ind:]
            else:
                return note
    return note

#build function that turns cell into a regex and searches text for it, returning the span of the match 
def findoffsetlabel(regextext, note, offset): 
    spanlist=[]
    if ((note != None) and len(note) > 0):
        try: 
            regextext=str.strip(regextext)
            regex=re.compile(re.escape(" "+regextext)) #regex escape? 
        except: 
            return []
        for m in regex.finditer(note): 
            spanlist.append((m.start()+1+offset, m.end()+offset))
    return spanlist 


def defaultlabels(doclength): 
    return '['+int(doclength)*"'O',"+']'