import json
from seqeval.metrics import accuracy_score, classification_report, f1_score
import spacy
from spacy.gold import biluo_tags_from_offsets
from utils import MyLogger

'''
This script imports two jsonl files from prodigy, converts to format that works with seqeval, 
and runs the classification report comparing the two. 
it is labelset agnostic - presuming the two files improted from prodigy have the same labelset, it doesn't matter what they are
'''


def jsontobiotags(jsonlfilepath, startline=0): 
    result = []
    with open(jsonlfilepath) as f:
        for line in f: 
            result.append(json.loads(line))
            
    listoftags = []
    nlp = spacy.blank("en")
    for item in result: 
        text = item['text']
        spans = item['spans']
        doc = nlp(text)
        offsets = [(span["start"], span["end"], span["label"]) for span in spans]
        biluo_tags = biluo_tags_from_offsets(doc, offsets)
        doc.ents = [doc.char_span(start, end, label) for start, end, label in offsets]
        iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O" for t in doc]
        listoftags.append(iob_tags)
    return listoftags[startline:]

startline = 0
correctedtagpath = 'regexmodel-corrected-out.jsonl'  # file path for the .jsonl file produced after making corrections to the model predictions with prodigy
correctedtags = jsontobiotags(correctedtagpath, startline = startline)
print("corrected tags come from file"+correctedtagpath)

modeltagpath = 'regexconsult-out.jsonl' # file path for the .jsonl file with the model's original predictions (should be the output paths from exporttoprodigy_bert.py or exporttoprodigy_regex.py)
modeltags = jsontobiotags(modeltagpath, startline = 0)
print("original tags come from file"+modeltagpath) 

print("tags in each list:", len(correctedtags), len(modeltags)) 

newmodeltags = []
newcorrectedtags = []
for modeltag, correctedtag in zip(modeltags[0:len(correctedtags)], correctedtags): 
    if len(modeltag) != len(correctedtag): 
        #print(len(modeltag), len(correctedtag))
        pass 

    else: 
        newmodeltags.append(modeltag)
        newcorrectedtags.append(correctedtag)

print("tags remaining in each list:", len(newmodeltags), len(newcorrectedtags)) 

print("Validation Accuracy: " + str(accuracy_score(correctedtags, modeltags[0:len(correctedtags)])))
print(classification_report(newcorrectedtags, newmodeltags))

