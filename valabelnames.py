# labelnames = ['vaoddistcc', 'vaoddistccph', 'vaoddistsc', 'vaoddistscph', 'vaodnearcc', 'vaodnearsc', 'vaosdistcc', 'vaosdistccph', 'vaosdistsc', 'vaosdistscph','vaosnearcc', 'vaosnearsc'] 
# 'near' labels ('vaosnearsc', 'vaosnearcc', 'vaodnearcc', 'vaodnearsc') were hardly present so we removed them

labelnames = ['vaoddistcc', 'vaoddistccph', 'vaoddistsc', 'vaoddistscph', 'vaosdistcc', 'vaosdistccph', 'vaosdistsc', 'vaosdistscph'] 
keynameslist = []
for name in labelnames: 
    keynameslist.extend(['B-' + name, 'I-' + name])
tag2id = {tag: id for id, tag in enumerate(keynameslist, start = 1)}
tag2id['O'] = 0
id2tag = {id: tag for tag, id in tag2id.items()}
inferenceid2tag = {"LABEL_"+str(id): tag[2:] for tag, id in tag2id.items()}
print(inferenceid2tag)
