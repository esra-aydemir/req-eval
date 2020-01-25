import spacy
import re
import pandas as pd
import pickle
import inflect

def preprocess(sentence):
    ref = sentence.split('<referential>')[1].split('</referential>')[0]
    text = re.sub(r'<referential>(.*?)</referential>',r'\1',sentence)
    chunks = []
    doc = nlp(text)
    for d in doc:
        if d.text == ref:
            ref = d
            break
    for chunk in doc.noun_chunks:
        chunks.append(chunk)
    chunksadp = [[x for x in chunks[0]]]
    for i in range(1,len(chunks)):
        end = chunksadp[-1][-1].i
        chunkOld = chunksadp[-1]
        chunkNew = [x for x in chunks[i]]
        if(chunkNew[0].i==end+2 and doc[i+1].pos_ == 'ADP'):
            chunkOld+=[doc[i+1]] + chunkNew
            chunksadp[-1] = chunkOld
        else:
            chunksadp.append(chunkNew)
    ng = []
    before = True
    refNG = ref.text
    for i in range(len(chunksadp)):
        c = chunksadp[i]
        if(ref.text in [x.text for x in c]):
            refNG = c
            before = False
        elif before:
            ng.append({'NG':c,'status':'beforeRef'})
        else:
            ng.append({'NG':c,'status':'afterRef'})
    return [text,doc,ref,refNG,ng]

def firstNG(cs):
    return ' '.join([x.text for x in cs[0]['NG']]).strip()

def processMulti(label,sentence):
    labels_ = [label+'-'+x for x in re.findall(r"<referential id=\"(.+?)\">",sentence)]
    doc = nlp(sentence)
    res = doc[0].text
    for i in doc.noun_chunks:
        txt = ' '.join([x.text for x in i])
        if 'referential' not in txt:
            res = txt
            break
    return [res,labels_]

def useNGBeforeRef(candidates):
    filteredCandidates = []
    for c in candidates:
        if c['status'] == 'beforeRef':
            filteredCandidates.append(c)
    if(len(filteredCandidates) == 0):
        return candidates
    return filteredCandidates                

def pluralityCheck(ref,candidates):
    refPlural = False
    if ref.text in pluralRefs:
        refPlural = True
    filteredCandidates = []
    for c in candidates:
        head = findHead(c['NG'])
        if((not head.text.isupper()) and inflect.singular_noun(head.text)):
            #case plural
            if(refPlural):
                filteredCandidates.append(c)
        else:
            #case singular
            if(not refPlural):
                filteredCandidates.append(c)
    if(len(filteredCandidates) == 0):
        return candidates
    return filteredCandidates

def findHead(ng):
    for x in ng:
        if(x.head not in ng) and (x.pos_ in mainNounPoses):
            return x
    return ng[0]   

def headDepCheck(ref,candidates):
    refDep = ref.dep_.replace('pass','')
    filteredCandidates = []
    heads = []
    for c in candidates:
        head = findHead(c['NG'])
        headDep = head.dep_.replace('pass','')
        if(headDep == refDep):
            filteredCandidates.append(c)
    if(len(filteredCandidates) == 0):
        return candidates
    return filteredCandidates            

def sameTokenInCandidateAndRefNG(refng,candidates):
    refNG = refng.copy()
    if(len(refNG)<2):
        # case refNG only contains ref
        return candidates
    filteredCandidates = []
    for c in candidates:
        for t in c['NG']:
            if(t in refNG and (t.pos_ in mainNounPoses or t.pos_ == 'ADJ')):
                filteredCandidates.append(c)
                break
    if(len(filteredCandidates) == 0):
        return candidates
    return filteredCandidates            

def sameNounInCandidateAndAfterRef(doc,ref,candidates):
    candidatesBeforeRef = []
    for c in candidates:
        if(c['status'] =='beforeRef'):
            candidatesBeforeRef.append(c)
    if(len(candidatesBeforeRef) == 0):
        return candidates
    filteredCandidates = []
    tokensAfterRef = doc[ref.i:]
    for c in candidatesBeforeRef:
        for t in c['NG']:
            if(t.pos_ in mainNounPoses or t.pos_ == 'ADJ'):
                if(t in tokensAfterRef):
                    filteredCandidates.append(c)
                    break
    if(len(filteredCandidates) == 0):
        return candidates
    return filteredCandidates            


AMB = 'AMBIGUOUS'
UNAMB = 'UNAMBIGUOUS'
nlp = spacy.load("en_core_web_sm")
refs = ['It', 'he', 'him', 'it', 'its', 'their', 'them', 'they']
pluralRefs = ['their','them','they']
mainNounPoses = ['NOUN','NUM','PRON','PROPN','X']
inflect = inflect.engine()
training_set = pd.read_csv('training_set.csv')

result = []
labels = []
ds = []
for m in range(len(training_set)):
    label = training_set.iloc[m][0]
    sentence = training_set.iloc[m][1]
    d = {}
    if('<referential>' not in sentence):
        [res_,labels_] = processMulti(label,sentence)
        for l in labels_:
            labels.append(l)
            result.append(res_)
        d['cause'] = 'multi ref'
        continue;
    labels.append(label)
    [text,doc,ref,refNG,candidatesList] = preprocess(sentence)
    candidates = candidatesList.copy()
    ds.append(d)
    d = {'text':text}
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 ng amb'
        continue;
    candidates = useNGBeforeRef(candidates)
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 use of ng before ref'
        continue;
    candidates = pluralityCheck(ref,candidates)
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 plurality match'
        continue;
    candidates = headDepCheck(ref,candidates)
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 same head dep'
        continue;
    candidates = sameTokenInCandidateAndRefNG(refNG,candidates)
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 sameTokenInCandidateAndRefNG'
        continue;
    candidates = sameNounInCandidateAndAfterRef(doc,ref,candidates)
    if(len(candidates) == 1):
        result.append(firstNG(candidates))
        d['cause'] = '1 sameNounInCandidateAndAfterRef'
        continue;
    d['cause'] = 'non'
    result.append(firstNG(candidates))


df = pd.DataFrame({'sent_id': labels,
'resolution':result})

df.to_csv('disambiguation_out.csv',index=False)