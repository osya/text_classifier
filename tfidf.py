#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

docA = "the cat sat on my face"
docB = "the dog sat on my bed"

bowA = docA.split(" ")
bowB = docB.split(" ")

wordSet = set(bowA).union(bowB)

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

# print(pd.DataFrame([wordDictA, wordDictB]))

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word]=count/float(bowCount)
    return tfDict

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word]+=1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))

    return idfDict


def computeTFIDF(tfBow, idfs):
    tfIdf = {}
    for word, val in tfBow.items():
        tfIdf[word] = val * idfs[word]
    return tfIdf

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)
idfs = computeIDF([wordDictA, wordDictB])
tfIdfBowA = computeTFIDF(tfBowA, idfs)
tfIdfBowB = computeTFIDF(tfBowB, idfs)


print(pd.DataFrame([tfIdfBowA, tfIdfBowB]))
