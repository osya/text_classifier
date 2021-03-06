#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import zipfile
import nltk
from nltk.stem.porter import PorterStemmer

zf = zipfile.ZipFile('data/smsspamcollection.zip')
df = pd.read_csv(zf.open('SMSSpamCollection'), sep='\t', names=['Status', 'Message'])
df['Status_num'] = pd.get_dummies(df['Status'])['ham']

df_x, df_y = df['Message'], df['Status_num']
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1)


def stem_tokens(tokens, stemmer):
    stemmed = set()
    for item in tokens:
        stemmed.add(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, PorterStemmer())
    return stems

cv = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 1), min_df=1, stop_words='english')
# cv = TfidfVectorizer(ngram_range=(1, 1), min_df=1, stop_words='english')
# cv = CountVectorizer(ngram_range=(1,1), stop_words='english')

x_train_dtm = cv.fit_transform(x_train)

model = MultinomialNB()
# model = LogisticRegression(penalty='l2', C=1)
# model = DecisionTreeClassifier(random_state=0, max_depth=2)
# model = KNeighborsClassifier()
model.fit(x_train_dtm, y_train)

x_test_dtm = cv.transform(x_test)
y_pred = model.predict(x_test_dtm)

print('Accuracy is %2.2f' % accuracy_score(y_test, y_pred))
print('ROC AUC score is %2.2f' % roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()

# Export Decision tree in a file
# export_graphviz(model)
