#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['Status', 'Message'])
df['Status'] = pd.get_dummies(df['Status'])['ham']

df_x, df_y = df['Message'], df['Status']
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

cv = TfidfVectorizer(ngram_range=(1,1), min_df=1, stop_words='english')
# cv = CountVectorizer(ngram_range=(1,1), stop_words='english')

x_train_cv = cv.fit_transform(x_train)

model = MultinomialNB()
# model = LogisticRegression(penalty='l2', C=1)
model.fit(x_train_cv, y_train)

x_test_cv = cv.transform(x_test)
pred = model.predict(x_test_cv)

print('Accuracy is %2.2f' % accuracy_score(y_test, pred))
