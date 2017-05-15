#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
default_tagger = nltk.DefaultTagger('NN')
print (default_tagger.evaluate(brown_tagged_sents))
