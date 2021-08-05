#!/usr/bin/env python
import sys
data_path=sys.argv[1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


tr = pd.read_csv(data_path+'/train.csv')
tr = tr.replace(np.nan, '', regex=True)
tr


#Cleaning data(removing punctuations, stop words) and tokenize for keywords anf text fields
def clean_tokenize(dframe):
    #First we remove punctuations from the tweet text
    x = dframe.apply(lambda x : ''.join([words for words in x if words not in string.punctuation])) 
    #re.split('\W',tr_clean['text'][3])
    # them we remove the stopwords by first coverting the text to word tokens and removing the stop words
    x = x.apply(lambda x : ' '.join([w for w in word_tokenize(x) if not w.lower() in stop_words]))
    # then we good word normlization
    lem = WordNetLemmatizer()
    x = x.apply(lambda x : ' '.join([lem.lemmatize(w,'v') for w in word_tokenize(x)]))
    return x.tolist()
#using Multinomail naive bayes classifier to model the features
def multinomailNb(model,inp_test,out_test):  
    predictions = model.predict(inp_test)
    print(confusion_matrix(out_test,predictions))


clean_text = clean_tokenize(tr['text']) 
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
tfidf_model = tfidfvectorizer.fit_transform(clean_text)



inp_train, inp_val, out_train, out_val = train_test_split(tfidf_model, np.array(tr['target']), test_size=0.2)
model = MultinomialNB()
out = model.fit(inp_train,out_train)


train_val = multinomailNb(out,inp_val,out_val)


# Predict for test set
te = pd.read_csv(data_path+'/test.csv')
clean_text = clean_tokenize(te['text']) 
tfidf_test = tfidfvectorizer.transform(clean_text)


train_test = model.predict(tfidf_test)




print(train_test[1:100])



#checking the obtained output test values   **********remove
f = open(data_path+'/test.csv')
n=834
print(f.readlines()[n],train_test[n])





