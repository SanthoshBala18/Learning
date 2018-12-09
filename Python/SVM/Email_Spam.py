# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 00:31:23 2018

@author: Sandy

SPAM Email Classification model using SVM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import re
from nltk.stem import PorterStemmer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

def preprocess_email(email):
    
    #Convert the email to lower case
    email = email.strip().lower()
    
    #Using regex to remove all htmp tags
    clean_html = re.compile('<.*?>')
    email = re.sub(clean_html,'',email)
    
    #Converting all urls to "httpaddr"
    normalize_url = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+?/')
    email = re.sub(normalize_url,'httpaddr',email)
  
    #Converting all email address to "emailaddr"
    normalize_url = re.compile(r'[\w\.-]+@[\w\.-]+')
    email = re.sub(normalize_url,'emailaddr',email)
    
    #Replacing all numbers with text number
    normalize_number = re.compile(r'\d+')
    email = re.sub(normalize_number,'number',email)

    #Replacing all numbers with text number
    normalize_dollar = re.compile(r'[$]')
    email = re.sub(normalize_dollar,'dollar ',email)
    
    #Replacing all puncuation and special characters by white space
    normalize_punctuations = re.compile(r'[^0-9a-z\s]')
    email = re.sub(normalize_punctuations,'',email)
    
    #Replacing all tab and newline by white space
    normalize_punctuations = re.compile(r'[\t\n]')
    email = re.sub(normalize_punctuations,' ',email)

    #Stem the words
    email_words = email.split(' ')
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in email_words]
    stemmed_words = [word.strip() for word in stemmed_words]
    email = ' '.join(stemmed_words)
    
    return email
#Reading the vocabulary list and putting it in dictionary
def read_vocablist():
    fileContent = ""
    with open("vocab.txt","r") as f:
        fileContent = f.read().strip()
    vocab_pair = fileContent.split("\n")
    vocab_dict ={}
    for each in vocab_pair:
        value,key = each.split("\t")
        vocab_dict[key] = value
    
    return (vocab_dict)

#Tokenizing - Converting each email into corresponding number from vocabulary list
def convert_email(email,vocab_dict):
    words = email.split(' ')
    words = [vocab_dict[word] for word in words if word in vocab_dict.keys()]
    email = ' '.join(words)
    return (email)
    
#Convert the tokenized email into a feature vector using vocab_dict
def convert_into_feature(tokenized_email,vocab_dict):
    feature_vector = []
    for each in vocab_dict.values():
        if each in tokenized_email:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    
    return feature_vector
    
'''fileContent = ""
with open("emailSample1.txt","r") as f:
    fileContent = f.read()
    
email = preprocess_email(fileContent)
vocab_dict = read_vocablist()
tokenized_email = convert_email(email,vocab_dict)
convert_into_feature(tokenized_email,vocab_dict)'''


mat = scipy.io.loadmat("spamTrain")
mat_test = scipy.io.loadmat("spamTest")
X = mat['X']
y = mat['y']
X_test = mat_test['Xtest']
y_test = mat_test['ytest']

model = SVC()
model.fit(X,y)
y_pred = model.predict(X)
y_pred_test = model.predict(X_test)

print(classification_report(y_pred,y))
print(classification_report(y_pred_test,y_test))