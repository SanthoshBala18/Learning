import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from collections import Counter


TRAIN_DIR = "train-mails"
TEST_DIR = "test-mails"

def make_dictionary(foldername):
    files=[os.path.join(foldername,each) for each in os.listdir(foldername)]
    allwords=[]
    for file in files:
        with open(file) as read:
            for eachline in read:
                words = eachline.split()
                allwords+=(words)

    dictonary = Counter(allwords)
    keys = list(dictonary)
    for each in keys:
        if each.isalpha() == False:
            del dictonary[each]
        if len(each) == 1:
            del dictonary[each]

    return dictonary.most_common(3000)

def extract_feature_labels(foldername,dictionary):
    files=[os.path.join(foldername,each) for each in os.listdir(foldername)]
    feature_matrix = np.zeros((len(files),3000))
    train_labels = np.zeros(len(files))
    doc_ID = 0
    for eachfile in files:
        with open(eachfile) as read:
            for i,eachline in enumerate(read):
                if i==2:
                    words = eachline.split()
                    for word in words:
                        word_id=0
                        for j,d in enumerate(dictionary):
                            if d[0] == word:
                                word_id=j
                                feature_matrix[doc_ID,word_id] = words.count(word)
            train_labels[doc_ID] = 0
            filename = eachfile.split('/')[-1]
            if filename.startswith("spmsg"):
                train_labels[doc_ID] = 1
        doc_ID+=1
    return feature_matrix,train_labels


if __name__ == "__main__":
    common_words = make_dictionary(TRAIN_DIR)

    x_train,y_train = extract_feature_labels(TRAIN_DIR,common_words)
    x_test,y_test = extract_feature_labels(TEST_DIR,common_words)
    x_train = x_train[len(x_train)/10]
    y_train = y_train[len(y_train)/10]
    model = GaussianNB()
    
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    print("Accuracy score:",(acc*100))
    print(classification_report(y_test,y_pred))





















