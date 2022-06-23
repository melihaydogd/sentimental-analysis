import re
import os
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Stopwords are collected from txt file
with open("stopwords3.txt") as f:
    stopWords = f.read().splitlines()
stopWords = set(stopWords)

def create_data_with_vectorizer(vectorizer, file_name):
    '''
    return: data set values and labels for datasets
    '''
    text_data = []  # text list
    labels = []  # label list
    path_name = "./" + file_name
    files = os.listdir(path_name) 
    for file in files:
        with open(os.path.join(path_name, file), 'r', encoding="latin1") as f: # file is opened
            text_data.append(f.read().lower()) # text data of the file is collected
            review_type = file[file.find(".")-1:file.find(".")] # review type is etracted from file name
            # correct review type is appended to labels list
            if review_type == "N":
                labels.append(-1)
            elif review_type == "Z":
                labels.append(0)
            else:
                labels.append(1)
    # vectorizing operation done
    if file_name == "TRAIN":
        text_data = vectorizer.fit_transform(text_data)
    elif file_name == "VAL":
        text_data = vectorizer.transform(text_data)

    return text_data,labels

if __name__ == '__main__':
    vectorizer = TfidfVectorizer(stop_words=stopWords, lowercase=True, use_idf=True, smooth_idf=True, max_features=4000) # initialized vectorizer 
    train_data_vectorizer,train_labels_vectorizer = create_data_with_vectorizer(vectorizer,"TRAIN") # train data is created
    # train data vectorizer is pickled
    filename = 'train_data.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(train_data_vectorizer, outfile)
    outfile.close()
    # train labels list is pickled
    filename = 'train_labels.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(train_labels_vectorizer, outfile)
    outfile.close()   
    # vectorizer pickled
    filename = 'vectorizer.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(vectorizer, outfile)
    outfile.close()  

    # validation data and labels created
    validation_data_vectorizer,validation_labels_vectorizer = create_data_with_vectorizer(vectorizer,"VAL")
    # validation data is pickled
    filename = 'validation_data.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(validation_data_vectorizer, outfile)
    outfile.close()
    # validation labels is pickled
    filename = 'validation_labels.pickle'
    outfile = open(filename, 'wb')
    pickle.dump(validation_labels_vectorizer, outfile)
    outfile.close()  

