import pickle
import os
import sys
from sklearn.linear_model import  LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Model and its vectorizer loaded from pickle
filename = sys.argv[1]
pickleFile = open(filename, 'rb')
model, vectorizer = pickle.load(pickleFile)
pickleFile.close()

review_type = ["N", "Z", "P"] # Review types list 

folder_name = sys.argv[2]
files = os.listdir("./" + folder_name)

if ".DS_Store" in files: # if file is dsstore which is created in macos's than it is removed
    files.remove(".DS_Store")

text_data = []  # text list
labels = []  # label list

for i in range(len(files)):
   
    for review in review_type:
        try:
            f = open("./" + folder_name + "/" + str(i) + "_" + review + ".txt",  "r", encoding='latin-1') # file is opened
            text_data.append(f.read().lower()) # text data of the file is collected
            # correct review type is appended to labels list
            if review == "N":
                labels.append(-1)
            elif review == "Z":
                labels.append(0)
            else:
                labels.append(1)
        except FileNotFoundError:
            pass

text_data = vectorizer.transform(text_data) # transform handled

predict = model.predict(text_data) # model predictions are made

result = ""
for item in predict:
    if item == -1:
        result+="N"
    elif item == 0:
        result+="Z"
    else:
        result+="P"

print(result)