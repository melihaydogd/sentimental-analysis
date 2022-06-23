import pickle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Used for calculating class specific accuracies
def manipulate(vector, index, zero_index):
    new = np.array(vector)
    new[index] = 1
    new[zero_index] = 0
    return new

# train data is loaded from pickle
filename = "train_data.pickle"
pickleFile = open(filename, 'rb')
train_data = pickle.load(pickleFile)
pickleFile.close()
# train labels are loaded from pickle
filename = "train_labels.pickle"
pickleFile = open(filename, 'rb')
train_labels = pickle.load(pickleFile)
pickleFile.close()
# validation data is loaded from pickle
filename = "validation_data.pickle"
pickleFile = open(filename, 'rb')
validation_data= pickle.load(pickleFile)
pickleFile.close()
# validation labels are loaded from pickle
filename = "validation_labels.pickle"
pickleFile = open(filename, 'rb')
validation_labels = pickle.load(pickleFile)
pickleFile.close()
# vectorizer is loaded from pickle
filename = "vectorizer.pickle"
pickleFile = open(filename, 'rb')
vectorizer = pickle.load(pickleFile)
pickleFile.close()

target_names = ["Negative", "Neutral", "Positive"] # review types

model_names = [] # model names
accuracies = [] # accuracies will be stored
models = [] # models will be stored

def gaus():
    gaussian = GaussianNB() # Gaussian model is created
    gaussian.fit(train_data.toarray(), train_labels) # Gaussian model train is handled
    pred = gaussian.predict(validation_data.toarray())  # Gaussian model predictions are done
    accuracy = accuracy_score(validation_labels,pred) # accuracy score is calculated
    accuracies.append(accuracy)
    model_names.append("Naive Bayes")
    models.append(gaussian)
    print("Naive Bayes overall accuracy: " + str(accuracy)) 
    print(classification_report(validation_labels, pred, target_names=target_names))
    print()

def log():
    logreg = LogisticRegression(solver= 'newton-cg', C= 1) # Logistic regression model is created
    logreg.fit(train_data, train_labels) # Logisted regression model train is handled
    pred = logreg.predict(validation_data) # Logistic regression predictions are done
    accuracy= accuracy_score(validation_labels,pred) # Accuracy is calculated
    accuracies.append(accuracy)
    model_names.append("Logistic Regression")
    models.append(logreg)
    print("Logistic Regression overall accuracy: " + str(accuracy)) 
    print(classification_report(validation_labels, pred, target_names=target_names))
    print()

def forest():
    randomForest =RandomForestClassifier(n_estimators=150) # Random forest model is created
    randomForest.fit(train_data, train_labels) # Random forest model train is handled
    pred = randomForest.predict(validation_data) # Random forest model predictions are handled
    accuracy = accuracy_score(validation_labels,pred) # Random forest model accuracy calculated
    accuracies.append(accuracy)
    model_names.append("Random Forest")
    models.append(randomForest)
    print("Random Forest Classifier overall accuracy: " + str(accuracy))
    print(classification_report(validation_labels, pred, target_names=target_names))
    print()

def svm():
    svc = SVC(C=1)  # SVC initialized 
    svc.fit(train_data, train_labels)  # model train
    pred = svc.predict(validation_data) # SVC model predictions handled
    accuracy= accuracy_score(validation_labels,pred) # SVC accuracy calculated
    accuracies.append(accuracy)
    model_names.append("SVC")
    models.append(svc)
    print("Support Vector Classifier overall accuracy: " + str(accuracy))
    print(classification_report(validation_labels, pred, target_names=target_names))
    print()

# gaus()
log()
# forest()
svm()

# Bar Chart
plt.bar(model_names, accuracies)
plt.savefig('modelschart.jpeg')

# Finding model that gives maximum accuracy
max_accuracy = max(accuracies)
index = accuracies.index(max_accuracy)
print("Maximum accuracy is " + str(max_accuracy) + ". It is achieved by " + model_names[index] + ".")

# Best model which is logistic regression and its vectorizer is pickled
filename = 'step2_model_LearningFromHata.pkl'
outfile = open(filename, 'wb')
pickle.dump([models[index], vectorizer], outfile)
outfile.close()  