# Sentimental Analysis

This project is made for CMPE 462 Machine Learning course. By using IMDB comments, a model to predict the sentiment of a given sentence trained.

# main.py

It reads the files from TRAIN folder. By using TfidfVectorizer creates a feature matrix. 
Also, it find the labels of the files. Then, it pickles training data, validation data and TfidfVectorizer.
TfidfVectorizer is stored because when there is new data TfidfVectorizer should transfrom them.

In this file, there is some functions that are not used. These are our previous work that did not get
good results. It is explained in the report.

It run by this command:
	python main.py

Then, pickle files will be created.


# models.py

Open the pickles files that were created by main.py. Models are applied to the data. It saves the accuracy plot.
In that way, we can see which model gives highest accuracy. It also prints out accuracies and classification reports
of each model.

It pickles the model that gives the highest accuracy and TfidfVectorizer together. This is our
step2_model_LearningFromHata.pkl. 

It run by this command:
	python models.py

Then, pickle file will be created.


# 462project_step2_LearningFromHata.py

Takes two command line arguments. One of them step2_model_LearningFromHata.pkl and other one is dataset folder name.
It reads the files in dataset folder respectively and transforms them by TfidfVectorizer that comes with 
step2_model_LearningFromHata.pkl file. Then, predicts the results. At the end, it prints out the predictions as a string.

It run by this command:
	python 462project_step2_LearningFromHata.py step2_model_LearningFromHata.pkl <folder-name>

Example:
	python 462project_step2_LearningFromHata.py step2_model_LearningFromHata.pkl VAL