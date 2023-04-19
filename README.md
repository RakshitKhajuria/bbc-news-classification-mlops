# BBC - NEWS - CLASSIFICATION -MLOPS

Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.

Class Labels: 5 (business, entertainment, politics, sport, tech)

# Dataset Discription: 


[BBC Datasets Descrition](http://mlg.ucd.ie/datasets/bbc.html) 

[Dataset](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip)

# Files Description
* dataset/data_files: Data folders each containing several news txt files

* dataset/dataset.csv: csv file containing "news" and "type" as columns. "news" column represent news article and "type" represents news category among business, entertainment, politics, sport, tech.

* model/get_data.py: To gather all txt files into one csv file contianing two columns("news","type"). After successfull execution it will create dataset.csv file in dataset folder. 

* model/model.py: preprocessing, tf-idf feature extraction and model buildind and evaluation stuff

* model/test.ipynb: jupyter notebook 


# Method

Divided the feature extracted dataset into two parts train and test set. Train set contains 1780 examples and Test set contains 445 examples. 

# Result

Below table shows the result on test set

# 1. TfidfVectorizer + Model

| MODEL                  | ACCURACY | PRECISION | RECALL MACRO | F1 MACRO |
|------------------------|----------|-----------|--------------|----------|
| Random Forest          | 0.938889 | 0.944748  | 0.935345     | 0.938332 |
| Decision Tree          | 0.797222 | 0.795641  | 0.789634     | 0.790323 |
| Logistic Regression    | 0.963889 | 0.965929  | 0.960976     | 0.962801 |
| K-Neighbors Classifier | 0.955556 | 0.954670  | 0.955355     | 0.954221 |
| XGBClassifier          | 0.933333 | 0.937157  | 0.929557     | 0.932015 |
| CatBoosting Classifier | 0.961111 | 0.961879  | 0.960626     | 0.960915 |
| AdaBoost Classifier    | 0.766667 | 0.802877  | 0.753030     | 0.761275 |
| MultinomialNB          | 0.966667 | 0.969726  | 0.965025     | 0.966665 |
| RidgeClassifier        | 0.969444 | 0.969874  | 0.968333     | 0.968658 |
| SGDClassifier          | 0.972222 | 0.972638  | 0.971150     | 0.971473 |

