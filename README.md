# BBC - NEWS - CLASSIFICATION -MLOPS

## Table of contents

* [Demo](#Demo)
* [Dataset Discription](#Dataset-Discription)
* [Files Description](#Files-Description)
* [Method](#Method)
* [Result](#Result)
* [Run Locally](#Run-Locally)
* [Dockerizing Application](#Dockerizing-Application)
* [Deployment](#Deployment)

# Dataset Discription

Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.

Class Labels: 5 (business, entertainment, politics, sport, tech)

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

# Results

**Below table shows the result on test set**

## 1. TfidfVectorizer + Model

| **MODEL**              | **ACCURACY** | **PRECISION MACRO** | **RECALL MACRO** | **F1 MACRO** |
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

## 2. CountVectorizer + model

| **MODEL**              | **ACCURACY** | **PRECISION MACRO** | **RECALL MACRO** | **F1 MACRO** |
|------------------------|----------|-----------------|--------------|----------|
| Random Forest          | 0.952778 | 0.956768        | 0.949247     | 0.951832 |
| Decision Tree          | 0.794444 | 0.788847        | 0.787376     | 0.785466 |
| Logistic Regression    | 0.958333 | 0.959079        | 0.955287     | 0.956771 |
| K-Neighbors Classifier | 0.580556 | 0.824127        | 0.564493     | 0.589024 |
| XGBClassifier          | 0.947222 | 0.947193        | 0.944287     | 0.945144 |
| CatBoosting Classifier | 0.955556 | 0.957578        | 0.953577     | 0.955211 |
| AdaBoost Classifier    | 0.750000 | 0.768979        | 0.742884     | 0.747008 |
| MultinomialNB          | 0.975000 | 0.975004        | 0.974890     | 0.974578 |
| RidgeClassifier        | 0.916667 | 0.920257        | 0.910896     | 0.913691 |
| SGDClassifier          | 0.947222 | 0.947606        | 0.944972     | 0.945842 |

## 3. Hyperparameter Tuning 

| MODEL                  | ACCURACY | PRECISION MACRO | RECALL MACRO | F1 MACRO |
|------------------------|----------|-----------------|--------------|----------|
| Logistic Regression    | 0.975    | 0.98            | 0.97         | 0.97     |
| SGDClassifier          | 0.977    | 0.98            | 0.98         | 0.98     |

# Run Locally

Clone the project

```bash
  git clone https://github.com/Ryzxxl/bbc-news-classification-mlops.git
```

Go to the project directory

```bash
  cd bbc-news-classification-mlops
```

Install dependencies

```bash
  pip install -r requirements.txt
```

To run the **Flask App**
```bash
  python app_flask.py
```
To run the **Streamlit App**
```bash
  streamlit run application.py
```

# Dockerizing Application

```bash
  docker build -t appname:latest .
  docker run -p 8080:8080 appname:latest
```

# Deployment

## To deploy this project on EC2 with Docker

```bash
#optinal
    sudo apt-get update -y 
    sudo apt-get upgrade 

#Required
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker

```
## Configure your EC2 as self-hosted runner

## Setup github secrets
```bash
AWS_ACCESS_KEY_ID= {{ AWS_ACCESS_KEY_ID}}

AWS_SECRET_ACCESS_KEY= {{AWS_SECRET_ACCESS_KEY}}

AWS_REGION = {{AWS_REGION}}

AWS_ECR_LOGIN_URI = demo>> 566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = demo>> simple-app
```
