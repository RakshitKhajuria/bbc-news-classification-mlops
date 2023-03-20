import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    






def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        accuracy_score_=accuracy_score(y_pred,y_test)

        precision_score_=precision_score(y_pred,y_test,average='macro')

        recall_score_=recall_score(y_pred,y_test,average='macro')

        f1_score_=f1_score(y_pred,y_test,average='macro')
        
        report={"accuracy_score":accuracy_score_,
                "precision_score":precision_score_,
                "recall_score":recall_score_,
                "f1_score":f1_score_}
        return report
    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    