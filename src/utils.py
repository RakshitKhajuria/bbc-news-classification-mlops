import os
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def encode_categorical(column):
    try:
        le = LabelEncoder()
        le.fit(column)
        encoded = le.transform(column)
        return encoded

    except Exception as e:
        raise CustomException(e, sys)