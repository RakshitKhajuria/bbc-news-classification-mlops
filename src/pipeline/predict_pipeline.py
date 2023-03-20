import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.TextProcessing import preprocessDataset

class PredictPipeline:
    def __init__(self,text: str):
        self.text = text
        

    def predict(self):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/proprocessor.pkl'
            LabelEncoder_path='artifacts/LabelEncoder.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            LabelEncoder=load_object(file_path=LabelEncoder_path)
            clean_text=preprocessDataset(self.text)
            vectorized_text=preprocessor.transform([clean_text])
            prob=model.predict_proba(vectorized_text) 

            result = LabelEncoder.inverse_transform(model.predict(vectorized_text))
            

            
            return result[0],prob
        
        except Exception as e:
            raise CustomException(e,sys)
        






 