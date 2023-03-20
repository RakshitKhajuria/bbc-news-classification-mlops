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
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            clean_text=preprocessDataset(self.text)
            vectorized_text=preprocessor.transform(clean_text)
            preds=model.predict(vectorized_text)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        






 