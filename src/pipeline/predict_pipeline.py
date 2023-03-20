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
            probability=model.predict_proba(vectorized_text) 

            result = LabelEncoder.inverse_transform(model.predict(vectorized_text))

            
            proba_df = pd.DataFrame(probability)
            proba_df_clean = proba_df.T.reset_index()  
            proba_df_clean.columns = ["Category", "Probability"]
            proba_df_clean["Category"]=LabelEncoder.inverse_transform(proba_df_clean["Category"])
            proba_df_clean.sort_values("Probability",ascending=False,inplace =True)


            
            return result[0],proba_df_clean
        
        except Exception as e:
            raise CustomException(e,sys)
        






 