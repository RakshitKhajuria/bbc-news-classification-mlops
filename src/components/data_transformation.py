import sys
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.TextProcessing import preprocessDataset
from src.utils import encode_categorical
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:

            TFIDF=TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english') 
            pipeline=Pipeline(steps=[("tf-idf",TFIDF)])
            return pipeline

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
  
            preprocessing_obj=self.get_data_transformer_object()

            target_feature_name="Category"
            input_feature_name = ["Text"]

            input_feature_train_df=train_df.drop(columns=[target_feature_name],axis=1)
            target_feature_train_df=train_df[target_feature_name]

            input_feature_test_df=test_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df=test_df[target_feature_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_df["clean_text"]=preprocessDataset(input_feature_train_df["Text"]) 
            input_feature_test_df["clean_text"]=preprocessDataset(input_feature_test_df["Text"])

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df["clean_text"])
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df["clean_text"])
            target_feature_train_df=encode_categorical(target_feature_train_df)
            target_feature_test_df=encode_categorical(target_feature_test_df)


            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                input_feature_train_arr,target_feature_train_df,
                input_feature_test_arr,target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)