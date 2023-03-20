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
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import LabelEncoder
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    LabelEncoder_obj_file_path=os.path.join('artifacts',"LabelEncoder.pkl")


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

            logging.info(
                f"Preprocessing Data......."
            )

            input_feature_train_df['Clean_Text'] = input_feature_train_df['Text'].apply(preprocessDataset)
            input_feature_test_df['Clean_Text'] = input_feature_test_df['Text'].apply(preprocessDataset)
            input_feature_train_df.to_csv("artifacts/clean_train.csv")
            input_feature_test_df.to_csv("artifacts/clean_test.csv")

            logging.info(
                f"All Data Preprocessed"
            )

            logging.info(
                f"Applying TF-IDF"
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df["Clean_Text"])
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df["Clean_Text"])

            logging.info(
                f"Feature Extraction Done"
            )
    
           
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            target_feature_train_df=label_encoder.transform(target_feature_train_df)
            target_feature_test_df=label_encoder.transform(target_feature_test_df)

            logging.info(
                f"Converted categorical column to numerical column"
            )


            # train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            save_object(

                file_path=self.data_transformation_config.LabelEncoder_obj_file_path,
                obj=label_encoder

            )

            return (
                input_feature_train_arr,target_feature_train_df,
                input_feature_test_arr,target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)