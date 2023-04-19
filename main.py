from src.components import data_ingestion,data_transformation,model_trainer
import sys, os
from src.exception import CustomException
from src.logger import logging

DataIngestion=data_ingestion.DataIngestion()
train_path,test_path=DataIngestion.initiate_data_ingestion()
DataTransformation=data_transformation.DataTransformation()
X_train,y_train,X_test,y_test,_=DataTransformation.initiate_data_transformation(train_path,test_path)
ModelTrainer=model_trainer.ModelTrainer()
ModelTrainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
