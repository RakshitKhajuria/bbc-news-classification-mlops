from src.components import data_ingestion,data_transformation
import sys, os
from src.exception import CustomException
from src.logger import logging


DataIngestion=data_ingestion.DataIngestion()
train_path,test_path=DataIngestion.initiate_data_ingestion()
DataTransformation=data_transformation.DataTransformation()
train_arr,test_arr,_=DataTransformation.initiate_data_transformation(train_path,test_path)