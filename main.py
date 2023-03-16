from src.components import data_ingestion
import sys, os
from src.exception import CustomException
from src.logger import logging


DataIngestion=data_ingestion.DataIngestion()
train_path,test_path=DataIngestion.initiate_data_ingestion()