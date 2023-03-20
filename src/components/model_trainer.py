import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression



from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        logging.info("Initiating model trainer")
        try:
            # logging.info("Split training and test input data")
            # X_train,y_train,X_test,y_test=(
            #     train_array[:,:-1],
            #     train_array[:,-1],
            #     test_array[:,:-1],
            #     test_array[:,-1]
            # )
            model = LogisticRegression(C= 10, 
                max_iter= 100,
                multi_class= 'multinomial',
                penalty= 'l2',
                solver= 'newton-cg')
            
            logging.info("Training the model......")
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model=model)

            logging.info(f"Getting Model Report {model_report}")

            if model_report["accuracy_score"]<0.6:
                raise Exception("No best model found")
            logging.info(f" Model Accuracy is above the Threshold.... Procedding Further")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info(f"Saving Model to Directory")
            logging.info(f"Getting Model Accuracy")
            return model_report["accuracy_score"]
        
        except Exception as e:
            raise CustomException(e,sys)