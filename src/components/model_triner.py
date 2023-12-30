# Basic Import
import numpy as np
import pandas as pd
import sys
sys.path.append('D:\\ML_projects\\PhishingClassifier\\src')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from exception import CustomException
from logger import logging

from utils import save_object
from utils import evaluate_model_classification

from dataclasses import dataclass
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Loading preprocessed data...')
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            

            

            # Split the data into training and testing sets
            
            logging.info('preprocessed data is splitted in train and test data.')
            

            logging.info("models training is started")
            models = {
                      'logistic_regression': LogisticRegression(),
                      'decision_tree': DecisionTreeClassifier(),
                      'random_forest': RandomForestClassifier(),
                      'gradient boosting':GradientBoostingClassifier(),
                      'svm':SVC(),
                      'knn':KNeighborsClassifier()
                     }

            
            classification_report:dict=evaluate_model_classification(X_train,y_train,X_test,y_test,models)
            print(classification_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {classification_report}')

           # To get best model score from dictionary 
            best_model_score = max(sorted(classification_report.values()))

            best_model_name = list(classification_report.keys())[
                list(classification_report.values()).index(best_model_score)
            ]  # Assuming you have a 'model_name' key in your dictionary
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , accuracy : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , accuracy : {best_model_score}')
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          
            
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)