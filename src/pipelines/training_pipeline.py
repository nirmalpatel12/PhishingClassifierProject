import os
import sys
sys.path.append('D:\\ML_projects\\PhishingClassifier\\src')
from logger import logging
from exception import CustomException
import pandas as pd
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig
from components.model_triner import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()
    config = DataTransformationConfig()

    # Update the paths for train and test data files
    train_path = train_data_path
    test_path = test_data_path

    # Initialize data transformation
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()

    # Run model training
    model_trainer.initate_model_training(train_arr, test_arr)
