import os
import sys
import src
from src.exception import CustomException
from src.configured_logger import logging as logger

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    def run(self):
        try:
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion completed successfully.")
            train_arr,test_arr = self.data_transformation.initiate_data_transformation(train_path, test_path)
            logger.info("Data transformation completed successfully.")
            model_trainer = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error occurred during training pipeline: {e}")
            raise CustomException(e, sys)   
