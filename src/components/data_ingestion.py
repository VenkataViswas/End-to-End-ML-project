import os
import sys
from src.exception import CustomException
from src.configured_logger import logging as Logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        Logging.info("Data Ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            Logging.info("Raw data loaded successfully")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            df_train.to_csv(self.ingestion_config.train_data_path, index=False)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False)
            Logging.info("Train and Test data saved successfully")                                                          
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            Logging.info("Data Ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)