import sys
import os
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.configured_logger import logging as logger
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Saves an object to a file using dill.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    
    Returns:
    None
    """
    try:
        logger.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logger.info("Object saved successfully.")
    except Exception as e:
        raise CustomException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple regression models and returns their performance metrics.
    
    Parameters:
    - X_train (np.ndarray): Training feature set.
    - y_train (np.ndarray): Training target variable.
    - X_test (np.ndarray): Testing feature set.
    - y_test (np.ndarray): Testing target variable.
    - models (dict): Dictionary of model names and their instances.
    
    Returns:
    dict: A dictionary containing model names and their R-squared scores.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square
            logger.info(f"{model_name} R-squared score: {r2_square}")
        return model_report
    except Exception as e:
        raise CustomException(e, sys) from e