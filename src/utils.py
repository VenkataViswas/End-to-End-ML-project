import sys
import os
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.configured_logger import logging as logger

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