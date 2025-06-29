import os
import sys
from src.configured_logger import logging as logger
from src.exception import CustomException
from src.utils import save_object,evaluate_models
# Imorting the models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
# Loading evaluation Metrics
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logger.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'XGBoost Regressor': XGBRegressor(eval_metric='rmse')
            }        

            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logger.info(f"Best model found: {best_model_name} with R-squared score: {best_model_score}")

            ## If you want to do something with new data , you can load the preprocessor 
            # preprocessor = load_object(preprocessor_path) 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            # return R2 score of the best model
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)