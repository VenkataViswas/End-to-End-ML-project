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

    def initiate_model_trainer(self, train_array, test_array):
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
            params = {
                        'KNeighbors Regressor': {
                            'n_neighbors': [3, 5, 7, 9, 11]
                        },
                        'Decision Tree Regressor': {
                            'max_depth': [5, 10, 15, 20],
                            'min_samples_split': [2, 5, 10]
                        },
                        'Random Forest Regressor': {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [5, 10, 20],
                            'min_samples_split': [2, 5]
                        },
                        'AdaBoost Regressor': {
                            'n_estimators': [50, 100, 150],
                            'learning_rate': [0.01, 0.1, 1.0]
                        },
                        'Gradient Boosting Regressor': {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'max_depth': [3, 5, 7]
                        },
                        'CatBoost Regressor': {
                            'iterations': [500, 1000],
                            'learning_rate': [0.03, 0.1],
                            'depth': [4, 6, 8]
                        },
                        'XGBoost Regressor': {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.01, 0.1],
                            'max_depth': [3, 5, 7]
                        }
                    }


            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logger.info(f"Best model found: {best_model_name} with R-squared score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            # return R2 score of the best model
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)