import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
## creating a pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.configured_logger import logging as logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path  = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    This class is responsible for transforming the data by applying various preprocessing techniques.
    It includes handling numerical and categorical features, imputing missing values, scaling, and encoding.
    """
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            logger.info("Numerical features: %s", numerical_features)
            logger.info("Categorical features: %s", categorical_features)
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ])
            logger.info("Numerical pipeline created successfully for imputer and standard scaler.")
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
                ])
            logger.info("Categorical pipeline created successfully for imputer, onehotencoder and scaler.")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logger.info("Column transformer created successfully with numerical and categorical pipelines.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Train and test data loaded successfully.")
            
            logger.info("Obtaining preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()
            logger.info("Preprocessor object obtained successfully.")

            target_column_name = 'math_score'
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing dataframes.")
            input_features_train_transformed = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_transformed = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_transformed, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_transformed, np.array(target_feature_test_df)]
            logger.info("Data transformation completed successfully.")
            ## using save_object() in utils.py to save the preprocessor object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logger.info("Preprocessor object saved successfully.")
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)   