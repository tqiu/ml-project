import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Data transformation pipelines are defined.")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading train and test data completed.")

            preprocessor = self.get_data_transformer_obj()
            target_column = "math_score"

            feature_train_df = train_df.drop(columns=target_column)
            feature_test_df = test_df.drop(columns=target_column)

            logging.info("Applying preprocessor to train and test data")
            X_train = preprocessor.fit_transform(feature_train_df)
            X_test = preprocessor.transform(feature_test_df)

            train_arr = np.c_[X_train, np.array(train_df[target_column])]
            test_arr = np.c_[X_test, np.array(test_df[target_column])]

            logging.info("Saved preprocessor object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )

            return train_arr, test_arr
        
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestionConfig
    train_data_path = DataIngestionConfig.train_data_path
    test_data_path = DataIngestionConfig.test_data_path

    obj = DataTransformation()
    obj.initiate_data_transformation(train_data_path, test_data_path)
