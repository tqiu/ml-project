import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K Nearest Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(), 
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [200, 300, 500],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.005, 0.001]
                },
                "Linear Regression": {},
                "K Nearest Neighbors": {
                    "n_neighbors": [5, 7, 9]
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.01, 0.005, 0.001]
                },
                "CatBoost": {
                    "depth": [6, 8]
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.005, 0.001]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                models=models, params=params)
            best_model_name = max(model_report, key=lambda x: model_report[x])
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            
            logging.info(model_report)
            logging.info("Best model is found on test dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return best_model_score
        
        except Exception as e:
            raise CustomException(e)
        
