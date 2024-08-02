import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for name in list(models.keys()):
            model = models[name]
            param = params[name]

            gs = GridSearchCV(model, param)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
        
        # No hyperparameter tuning
        # for name, model in models.items():
        #     model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report
    
    except Exception as e:
        raise CustomException(e)

