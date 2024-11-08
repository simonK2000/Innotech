import joblib
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ConfusionMatrixReport
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
import pickle
from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
from deepchecks.tabular.suites import model_evaluation
from deepchecks.tabular.feature_importance import calculate_feature_importance

import sys
sys.path.append('./')

from training.src.train_model import load_data

def load_model(model_path: str):
    return joblib.load(model_path)

def test_xgboost():

    with initialize(version_base=None, config_path='../../config'):
        config = compose(config_name='main')

    model_path = abspath(config.model.path)
    print(model_path)
    model = load_model(abspath(config.model.path))
    print(model)

    X_train, X_test, y_train, y_test = load_data(config.processed)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_ds = Dataset(train_df,cat_features=["City_Bangalore",
                                              "City_New Delhi", 
                                              "City_Pune",
                                              "EverBenched_T.Yes"], label='LeaveOrNot')
    validation_ds = Dataset(test_df,cat_features=["City_Bangalore",
                                              "City_New Delhi", 
                                              "City_Pune",
                                              "EverBenched_T.Yes"],label='LeaveOrNot')
    
    fi = calculate_feature_importance(model, train_ds)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    
 
    print(train_df)
    print(test_df)
    
    result = model_evaluation().run(train_dataset=train_ds, test_dataset=validation_ds,
            feature_importance=fi, y_proba_train=train_proba, y_proba_test=test_proba)
    print(result)

if __name__ == "__main__":
    test_xgboost()