import numpy as np
import pandas as pd
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel
import hydra
import joblib
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import uvicorn
from fastapi import FastAPI


with initialize(version_base=None, config_path='../../config'):
    config = compose(config_name='main')
    FEATURES = config.process.features
    MODEL_NAME = config.model.name


class Employee(BaseModel):
    City: str = 'Pune'
    PaymentTier: int = 1
    Age: int = 25
    Gender: str = 'Female'
    EverBenched: str = 'No'
    ExperienceInCurrentDomain: int = 1


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        'City': ['Bangalore', 'New Delhi', 'Pune'],
        'Gender': ['Male', 'Female', 'Female'],
        'EverBenched': ['Yes', 'Yes', 'No'],
        'PaymentTier': [0, 0, 0],
        'Age': [0, 0, 0],
        'ExperienceInCurrentDomain': [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace('[', '_', regex=True).str.replace(
        ']', '', regex=True
    )
    return X
# test

def transform_data(df: pd.DataFrame):
    """Transform the data"""
    dummy_df = add_dummy_data(df)
    feature_str = ' + '.join(FEATURES)
    dummy_X = dmatrix(f'{feature_str} - 1', dummy_df, return_type='dataframe')
    dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)

def load_model(model_path: str):
    return joblib.load(model_path)

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Employee satisfaction ML API'}


@app.post('/employee/predict')
def predict(employee: Employee) -> dict:
    df = pd.DataFrame(employee.dict(), index=[0])
    df = transform_data(df)
    model = load_model(abspath(config.model.path))
    results = model.predict_proba(df)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    print(predictions)
    return {
        'prediction': int(predictions[0])
    }


if __name__ == '__main__':
    employ = Employee()
    uvicorn.run(app, host='0.0.0.0', port=5001)
    

    







