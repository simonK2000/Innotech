from dagshub.streaming import install_hooks
import joblib
from hydra.utils import to_absolute_path as abspath
import joblib
from hydra import compose, initialize
import pandas as pd
import dagshub

with initialize(version_base=None, config_path='config'):
    config = compose(config_name='main')
    FEATURES = config.process.features
    MODEL_NAME = config.model.name

dagshub.auth.add_app_token(config.mlflow_api)
    
from dagshub.streaming import install_hooks
install_hooks(project_root='.', repo_url=config.repo_url, branch='main')


m = joblib.load('models/xgboost')
joblib.dump(m, 'models/xgboost')

X_test = pd.read_csv('data/processed/X_test.csv')
X_test.to_csv('data/X_test.csv')

X_train = pd.read_csv('data/processed/X_train.csv')
X_train.to_csv('data/X_train.csv')

y_test = pd.read_csv('data/processed/y_test.csv')
y_test.to_csv('data/y_test.csv')

y_train = pd.read_csv('data/processed/y_train.csv')
y_train.to_csv('data/y_train.csv')