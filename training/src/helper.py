import os

import hydra
import mlflow
from dagshub import DAGsHubLogger
from omegaconf import DictConfig
import pandas as pd

class BaseLogger:
    def __init__(self):
        self.logger = DAGsHubLogger()

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        self.logger.log_metrics(metrics)

    def log_params(self, params: dict):
        mlflow.log_params(params)
        self.logger.log_hyperparams(params)

    def log_model(self, model):
        mlflow.log_artifact(model)




