from typing import List

from prediction_model.data_model import FactorsAll, GBMPrediction, GBMParameters, GBMMetrics
from prediction_model.performance_metrics import PerformanceMetrics

import lightgbm
import numpy as np
from scipy.stats import randint
import pandas as pd


class LightGBM:
    def __init__(self, model_id, training_data, randomize_params=False):
        self.x_train, self.y_train = self.as_dataframe(training_data)
        self.model_id = model_id
        default_params = {
            "max_depth": 5,
            "num_leaves": 10,
            "min_data_in_leaf": 100,
            "seed": 5,
            # "TIME_STEPS": 12,
            "verbose": -1,
        }
        self.useMinMax = 0
        if randomize_params:
            self.params = self.generate_random_params()
        else:
            self.params = default_params

        self.model = self.train()

    @staticmethod
    def generate_random_params():
        return {
            "max_depth": randint.rvs(1, 20),
            "num_leaves": randint.rvs(2, 30),
            "min_data_in_leaf": randint.rvs(1, 20),
            # "TIME_STEPS": randint.rvs(5, 50),
            "seed": 5,
            "verbose": -1,
        }

    def train(self):
        train_data = lightgbm.Dataset(
            self.x_train,
            label=self.y_train,
        )
        model = lightgbm.train(
            self.params, train_data, early_stopping_rounds=0
        )
        return model

    def evaluate(self, data: List[FactorsAll]) -> PerformanceMetrics:
        predictions = []
        for r in data:
            predictions.append(
                GBMPrediction.build_record(
                    (
                        self.model_id,
                        r.datadate,
                        r.gvkey,
                        self.model.predict(np.array([r.as_array()]))[0],  # ARRAY OF 1
                        r.rtn,
                    )
                )
            )

        metrics = PerformanceMetrics(predictions)

        return metrics

    def as_dataframe(self, records: List[FactorsAll]):
        x_array, y_array = self.as_equation(records)
        x = pd.DataFrame(x_array)
        y = pd.DataFrame(y_array)

        return x, y

    @staticmethod
    def as_equation(records: List[FactorsAll]):
        """Turns modeled records into arrays to feed model."""
        y = np.array([r.rtn for r in records], dtype=float)
        # TURN INTO COLUMN VECTOR
        y = np.transpose(y)
        x = np.array([r.as_array() for r in records], dtype=float)

        return x, y


class LightGBMResults:
    def __init__(self, models: List[LightGBM], validation_data, test_data):
        self.models = models
        self.validation_data = validation_data
        self.testing_data = test_data

    def select_model(self, val_criterion):
        res = []
        for model in self.models:
            model_eval = model.evaluate(self.validation_data)
            res.append((model, model_eval))

        if val_criterion == "dir_acc":
            chosen_model = max(res, key=lambda item: getattr(item[1], val_criterion))
        else:
            chosen_model = min(res, key=lambda item: getattr(item[1], val_criterion))

        # ACTUAL MODEL
        return chosen_model[0]

    def test_model(
            self,
            sample,
            val_criterion: str,
            selected_model: LightGBM,
    ):
        model_eval: PerformanceMetrics = selected_model.evaluate(self.testing_data)

        # MODEL PERFORMANCE METRICS
        metrics = GBMMetrics.build_record(
            (
                sample.testing_start,
                sample.testing_end,
                selected_model.model_id,
                val_criterion,
                model_eval.rtn_bottom,
                model_eval.rtn_weighted,
                model_eval.rtn_random,
                model_eval.rtn_benchmark,
                model_eval.mse,
                model_eval.rmse,
                model_eval.mae,
                model_eval.mape,
                model_eval.dir_acc,
                sample.training_start,
                sample.training_end,
                sample.validation_start,
                sample.validation_end,
            )
        )

        # MODEL PREDICTIONS
        for p in model_eval.predictions:
            p.set_val_criterion(val_criterion=val_criterion)
        predictions = model_eval.predictions

        # MODEL PARAMETERS
        key = (
            sample.testing_start,
            sample.testing_end,
            selected_model.model_id,
            val_criterion,
        )
        parameters = GBMParameters.build_record(
            key, selected_model.params
        )

        return metrics, predictions, parameters
