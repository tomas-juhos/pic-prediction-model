from decimal import Decimal
import logging
from typing import List

from prediction_model.data_model import (
    FactorsAll,
    RegressionPrediction,
    RegressionParameters,
    RegressionMetrics,
)
from prediction_model.performance_metrics import PerformanceMetrics

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.tools import pinv_extended

logger = logging.getLogger(__name__)


class Regression:
    alpha = None
    beta = None
    f_pvalue = None
    r_sqr = None

    def __init__(self, model_name, training_data, train_criterion: str):
        self.training_data = training_data
        self.name = model_name
        self.train_criterion = train_criterion
        self.model = self.build_models(model_name)

    def build_models(self, model_name):
        if model_name == "ols":
            return self.build_ols()
        else:
            return self.build_regularized_models()

    def build_ols(self):
        x, y = self.as_equation(self.training_data)
        model = sm.OLS(y, x)
        res = model.fit()

        self.alpha = 0
        self.beta = [
            Decimal((float(p))).quantize(Decimal("1.000000")) for p in res.params
        ]
        self.f_pvalue = res.f_pvalue

        return res

    def build_regularized_models(self):
        if self.name == "lasso":
            l1_wt = 1
        elif self.name == "ridge":
            l1_wt = 0
        else:
            logger.info("No valid regularization selected.")
            return

        x, y = self.as_equation(self.training_data)

        ols = sm.OLS(y, x)
        alphas = np.linspace(0.001, 10, num=50)
        models = []
        for a in alphas:
            self.model = ols.fit_regularized(alpha=a, L1_wt=l1_wt)
            model_eval: PerformanceMetrics = self.evaluate(self.training_data)

            pinv_wexog, _ = pinv_extended(ols.wexog)
            normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
            model_stats = sm.regression.linear_model.OLSResults(
                ols, self.model.params, normalized_cov_params
            )
            models.append((self.model, a, model_eval, model_stats))

        chosen_model = min(
            models, key=lambda item: getattr(item[2], self.train_criterion)
        )
        self.alpha = chosen_model[1]
        self.beta = [
            Decimal((float(p))).quantize(Decimal("1.000000"))
            for p in chosen_model[0].params
        ]
        self.f_pvalue = Decimal(float(chosen_model[3].f_pvalue)).quantize(
            Decimal("1.000000")
        )
        self.r_sqr = Decimal(float(chosen_model[3].rsquared)).quantize(
            Decimal("1.000000")
        )

        # ACTUAL MODEL
        return chosen_model[0]

    @staticmethod
    def as_equation(records: List[FactorsAll]):
        """Turns modeled records into arrays to feed model."""
        y = np.array([r.rtn for r in records], dtype=float)
        # TURN INTO COLUMN VECTOR
        y = np.transpose(y)
        x = np.array([r.as_array() for r in records], dtype=float)

        return x, y

    def evaluate(self, data: List[FactorsAll]) -> PerformanceMetrics:
        predictions = []
        for r in data:
            predictions.append(
                RegressionPrediction.build_record(
                    (
                        self.name,
                        self.train_criterion,
                        r.datadate,
                        r.gvkey,
                        self.model.predict(r.as_array())[0],  # ARRAY OF 1
                        r.rtn,
                    )
                )
            )

        metrics = PerformanceMetrics(predictions)

        return metrics


class RegressionResults:
    def __init__(self, models: List[Regression], validation_data, test_data):
        self.models = models
        self.validation_data = validation_data
        self.testing_data = test_data

    def select_model(self, val_criterion):
        res = []
        for model in self.models:
            model_eval = model.evaluate(self.validation_data)
            res.append((model, model_eval))

        chosen_model = min(res, key=lambda item: getattr(item[1], val_criterion))

        # ACTUAL MODEL
        return chosen_model[0]

    def test_model(
        self,
        sample,
        val_criterion: str,
        selected_model: Regression,
    ):
        model_eval: PerformanceMetrics = selected_model.evaluate(self.testing_data)

        # MODEL PERFORMANCE METRICS
        metrics = RegressionMetrics.build_record(
            (
                sample.testing_start,
                sample.testing_end,
                selected_model.name,
                selected_model.train_criterion,
                val_criterion,
                model_eval.rtn_bottom,
                model_eval.rtn_weighted,
                model_eval.mse,
                model_eval.rmse,
                model_eval.mae,
                model_eval.mape,
                model_eval.dir_acc,
                selected_model.f_pvalue,
                selected_model.r_sqr,
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
            selected_model.name,
            selected_model.train_criterion,
            val_criterion,
        )
        parameters = RegressionParameters.build_record(
            key, selected_model.alpha, selected_model.beta
        )

        return metrics, predictions, parameters
