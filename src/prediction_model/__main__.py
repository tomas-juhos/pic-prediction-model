from datetime import datetime
import logging
from sys import stdout
from typing import Any, Dict, List
import os

from prediction_model.date_helpers import generate_intervals
from prediction_model.persistance import source, target
import prediction_model.data_model as data_model
import prediction_model.model as pred_model
import prediction_model.queries as queries

import numpy as np
from scipy import stats

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=stdout,
)

logger = logging.getLogger(__name__)


# UNIVERSE IS US STOCKS BETWEEN 100M AND 1000M MKT CAP
class PredictionModel:
    YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    OPTIMIZATION_CRITERIA = ["MSE", "DIR_ACC", "RTN_BOTTOM", "RTN_WEIGHTED"]

    def __init__(self):
        self.source = source.Source(os.environ.get("SOURCE"))
        self.target = target.Target(os.environ.get("TARGET"))

        self.universe_constr = os.environ.get("UNIVERSE_CONSTR")
        self.mode = os.environ.get("MODEL")
        self.normalized_features = os.environ.get("NORMALIZED_FEATURES")

    def run(self):
        """Runs process."""
        logger.info(
            f"Starting process for {self.mode.upper()}/{self.universe_constr.upper()}..."
        )
        date_ranges = generate_intervals(self.YEARS)
        j = 0
        for date_range in date_ranges:
            range_start = self.target.get_next_training_start(
                universe_constr=self.universe_constr.upper(), mode=self.mode.upper()
            )
            if range_start:
                if range_start > date_range[1].date():
                    logger.info(f"Data for {date_range[1].year} was already persisted.")
                    j += 1
                    continue
                else:
                    date_range = (range_start, date_range[1])

            logger.info("Building history...")
            history = self.build_history(date_range)
            history = self.clean_history(history)

            available_dates = list(history.keys())
            available_dates.sort()

            logger.info("Organizing data into samples...")
            samples = self.organize_data(available_dates=available_dates)

            logger.info("Running predictions...")
            for sample in samples:
                if self.mode == "regression":
                    self.run_regression(sample, history)
                elif self.mode == "gbm":
                    self.run_lightgbm(sample, history)
                else:
                    logger.info("No valid running mode was provided (regression/gbm).")
            j += 1
            logger.info(f"Persisted results for {j}/{len(date_ranges)} date ranges.")

    def run_regression(self, sample: data_model.Sample, history):
        logger.debug("Running regression models.")
        logger.debug(f"Training: {sample.training_start} <-> {sample.training_end}.")
        logger.debug(
            f"Validating: {sample.validation_start} <-> {sample.validation_end}."
        )
        logger.debug(f"Testing: {sample.testing_start} <-> {sample.testing_end}.")
        logger.debug("--//--")

        training_data, validation_data, testing_data = self.get_sample_data(
            sample, history
        )

        models = []
        if not self.normalized_features:
            models.append(
                pred_model.Regression(
                    model_name="ols",
                    training_data=training_data,
                    train_criterion="n/a",
                )
            )

        for optimization_criterion in self.OPTIMIZATION_CRITERIA:
            models.append(
                pred_model.Regression(
                    model_name="lasso",
                    training_data=training_data,
                    train_criterion=optimization_criterion,
                )
            )
            models.append(
                pred_model.Regression(
                    model_name="ridge",
                    training_data=training_data,
                    train_criterion=optimization_criterion,
                )
            )

        results = pred_model.RegressionResults(
            models=models,
            validation_data=validation_data,
            test_data=testing_data,
        )

        metrics_records = []
        prediction_records = []
        parameters_records = []
        for optimization_criterion in self.OPTIMIZATION_CRITERIA:
            selected_model = results.select_model(val_criterion=optimization_criterion)

            (
                model_metrics,
                model_predictions,
                model_parameters,
            ) = results.test_model(
                sample=sample,
                universe_constr=self.universe_constr,
                val_criterion=optimization_criterion,
                selected_model=selected_model,
            )

            metrics_records.append(model_metrics)
            prediction_records.extend(model_predictions)
            parameters_records.append(model_parameters)

        metrics_records = [r.as_tuple() for r in metrics_records]
        prediction_records = [r.as_tuple() for r in prediction_records]
        parameters_records = [r.as_tuple() for r in parameters_records]

        self.target.execute(queries.RegressionMetricsQueries.UPSERT, metrics_records)
        self.target.execute(
            queries.RegressionPredictionsQueries.UPSERT, prediction_records
        )
        self.target.execute(
            queries.RegressionParametersQueries.UPSERT, parameters_records
        )
        self.target.commit_transaction()

    def run_lightgbm(self, sample, history):
        logger.debug("Running regression models.")
        logger.debug(f"Training: {sample.training_start} <-> {sample.training_end}.")
        logger.debug(
            f"Validating: {sample.validation_start} <-> {sample.validation_end}."
        )
        logger.debug(f"Testing: {sample.testing_start} <-> {sample.testing_end}.")
        logger.debug("--//--")

        training_data, validation_data, testing_data = self.get_sample_data(
            sample, history
        )

        model_id = self.target.get_gbm_model_id()
        models = [
            pred_model.LightGBM(
                model_id=model_id, training_data=training_data, randomize_params=False
            )
        ]
        for i in range(9):
            model_id = self.target.get_gbm_model_id()
            models.append(
                pred_model.LightGBM(
                    model_id=model_id,
                    training_data=training_data,
                    randomize_params=True,
                )
            )

        results = pred_model.LightGBMResults(
            models=models, validation_data=validation_data, test_data=testing_data
        )

        metrics_records = []
        prediction_records = []
        parameters_records = []
        for optimization_criterion in self.OPTIMIZATION_CRITERIA:
            selected_model = results.select_model(val_criterion=optimization_criterion)

            (
                model_metrics,
                model_predictions,
                model_parameters,
            ) = results.test_model(
                sample=sample,
                universe_constr=self.universe_constr,
                val_criterion=optimization_criterion,
                selected_model=selected_model,
            )

            metrics_records.append(model_metrics)
            prediction_records.extend(model_predictions)
            parameters_records.append(model_parameters)

        metrics_records = [r.as_tuple() for r in metrics_records]
        prediction_records = [r.as_tuple() for r in prediction_records]
        parameters_records = [r.as_tuple() for r in parameters_records]

        self.target.execute(queries.GBMMetricsQueries.UPSERT, metrics_records)
        self.target.execute(queries.GBMPredictionsQueries.UPSERT, prediction_records)
        self.target.execute(queries.GBMParametersQueries.UPSERT, parameters_records)
        self.target.commit_transaction()

    def build_history(self, date_range) -> Dict[datetime, List[data_model.FactorsAll]]:
        """Loads date range into memory grouped by date, modeled as Factors."""
        # HISTORY FOR BASE DATA
        base_records = self.source.fetch_records(
            table="daily_base", date_range=date_range
        )
        base_records = [data_model.FactorsBase.build_record(r) for r in base_records]
        history: Dict[datetime, Any] = {}
        for record in base_records:
            if record.datadate in history.keys():
                history[record.datadate][record.gvkey] = [record]
            else:
                history[record.datadate] = {record.gvkey: [record]}

        # APPENDING METRICS DATA
        metrics_records = self.source.fetch_records(
            table="daily_metrics", date_range=date_range
        )
        metrics_records = [
            data_model.FactorsMetrics.build_record(r) for r in metrics_records
        ]
        for record in metrics_records:
            history[record.datadate][record.gvkey].append(record)

        # BUILDING BIIIG RECORD
        for d in history.keys():
            history[d] = [
                data_model.FactorsAll.build_record(v[0], v[1])
                for v in history[d].values()
                if len(v) == 2
            ]

        return history

    @staticmethod
    def from_history(
        dates: List[datetime], history: Dict[datetime, List[data_model.FactorsAll]]
    ):
        """Gets records from history for the provided dates."""
        records = []
        for d in dates:
            temp = history[d]
            records.extend(temp[:100])
        return records

    def clean_history(self, history: Dict[datetime, List[data_model.FactorsAll]]):
        res = {}
        for d in history.keys():
            records = [r for r in history[d] if r.is_complete]
            records.sort(
                key=lambda x: getattr(x, self.universe_constr.lower()), reverse=True
            )
            res[d] = records[:100]
        return res

    @staticmethod
    def organize_data(available_dates) -> List[data_model.Sample]:
        i = 0
        samples = []
        while 11 + i < len(available_dates):
            training_dates = available_dates[i : 5 + i]
            validation_dates = available_dates[5 + i : 10 + i]
            testing_dates = [available_dates[11 + i]]

            samples.append(
                data_model.Sample.build_record(
                    (training_dates, validation_dates, testing_dates)
                )
            )
            i += 1

        return samples

    def get_sample_data(self, sample: data_model.Sample, history):
        training_data = self.from_history(sample.training_dates, history)
        validation_data = self.from_history(sample.validation_dates, history)
        testing_data = self.from_history(sample.testing_dates, history)

        if self.normalized_features:
            training_data = self.normalize(training_data)
            validation_data = self.normalize(validation_data)
            testing_data = self.normalize(testing_data)

        return training_data, validation_data, testing_data

    @staticmethod
    def normalize(records: List[data_model.FactorsAll]) -> List[data_model.FactorsAll]:
        x = np.array([r.as_array() for r in records], dtype=float)
        x = stats.zscore(x, axis=1)

        res = []
        for i, record in enumerate(records):
            res.append(
                data_model.FactorsAll.from_array(
                    key=(record.datadate, record.gvkey), array=x[i], rtn=record.rtn
                )
            )
        return res


prediction_model = PredictionModel()
prediction_model.run()
