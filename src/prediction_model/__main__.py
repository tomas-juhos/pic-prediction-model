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
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=stdout,
)

logger = logging.getLogger(__name__)


# UNIVERSE IS US STOCKS UNDER 1000M MKT CAP
class PredictionModel:
    YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    optimization_criteria = ["mse", "rtn_bottom", "rtn_weighted"]

    def __init__(self, normalized_features):
        self.source = source.Source(os.environ.get("SOURCE"))
        self.target = target.Target(os.environ.get("TARGET"))

        self.normalized_features = normalized_features

    def run(self, normalized_features):
        """Runs process."""
        logger.info("Starting process...")
        date_ranges = generate_intervals(self.YEARS)
        j = 0
        for date_range in date_ranges:
            # TODO REMOVE AFTER TESTING
            # date_range = (datetime(2010, 1, 1), datetime(2010, 1, 22))

            logger.info("Building history...")
            history = self.build_history(date_range)
            history = self.clean_history(history)

            available_dates = list(history.keys())
            available_dates.sort()

            # WILL TURN INTO FOR LOOP
            i = 0
            metrics_records = []
            prediction_records = []
            parameters_records = []
            while 11 + i < len(available_dates):
                training_dates = available_dates[i: 5 + i]
                validation_dates = available_dates[5 + i: 10 + i]
                testing_dates = [available_dates[11 + i]]

                logger.info(f"Testing models: {testing_dates[0]}")

                training_data = self.from_history(training_dates, history)
                validation_data = self.from_history(validation_dates, history)
                testing_data = self.from_history(testing_dates, history)

                if normalized_features:
                    training_data = self.normalize(training_data)
                    validation_data = self.normalize(validation_data)
                    testing_data = self.normalize(testing_data)

                # MAYBE TURN CODE BELOW INTO run_regression FUNCTION
                models = []
                if not normalized_features:
                    models.append(
                        pred_model.Regression(
                            model_name="ols",
                            training_data=training_data,
                            train_criterion="n/a",
                        )
                    )

                for optimization_criterion in self.optimization_criteria:
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

                for optimization_criterion in self.optimization_criteria:
                    selected_model = results.select_model(
                        val_criterion=optimization_criterion
                    )

                    (
                        model_metrics,
                        model_predictions,
                        model_parameters,
                    ) = results.test_model(
                        training_range=(training_dates[0], training_dates[-1]),
                        validation_range=(validation_dates[0], validation_dates[-1]),
                        testing_range=(testing_dates[0], testing_dates[-1]),
                        val_criterion=optimization_criterion,
                        selected_model=selected_model,
                    )

                    metrics_records.append(model_metrics)
                    prediction_records.extend(model_predictions)
                    parameters_records.append(model_parameters)

                i += 1

            metrics_records = [r.as_tuple() for r in metrics_records]
            prediction_records = [r.as_tuple() for r in prediction_records]
            parameters_records = [r.as_tuple() for r in parameters_records]

            self.target.execute(
                queries.RegressionMetricsQueries.UPSERT, metrics_records
            )
            self.target.execute(
                queries.RegressionPredictionsQueries.UPSERT, prediction_records
            )
            self.target.execute(
                queries.RegressionParametersQueries.UPSERT, parameters_records
            )
            self.target.commit_transaction()
            j += 1
            logger.info(f"Persisted results for {j}/{len(date_ranges)} date ranges.")

    def run_regression(self, sample: data_model.Sample):
        metrics_records = []
        prediction_records = []
        parameters_records = []

        models = []
        if not self.normalized_features:
            models.append(
                pred_model.Regression(
                    model_name="ols",
                    training_data=sample.training_data,
                    train_criterion="n/a",
                )
            )

        for optimization_criterion in self.optimization_criteria:
            models.append(
                pred_model.Regression(
                    model_name="lasso",
                    training_data=sample.training_data,
                    train_criterion=optimization_criterion,
                )
            )
            models.append(
                pred_model.Regression(
                    model_name="ridge",
                    training_data=sample.training_data,
                    train_criterion=optimization_criterion,
                )
            )

        results = pred_model.RegressionResults(
            models=models,
            validation_data=sample.validation_data,
            test_data=sample.testing_data,
        )

        for optimization_criterion in self.optimization_criteria:
            selected_model = results.select_model(
                val_criterion=optimization_criterion
            )

            (
                model_metrics,
                model_predictions,
                model_parameters,
            ) = results.test_model(
                training_range=(sample.training_start, sample.training_end),
                validation_range=(sample.validation_start, sample.validation_end),
                testing_range=(sample.testing_start, sample.testing_end),
                val_criterion=optimization_criterion,
                selected_model=selected_model,
            )

            metrics_records.append(model_metrics)
            prediction_records.extend(model_predictions)
            parameters_records.append(model_parameters)

        metrics_records = [r.as_tuple() for r in metrics_records]
        prediction_records = [r.as_tuple() for r in prediction_records]
        parameters_records = [r.as_tuple() for r in parameters_records]

        self.target.execute(
            queries.RegressionMetricsQueries.UPSERT, metrics_records
        )
        self.target.execute(
            queries.RegressionPredictionsQueries.UPSERT, prediction_records
        )
        self.target.execute(
            queries.RegressionParametersQueries.UPSERT, parameters_records
        )
        self.target.commit_transaction()

    def run_lightgbm(self):
        # todo implement
        pass

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

    @staticmethod
    def clean_history(history: Dict[datetime, List[data_model.FactorsAll]]):
        res = {}
        for d in history.keys():
            records = [r for r in history[d] if r.is_complete]
            records.sort(key=lambda x: getattr(x, "loan_rate_avg"), reverse=True)
            res[d] = records[:100]
        return res

    def organize_data(self, history, available_dates, normalized_features: bool = True) -> List[data_model.Sample]:
        i = 0
        samples = []
        while 11 + i < len(available_dates):
            training_dates = available_dates[i: 5 + i]
            validation_dates = available_dates[5 + i: 10 + i]
            testing_dates = [available_dates[11 + i]]

            logger.info(f"Testing models: {testing_dates[0]}")

            training_data = self.from_history(training_dates, history)
            validation_data = self.from_history(validation_dates, history)
            testing_data = self.from_history(testing_dates, history)

            if normalized_features:
                training_data = self.normalize(training_data)
                validation_data = self.normalize(validation_data)
                testing_data = self.normalize(testing_data)

            samples.append(data_model.Sample.build_record((
                training_dates[0],
                training_dates[-1],
                validation_dates[0],
                validation_dates[-1],
                testing_dates[0],
                testing_dates[-1],
                training_data,
                validation_data,
                testing_data
            )))

        return samples

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
prediction_model.run(normalized_features=False)
