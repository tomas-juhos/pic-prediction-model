from datetime import datetime
from typing import List


class Sample:
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    testing_start: datetime
    testing_end: datetime
    training_dates: List
    validation_dates: List
    testing_dates: List

    @classmethod
    def build_record(cls, record) -> "Sample":
        res = cls()
        res.training_dates = record[0]
        res.validation_dates = record[1]
        res.testing_dates = record[2]
        res.training_start = res.training_dates[0].date()
        res.training_end = res.training_dates[-1].date()
        res.validation_start = res.validation_dates[0].date()
        res.validation_end = res.validation_dates[-1].date()
        res.testing_start = res.testing_dates[0].date()
        res.testing_end = res.testing_dates[-1].date()

        return res
