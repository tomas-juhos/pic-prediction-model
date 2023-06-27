from datetime import datetime
from typing import List


class Sample:
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    testing_start: datetime
    testing_end: datetime
    training_data: List
    validation_data: List
    testing_data: List

    @classmethod
    def build_record(cls, record) -> "Sample":
        res = cls()
        res.training_start = record[0]
        res.training_end = record[1]
        res.validation_start = record[2]
        res.validation_end = record[3]
        res.testing_start = record[4]
        res.testing_end = record[5]
        res.training_data = record[6]
        res.validation_data = record[7]
        res.testing_data = record[8]

        return res
