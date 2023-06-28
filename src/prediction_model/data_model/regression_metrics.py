from datetime import datetime
from decimal import Decimal


class RegressionMetrics:
    testing_start: datetime
    testing_end: datetime
    model: str
    test_criterion: str
    val_criterion: str
    strategy: str

    rtn_bottom: Decimal
    rtn_weighted: Decimal

    mse: Decimal
    f_pvalue: Decimal
    r_sqr: Decimal

    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime

    @classmethod
    def build_record(cls, record) -> "RegressionMetrics":
        res = cls()

        res.testing_start = record[0]
        res.testing_end = record[1]
        res.model = record[2]
        res.test_criterion = record[3]
        res.val_criterion = record[4]

        res.rtn_bottom = record[5]
        res.rtn_weighted = record[6]

        res.mse = record[7]
        res.f_pvalue = record[8]
        res.r_sqr = record[9]

        res.training_start = record[10]
        res.training_end = record[11]
        res.validation_start = record[12]
        res.validation_end = record[13]

        return res

    def as_tuple(self):
        return (
            self.testing_start,
            self.testing_end,
            self.model,
            self.test_criterion,
            self.val_criterion,
            self.rtn_bottom,
            self.rtn_weighted,
            self.mse,
            self.f_pvalue,
            self.r_sqr,
            self.training_start,
            self.training_end,
            self.validation_start,
            self.validation_end,
        )
