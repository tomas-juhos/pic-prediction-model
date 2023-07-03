from datetime import datetime
from decimal import Decimal


class RegressionMetrics:
    testing_start: datetime
    testing_end: datetime
    model: str
    train_criterion: str
    val_criterion: str
    strategy: str

    rtn_bottom: Decimal
    rtn_weighted: Decimal
    rtn_random: Decimal
    rtn_benchmark: Decimal

    mse: Decimal
    rmse: Decimal
    mae: Decimal
    mape: Decimal
    dir_acc: Decimal
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
        res.train_criterion = record[3]
        res.val_criterion = record[4]

        res.rtn_bottom = record[5]
        res.rtn_weighted = record[6]
        res.rtn_random = record[7]
        res.rtn_benchmark = record[8]

        res.mse = record[9]
        res.rmse = record[10]
        res.mae = record[11]
        res.mape = record[12]
        res.dir_acc = record[13]
        res.f_pvalue = record[14]
        res.r_sqr = record[15]

        res.training_start = record[16]
        res.training_end = record[17]
        res.validation_start = record[18]
        res.validation_end = record[19]

        return res

    def as_tuple(self):
        return (
            self.testing_start,
            self.testing_end,
            self.model,
            self.train_criterion,
            self.val_criterion,
            self.rtn_bottom,
            self.rtn_weighted,
            self.rtn_random,
            self.rtn_benchmark,
            self.mse,
            self.rmse,
            self.mae,
            self.mape,
            self.dir_acc,
            self.f_pvalue,
            self.r_sqr,
            self.training_start,
            self.training_end,
            self.validation_start,
            self.validation_end,
        )
