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

        res.mse = record[7]
        res.rmse = record[8]
        res.mae = record[9]
        res.mape = record[10]
        res.dir_acc = record[11]
        res.f_pvalue = record[12]
        res.r_sqr = record[13]

        res.training_start = record[14]
        res.training_end = record[15]
        res.validation_start = record[16]
        res.validation_end = record[17]

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
