from datetime import datetime
from decimal import Decimal
from typing import Optional


class Prediction:
    model: str
    train_criterion: str
    val_criterion: str = None
    datadate: datetime
    gvkey: int
    predicted_rtn: Optional[Decimal] = None
    real_rtn: Optional[Decimal] = None
    chosen_bottom: bool = False
    chosen_weighted: bool = False

    @classmethod
    def build_record(cls, record) -> "Prediction":
        res = cls()
        res.model = record[0]
        res.train_criterion = record[1]
        res.datadate = record[2]
        res.gvkey = record[3]
        res.predicted_rtn = record[4]
        res.real_rtn = record[5]

        return res

    def set_val_criterion(self, val_criterion):
        # THE VALIDATION CRITERION IS ONLY SET IN THE TESTING STAGE
        self.val_criterion = val_criterion

    def as_tuple(self):
        return (
            self.model,
            self.train_criterion,
            self.val_criterion,
            self.datadate,
            self.gvkey,
            self.predicted_rtn,
            self.real_rtn,
            self.chosen_bottom,
            self.chosen_weighted,
        )

    def choose_bottom(self):
        self.chosen_bottom = True

    def choose_weighted(self):
        self.chosen_weighted = True
