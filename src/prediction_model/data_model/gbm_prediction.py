from datetime import datetime
from decimal import Decimal
from typing import Optional


class GBMPrediction:
    model_id: str
    val_criterion: str = None
    datadate: datetime
    gvkey: int
    predicted_rtn: Optional[Decimal] = None
    real_rtn: Optional[Decimal] = None
    dir_acc: Optional[int] = None
    chosen_bottom: bool = False
    chosen_weighted: bool = False

    @classmethod
    def build_record(cls, record) -> "GBMPrediction":
        res = cls()
        res.model_id = record[0]
        res.datadate = record[1]
        res.gvkey = record[2]
        res.predicted_rtn = Decimal(record[3])
        res.real_rtn = record[4]
        if (res.predicted_rtn > 0 and res.real_rtn > 0) or (res.predicted_rtn < 0 and res.real_rtn < 0):
            res.dir_acc = 1
        else:
            res.dir_acc = 0

        return res

    def set_val_criterion(self, val_criterion):
        # THE VALIDATION CRITERION IS ONLY SET IN THE TESTING STAGE
        self.val_criterion = val_criterion

    def as_tuple(self):
        return (
            self.model_id,
            self.val_criterion,
            self.datadate,
            self.gvkey,
            self.predicted_rtn,
            self.real_rtn,
            self.dir_acc,
            self.chosen_bottom,
            self.chosen_weighted,
        )

    def choose_bottom(self):
        self.chosen_bottom = True

    def choose_weighted(self):
        self.chosen_weighted = True
