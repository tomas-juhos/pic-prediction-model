from decimal import Decimal
from typing import List

import numpy as np


class PerformanceMetrics:
    # LIST OF PREDICTION OBJECTS
    #
    def __init__(self, predictions: List):
        self.predictions: List = predictions
        self.mse = self.compute_mse()
        self.rtn_bottom = self.compute_rtn_bottom()
        self.rtn_weighted = self.compute_rtn_weighted()

    def compute_mse(self):
        se = []
        for p in self.predictions:
            se.append((Decimal(p.predicted_rtn) - p.real_rtn) ** Decimal(2))
        if len(se) == 0:
            return 0
        mse = sum(se) / len(se)
        return mse

    def compute_rtn_bottom(self):
        """Computes total return of bottom 20 rtns based on prediction."""
        rtns = np.array([], dtype=float)
        dates = sorted(set(p.datadate for p in self.predictions))
        for d in dates:
            preds = [p for p in self.predictions if p.datadate == d]
            # SORT BY PREDICTED RETURN AND GET 20 WORST RETURNS
            chosen_preds = sorted(preds, key=lambda x: getattr(x, "predicted_rtn"))[:20]

            for p in self.predictions:
                if p in chosen_preds:
                    p.choose_bottom()

            # GET THE REAL RETURNS FOR THOSE KEYS
            chosen_real_returns = [p.real_rtn for p in chosen_preds]

            # COMPUTE TOTAL RETURN (EQUALLY WEIGHTED)
            rtn = sum(chosen_real_returns) / len(chosen_real_returns)
            rtns = np.append(rtns, [float(rtn)])

        total_rtn = (rtns + 1).cumprod() - 1

        if len(total_rtn) == 0:
            return 0

        return Decimal(total_rtn[-1])

    def compute_rtn_weighted(self):
        rtns = np.array([], dtype=float)
        dates = sorted(set(p.datadate for p in self.predictions))
        for d in dates:
            preds = [
                p for p in self.predictions if p.datadate == d and p.predicted_rtn < 0
            ]
            if not preds:
                return 0
            chosen_preds = sorted(preds, key=lambda x: getattr(x, "predicted_rtn"))[:20]
            for p in self.predictions:
                if p in chosen_preds:
                    p.choose_weighted()

            total_weight = sum([abs(p.predicted_rtn) for p in chosen_preds])
            if total_weight == 0:
                return 0

            real_rtn = []
            for p in chosen_preds:
                weight = Decimal(abs(p.predicted_rtn) / total_weight)
                weighted_rtn = p.real_rtn * weight
                real_rtn.append(weighted_rtn)

            rtn = sum(real_rtn)
            rtns = np.append(rtns, [float(rtn)])

        total_rtn = (rtns + 1).cumprod() - 1

        if len(total_rtn) == 0:
            return 0
        return Decimal(total_rtn[-1])
