from datetime import datetime
from decimal import Decimal
from typing import Optional


class RegressionParameters:
    testing_start: datetime
    testing_end: datetime
    model: str
    train_criterion: str
    val_criterion: str

    alpha: Optional[Decimal]
    utilization_pct: Optional[Decimal]
    bar: Optional[Decimal]
    age: Optional[Decimal]
    tickets: Optional[Decimal]
    units: Optional[Decimal]
    market_value_usd: Optional[Decimal]
    loan_rate_avg: Optional[Decimal]
    loan_rate_max: Optional[Decimal]
    loan_rate_min: Optional[Decimal]
    loan_rate_range: Optional[Decimal]
    utilization_pct_delta: Optional[Decimal]
    bar_delta: Optional[Decimal]
    age_delta: Optional[Decimal]
    tickets_delta: Optional[Decimal]
    units_delta: Optional[Decimal]
    market_value_usd_delta: Optional[Decimal]
    loan_rate_avg_delta: Optional[Decimal]
    loan_rate_max_delta: Optional[Decimal]
    loan_rate_min_delta: Optional[Decimal]
    loan_rate_range_delta: Optional[Decimal]
    short_interest: Optional[Decimal]
    short_ratio: Optional[Decimal]
    market_cap: Optional[Decimal]
    shares_out: Optional[Decimal]

    @classmethod
    def build_record(cls, key, alpha, beta):
        res = cls()
        res.testing_start = key[0]
        res.testing_end = key[1]
        res.model = key[2]
        res.train_criterion = key[3]
        res.val_criterion = key[4]

        res.alpha = alpha
        res.utilization_pct = beta[0]
        res.bar = beta[1]
        res.age = beta[2]
        res.tickets = beta[3]
        res.units = beta[4]
        res.market_value_usd = beta[5]
        res.loan_rate_avg = beta[6]
        res.loan_rate_max = beta[7]
        res.loan_rate_min = beta[8]
        res.loan_rate_range = beta[9]
        res.utilization_pct_delta = beta[10]
        res.bar_delta = beta[11]
        res.age_delta = beta[12]
        res.tickets_delta = beta[13]
        res.units_delta = beta[14]
        res.market_value_usd_delta = beta[15]
        res.loan_rate_avg_delta = beta[16]
        res.loan_rate_max_delta = beta[17]
        res.loan_rate_min_delta = beta[18]
        res.loan_rate_range_delta = beta[19]
        res.short_interest = beta[20]
        res.short_ratio = beta[21]
        res.market_cap = beta[22]
        res.shares_out = beta[23]

        return res

    def as_tuple(self):
        return (
            self.testing_start,
            self.testing_end,
            self.model,
            self.train_criterion,
            self.val_criterion,
            self.alpha,
            self.utilization_pct,
            self.bar,
            self.age,
            self.tickets,
            self.units,
            self.market_value_usd,
            self.loan_rate_avg,
            self.loan_rate_max,
            self.loan_rate_min,
            self.loan_rate_range,
            self.utilization_pct_delta,
            self.bar_delta,
            self.age_delta,
            self.tickets_delta,
            self.units_delta,
            self.market_value_usd_delta,
            self.loan_rate_avg_delta,
            self.loan_rate_max_delta,
            self.loan_rate_min_delta,
            self.loan_rate_range_delta,
            self.short_interest,
            self.short_ratio,
            self.market_cap,
            self.shares_out,
        )
