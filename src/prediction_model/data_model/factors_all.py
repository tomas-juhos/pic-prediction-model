"""Factors data_model."""

from datetime import datetime
from decimal import Decimal
import logging
from typing import Optional

from .base import Modeling
from .factors_base import FactorsBase
from .factors_metrics import FactorsMetrics

import numpy as np

logger = logging.getLogger(__name__)


class FactorsAll(Modeling):
    """Aggregate base record object class."""

    datadate: datetime
    gvkey: int

    utilization_pct: Optional[Decimal] = None
    bar: Optional[Decimal] = None
    age: Optional[Decimal] = None
    tickets: Optional[Decimal] = None
    units: Optional[Decimal] = None
    market_value_usd: Optional[Decimal] = None
    loan_rate_avg: Optional[Decimal] = None
    loan_rate_max: Optional[Decimal] = None
    loan_rate_min: Optional[Decimal] = None
    loan_rate_range: Optional[Decimal] = None
    loan_rate_stdev: Optional[Decimal] = None

    utilization_pct_delta: Optional[Decimal] = None
    bar_delta: Optional[int] = None
    age_delta: Optional[Decimal] = None
    tickets_delta: Optional[int] = None
    units_delta: Optional[Decimal] = None
    market_value_usd_delta: Optional[Decimal] = None
    loan_rate_avg_delta: Optional[Decimal] = None
    loan_rate_max_delta: Optional[Decimal] = None
    loan_rate_min_delta: Optional[Decimal] = None
    loan_rate_range_delta: Optional[Decimal] = None
    loan_rate_stdev_delta: Optional[Decimal] = None
    short_interest: Optional[Decimal] = None
    short_ratio: Optional[Decimal] = None

    market_cap: Optional[Decimal] = None
    shares_out: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    rtn: Optional[Decimal] = None

    @classmethod
    def build_record(cls, base_record: FactorsBase, metrics_record: FactorsMetrics):
        res = cls()

        res.datadate = base_record.datadate
        res.gvkey = base_record.gvkey
        res.utilization_pct = base_record.utilization_pct
        res.bar = base_record.bar
        res.age = base_record.age
        res.tickets = base_record.tickets
        res.units = base_record.units
        res.market_value_usd = base_record.market_value_usd
        res.loan_rate_avg = base_record.loan_rate_avg
        res.loan_rate_max = base_record.loan_rate_max
        res.loan_rate_min = base_record.loan_rate_min
        res.loan_rate_range = base_record.loan_rate_range
        res.loan_rate_stdev = base_record.loan_rate_stdev

        res.utilization_pct_delta = metrics_record.utilization_pct_delta
        res.bar_delta = metrics_record.bar_delta
        res.age_delta = metrics_record.age_delta
        res.tickets_delta = metrics_record.tickets_delta
        res.units_delta = metrics_record.units_delta
        res.market_value_usd_delta = metrics_record.market_value_usd_delta
        res.loan_rate_avg_delta = metrics_record.loan_rate_avg_delta
        res.loan_rate_max_delta = metrics_record.loan_rate_max_delta
        res.loan_rate_min_delta = metrics_record.loan_rate_min_delta
        res.loan_rate_range_delta = metrics_record.loan_rate_range_delta
        res.loan_rate_stdev_delta = metrics_record.loan_rate_stdev_delta
        res.short_interest = metrics_record.short_interest
        res.short_ratio = metrics_record.short_ratio

        res.market_cap = base_record.market_cap
        res.shares_out = base_record.shares_out
        res.volume = base_record.volume
        # WINDSORIZED RETURNS BELOW
        res.rtn = base_record.rtn

        return res

    @classmethod
    def from_array(cls, key, array, rtn):
        res = cls()

        res.datadate = key[0]
        res.gvkey = key[1]
        res.utilization_pct = array[0]
        res.bar = array[1]
        res.age = array[2]
        res.tickets = array[3]
        res.units = array[4]
        res.market_value_usd = array[5]
        res.loan_rate_avg = array[6]
        res.loan_rate_max = array[7]
        res.loan_rate_min = array[8]
        res.loan_rate_range = array[9]
        res.loan_rate_stdev = None
        res.utilization_pct_delta = array[10]
        res.bar_delta = array[11]
        res.age_delta = array[12]
        res.tickets_delta = array[13]
        res.units_delta = array[14]
        res.market_value_usd_delta = array[15]
        res.loan_rate_avg_delta = array[16]
        res.loan_rate_max_delta = array[17]
        res.loan_rate_min_delta = array[18]
        res.loan_rate_range_delta = array[19]
        res.loan_rate_stdev_delta = None
        res.short_interest = array[20]
        res.short_ratio = array[21]
        res.market_cap = array[22]
        res.shares_out = array[23]
        res.volume = array[24]
        # WINDSORIZED RETURNS BELOW
        res.rtn = rtn

        return res

    def as_array(self) -> np.ndarray:
        """Builds array with all the factors in X."""
        return np.array(
            [
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
                self.volume,
            ],
            dtype=float,
        )

    @property
    def is_complete(self):
        if (
            self.utilization_pct is None
            or self.utilization_pct is None
            or self.bar is None
            or self.age is None
            or self.tickets is None
            or self.units is None
            or self.market_value_usd is None
            or self.loan_rate_avg is None
            or self.loan_rate_max is None
            or self.loan_rate_min is None
            or self.loan_rate_range is None
            # LOAN RATE STDEV IS SPARSE
            # or self.loan_rate_stdev is None
        ):
            return False
        else:
            return True

    def __repr__(self):
        return (
            f"{self.datadate}, "
            f"{self.gvkey}, "
            f"lr: {self.loan_rate_avg}, "
            f"si: {self.short_interest}, "
            f"rtn: {self.rtn}"
        )
