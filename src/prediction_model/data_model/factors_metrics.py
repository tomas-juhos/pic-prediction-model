"""Metrics data_model."""

from datetime import datetime
from decimal import Decimal
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FactorsMetrics:
    """Metrics record object class."""

    datadate: datetime
    gvkey: int

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
    winsorized_5_rtn: Optional[Decimal] = None

    @classmethod
    def build_record(cls, record) -> "FactorsMetrics":
        res = cls()

        res.datadate = record[0]
        res.gvkey = record[1]
        res.utilization_pct_delta = record[2] if record[2] else 0
        res.bar_delta = record[3] if record[3] else 0
        res.age_delta = record[4] if record[4] else 0
        res.tickets_delta = record[5] if record[5] else 0
        res.units_delta = record[6] if record[6] else 0
        res.market_value_usd_delta = record[7] if record[7] else 0
        res.loan_rate_avg_delta = record[8] if record[8] else 0
        res.loan_rate_max_delta = record[9] if record[9] else 0
        res.loan_rate_min_delta = record[10] if record[10] else 0
        res.loan_rate_range_delta = record[11] if record[11] else 0
        res.loan_rate_stdev_delta = record[12] if record[12] else 0
        res.short_interest = record[13] if record[13] else 0
        res.short_ratio = record[14] if record[14] else 0
        res.market_cap = record[15]
        res.shares_out = record[16]
        res.volume = record[17]
        # WINDSORIZED RETURNS BELOW
        res.rtn = record[19]

        return res
