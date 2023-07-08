"""Factor data_model."""

from datetime import datetime
from decimal import Decimal
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FactorsBase:
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

    market_cap: Optional[Decimal] = None
    shares_out: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    rtn: Optional[Decimal] = None

    @classmethod
    def build_record(cls, record):
        res = cls()

        res.datadate = record[0]
        res.gvkey = record[1]
        res.utilization_pct = record[2] if record[2] else 0
        res.bar = record[3] if record[3] else 0
        res.age = record[4] if record[4] else 0
        res.tickets = record[5] if record[5] else 0
        res.units = record[6] if record[6] else 0
        res.market_value_usd = record[7] if record[7] else 0
        res.loan_rate_avg = record[8] if record[8] else 0
        res.loan_rate_max = record[9] if record[9] else 0
        res.loan_rate_min = record[10] if record[10] else 0
        res.loan_rate_range = record[11] if record[11] else 0
        res.loan_rate_stdev = record[12] if record[12] else 0
        res.market_cap = record[13]
        res.shares_out = record[14]
        res.volume = record[15]
        # WINDSORIZED RETURNS BELOW
        res.rtn = record[17]

        return res
