"""Helper functions to deal with timeframes."""
from datetime import datetime, timedelta
from typing import List
import logging

logger = logging.getLogger(__name__)


def is_month_end(d: datetime) -> bool:
    """Checks if date is the end of the month."""
    next_day = d + timedelta(days=1)
    if d.month != next_day.month:
        return True
    else:
        return False


def generate_months(years: List[int]):
    months = []
    for year in years:
        temp_d = datetime(year, 1, 1)
        month_start = temp_d
        while temp_d.year == year:
            if temp_d.day == 1:
                month_start = temp_d
            if is_month_end(temp_d):
                months.append((month_start, temp_d))
            temp_d = temp_d + timedelta(days=1)

    return months


def generate_intervals(years: List[int]):
    res = []
    for i in range(len(years)):
        res.append((datetime(years[i], 1, 1), datetime(years[i], 12, 31)))

    return res
