"""Source."""
from datetime import datetime
from typing import List, Tuple

import psycopg2
import psycopg2.extensions
from psycopg2.extras import execute_values


class Source:
    """Source class."""

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._connection = psycopg2.connect(connection_string)
        self._connection.autocommit = False
        self._tx_cursor = None

    @property
    def cursor(self) -> psycopg2.extensions.cursor:
        """Generate cursor.

        Returns:
            Cursor.
        """
        if self._tx_cursor is not None:
            cursor = self._tx_cursor
        else:
            cursor = self._connection.cursor()

        return cursor

    def disconnect(self) -> None:
        """Disconnect from database."""
        self._connection.close()

    def fetch_records(
        self, table: str, date_range: Tuple[datetime, datetime]
    ) -> List[Tuple]:
        cursor = self.cursor
        query_base = (
            "SELECT * "
            "FROM {table} "
            "WHERE datadate BETWEEN %s AND %s "
            "AND market_cap BETWEEN 100 AND 1000;"
        ).format(table=table)
        cursor.execute(query=query_base, vars=date_range)
        records = cursor.fetchall()

        return records

    def keys_fetch(self, table: str, keys: List[Tuple]) -> List[Tuple]:
        cursor = self.cursor
        query = (
            "SELECT * " "FROM {table} " "WHERE (datadate, gvkey) in (VALUES %s);"
        ).format(table=table)
        records = execute_values(cur=cursor, sql=query, argslist=keys, fetch=True)

        return records
