"""Target."""

from typing import List, Tuple

import psycopg2
import psycopg2.extensions
from psycopg2.extras import execute_values


class Target:
    """Target class."""

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

    def commit_transaction(self) -> None:
        """Commits a transaction."""
        self._connection.commit()

    def disconnect(self) -> None:
        """Disconnect from database."""
        self._connection.close()

    def execute(self, query: str, records: List[Tuple]) -> None:
        """Execute batch of records into database.

        Args:
            query: query to execute.
            records: records to persist.
        """
        cursor = self.cursor
        execute_values(cur=cursor, sql=query, argslist=records)
