"""Target."""

from datetime import timedelta
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

    def get_next_training_start(self, universe_constr, mode):
        cursor = self.cursor
        query = (
            "SELECT MAX(training_start) "
            "FROM {model}_metrics "
            "WHERE universe_constr = %s;"
        ).format(model=mode)

        cursor.execute(query=query, vars=(universe_constr,))
        d = cursor.fetchone()

        return d[0] + timedelta(days=1) if d[0] is not None else None

    def get_gbm_model_id(self):
        query = "SELECT NEXTVAL('gbm_model_id_seq');"
        cursor = self.cursor
        cursor.execute(query)
        model_id = cursor.fetchone()

        return model_id[0]
