"""Abstract data_model."""

from abc import ABC, abstractmethod
from typing import Tuple

from .factors_base import FactorsBase
from .factors_metrics import FactorsMetrics

import numpy as np


class Modeling(ABC):
    """Modeling abstract class."""

    @classmethod
    @abstractmethod
    def build_record(
        cls, base_record: FactorsBase, metrics_record: FactorsMetrics
    ) -> "Modeling":
        """Transforms record into record object.

        Args:
            base_record: Base data record object.
            metrics_record: Metrics record object.
        Returns:
            Record object for the given entity.
        """

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Returns object values as a tuple.

        Returns:
            Record object attributes as a tuple.
        """
