"""
TODO

Model evaluation report generating module.
"""

from typing import Any

import pandas as pd


class Report:
    def __init__(self, columns: list[str]) -> None:
        self._data = ...

    def add_row(self, row: list) -> Any:
        pass
