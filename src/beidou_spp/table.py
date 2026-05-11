"""Small table helper with optional pandas interoperability.

The course allows pandas, but some classroom machines may not have it installed
yet. Public APIs return a pandas DataFrame when pandas exists; otherwise they
return this lightweight CSV-capable object with a compatible subset used here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


class SimpleDataFrame:
    """Minimal DataFrame-like object used only as a no-pandas fallback."""

    def __init__(self, rows: Iterable[Dict], columns: Sequence[str] | None = None):
        self.rows = list(rows)
        if columns is None:
            ordered: List[str] = []
            for row in self.rows:
                for key in row:
                    if key not in ordered:
                        ordered.append(key)
            columns = ordered
        self.columns = list(columns)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.rows)

    def to_csv(self, path: str | Path, index: bool = False, encoding: str = "utf-8-sig") -> None:
        del index
        with Path(path).open("w", newline="", encoding=encoding) as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in self.columns})

    def to_dict(self, orient: str = "records") -> List[Dict]:
        if orient != "records":
            raise ValueError("SimpleDataFrame only supports orient='records'")
        return [dict(row) for row in self.rows]


def make_dataframe(rows: Iterable[Dict], columns: Sequence[str] | None = None):
    """Return pandas.DataFrame if available, otherwise SimpleDataFrame."""

    rows = list(rows)
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(rows, columns=columns)
    except ModuleNotFoundError:
        return SimpleDataFrame(rows, columns)


def records(table) -> List[Dict]:
    """Extract records from pandas or SimpleDataFrame."""

    if hasattr(table, "to_dict"):
        return list(table.to_dict("records"))
    return list(table)

