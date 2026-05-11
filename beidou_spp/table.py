"""轻量表格工具，并兼容可选的 pandas。

课程允许使用 pandas，但部分机器可能尚未安装。若存在 pandas，公共接口返回
pandas DataFrame；否则返回本文件中的轻量 CSV 表格对象。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


class SimpleDataFrame:
    """仅在未安装 pandas 时使用的极简 DataFrame 兼容对象。"""

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
            raise ValueError("SimpleDataFrame 只支持 orient='records'")
        return [dict(row) for row in self.rows]


def make_dataframe(rows: Iterable[Dict], columns: Sequence[str] | None = None):
    """优先返回 pandas.DataFrame；不可用时返回 SimpleDataFrame。"""

    rows = list(rows)
    try:
        import pandas as pd

        return pd.DataFrame(rows, columns=columns)
    except ModuleNotFoundError:
        return SimpleDataFrame(rows, columns)


def records(table) -> List[Dict]:
    """从 pandas 或 SimpleDataFrame 中取出记录列表。"""

    if hasattr(table, "to_dict"):
        return list(table.to_dict("records"))
    return list(table)
