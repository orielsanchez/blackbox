from datetime import timedelta
from typing import List, Tuple

import pandas as pd


def generate_walkforward_windows(
    start_date: str, end_date: str, train_days: int, test_days: int
) -> List[Tuple[Tuple[pd.Timestamp, pd.Timestamp], Tuple[pd.Timestamp, pd.Timestamp]]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []

    current = start
    while current + timedelta(days=train_days + test_days) <= end:
        train_start = current
        train_end = current + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        windows.append(((train_start, train_end), (test_start, test_end)))
        current += timedelta(days=test_days)

    return windows
