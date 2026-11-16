"""Календарні ознаки: вихідні, державні свята (фіксовані дати), синусно-косинусне кодування."""
import numpy as np
import pandas as pd

# Фіксовані державні свята України (місяць, день). Рухомі свята (Великдень, Трійця)
# не враховано; за потреби список можна розширити.
FIXED_HOLIDAYS = {(1, 1), (3, 8), (5, 1), (6, 28), (8, 24), (12, 25)}


def is_holiday_index(idx):
    idx = pd.DatetimeIndex(idx)
    return np.array([int((m, d) in FIXED_HOLIDAYS) for m, d in zip(idx.month, idx.day)], dtype=int)


def calendar_frame(idx):
    """Календарні ознаки для набору часових міток."""
    idx = pd.DatetimeIndex(idx)
    cal = pd.DataFrame({
        "hour": np.asarray(idx.hour, dtype=int),
        "dayofweek": np.asarray(idx.dayofweek, dtype=int),
        "month": np.asarray(idx.month, dtype=int),
        "dayofyear": np.asarray(idx.dayofyear, dtype=int),
    })
    cal["is_weekend"] = (cal["dayofweek"] >= 5).astype(int)
    cal["is_holiday"] = is_holiday_index(idx)
    return cal
