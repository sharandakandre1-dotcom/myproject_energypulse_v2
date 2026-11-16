"""Генератор синтетичних демонстраційних даних (погодинних). Використовується лише для
налагодження інтерфейсу; не є результатами реального енергетичного обстеження."""
import numpy as np
import pandas as pd

from calendar_utils import is_holiday_index

OBJECTS = [
    ("Корпус А", "навчальний корпус", 5200.0, "Київ"),
    ("Корпус Б", "лабораторний корпус", 3800.0, "Київ"),
    ("Навчальний центр", "навчальний центр", 2600.0, "Київ"),
]
BASE_KW = [38.0, 27.0, 19.0]


def make_demo_data(days=180, seed=28):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.date_range(end=end, periods=days * 24, freq="h")
    hour = np.asarray(idx.hour)
    doy = np.asarray(idx.dayofyear)
    weekend = (np.asarray(idx.dayofweek) >= 5)
    holiday = is_holiday_index(idx).astype(bool)

    temp = (9 + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25)
            + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.2, len(idx)))
    day_profile = 0.35 + 0.65 * np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None)
    off_factor = np.where(weekend | holiday, 0.6, 1.0)
    heat = 1.0 + 0.02 * np.clip(15 - temp, 0, None)

    frames = []
    for j, (name, otype, area, loc) in enumerate(OBJECTS):
        base = BASE_KW[j]
        val = base * day_profile * off_factor * heat + rng.normal(0, 0.05 * base, len(idx))
        if j == 0:
            for k in rng.choice(np.arange(200, len(idx) - 200), 3, replace=False):
                val[k] *= 2.2          # штучні піки
        if j == 1:
            k = int(rng.integers(200, len(idx) - 200))
            val[k] *= 0.1              # штучний провал
        val = np.clip(val, 1.0, None)
        frames.append(pd.DataFrame({
            "timestamp": idx, "object_name": name, "consumption": np.round(val, 3),
            "power_kw": np.round(val, 3), "temperature": np.round(temp, 1),
            "object_type": otype, "area_m2": area, "location": loc,
        }))
    return pd.concat(frames, ignore_index=True)
