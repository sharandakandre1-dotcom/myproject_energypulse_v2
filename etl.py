"""Модуль збору та перевірки даних (ETL): читання CSV/Excel, нормалізація назв стовпців,
перетворення широкого формату (як у BDG2) у довгий, перевірка якості."""
import io

import numpy as np
import pandas as pd

SYNONYMS = {
    "timestamp": ["timestamp", "datetime", "date", "time", "дата", "час", "дата і час"],
    "object_name": ["object_name", "object", "building_id", "building", "meter", "об'єкт", "обєкт", "назва"],
    "consumption": ["consumption", "energy", "energy_kwh", "kwh", "meter_reading", "споживання", "кВт·год"],
    "temperature": ["temperature", "temperature_c", "airtemperature", "temp", "температура"],
    "power_kw": ["power_kw", "power", "потужність"],
    "object_type": ["object_type", "primaryspaceusage", "тип"],
    "area_m2": ["area_m2", "sqm", "площа"],
}


def read_table(name, raw_bytes):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", encoding_errors="replace", sep=None, engine="python")


def normalize_columns(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for target, names in SYNONYMS.items():
        for n in names:
            if n.lower() in lower and lower[n.lower()] not in rename and target not in rename.values():
                rename[lower[n.lower()]] = target
                break
    return df.rename(columns=rename)


def detect_format(df):
    """'long' якщо є timestamp, object_name, consumption; 'wide' якщо timestamp + багато числових стовпців."""
    cols = set(df.columns)
    if {"timestamp", "object_name", "consumption"} <= cols:
        return "long"
    if "timestamp" in cols:
        num = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
        if len(num) >= 1:
            return "wide"
    return "unknown"


def wide_columns(df):
    return [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]


def wide_to_long(df, columns):
    out = df[["timestamp"] + list(columns)].melt(id_vars="timestamp", var_name="object_name", value_name="consumption")
    out["object_name"] = out["object_name"].astype(str)
    return out


def attach_weather(long_df, weather):
    """Додає температуру. weather: timestamp, temperature[, site_id]. Для BDG2 назва об'єкта
    починається з ідентифікатора майданчика (до першого '_')."""
    w = normalize_columns(weather.copy())
    if "timestamp" not in w.columns or "temperature" not in w.columns:
        raise ValueError("У файлі погоди потрібні стовпці з часом і температурою.")
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w = w.dropna(subset=["timestamp"])
    d = long_df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    site_col = next((c for c in w.columns if str(c).lower() == "site_id"), None)
    if site_col is not None:
        d["_site"] = d["object_name"].astype(str).str.split("_").str[0].str.lower()
        w["_site"] = w[site_col].astype(str).str.lower()
        w = w.drop_duplicates(["_site", "timestamp"])[["_site", "timestamp", "temperature"]]
        d = d.drop(columns=[c for c in ["temperature"] if c in d.columns]).merge(w, on=["_site", "timestamp"], how="left")
        d = d.drop(columns="_site")
    else:
        w = w.drop_duplicates("timestamp")[["timestamp", "temperature"]]
        d = d.drop(columns=[c for c in ["temperature"] if c in d.columns]).merge(w, on="timestamp", how="left")
    return d


def validate_and_clean(df):
    """Повертає (очищений df, звіт). Некоректні рядки відкидаються, статистика фіксується."""
    rep = {"рядків у файлі": int(len(df))}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    bad_t = int(d["timestamp"].isna().sum())
    d["consumption"] = pd.to_numeric(d["consumption"], errors="coerce")
    bad_v = int(d["consumption"].isna().sum())
    d = d.dropna(subset=["timestamp", "consumption"])
    neg = int((d["consumption"] < 0).sum())
    d = d[d["consumption"] >= 0]
    for c in ["temperature", "power_kw", "area_m2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["object_name"] = d["object_name"].astype(str).str.strip()
    before = len(d)
    d = d.drop_duplicates(["timestamp", "object_name"], keep="last")
    rep.update({"некоректний час": bad_t, "некоректне значення": bad_v, "від'ємні значення": neg,
                "дублікати": int(before - len(d)), "коректних рядків": int(len(d)),
                "об'єктів": int(d["object_name"].nunique())})
    if "temperature" in d.columns:
        rep["рядків без температури"] = int(d["temperature"].isna().sum())
    d = d.sort_values(["object_name", "timestamp"]).reset_index(drop=True)
    return d, rep


def iqr_outlier_mask(values, k=3.0):
    """Позначає (не видаляє) екстремальні значення за правилом міжквартильного розмаху."""
    v = np.asarray(values, dtype=float)
    q1, q3 = np.nanpercentile(v, [25, 75])
    iqr = q3 - q1
    return (v < q1 - k * iqr) | (v > q3 + k * iqr)
