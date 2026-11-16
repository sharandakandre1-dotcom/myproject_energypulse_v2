"""Шар зберігання даних. Структура «зірка» (5 таблиць) відтворена в SQLite, щоб застосунок
запускався без окремого сервера. Еквівалентний скрипт для MS SQL Server: schema_mssql.sql
(в цьому прототипі не виконується)."""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from calendar_utils import is_holiday_index

DB_PATH = "energy_pulse.db"

DDL = """
CREATE TABLE IF NOT EXISTS Dim_Time (
    time_id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT NOT NULL UNIQUE,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    month INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL,
    is_holiday INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS Dim_Object (
    object_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    object_type TEXT,
    area_m2 REAL,
    location TEXT
);
CREATE TABLE IF NOT EXISTS Dim_Model (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    mae REAL,
    rmse REAL,
    mape REAL,
    trained_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS Fact_Consumption (
    consumption_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_id INTEGER NOT NULL REFERENCES Dim_Time(time_id),
    object_id INTEGER NOT NULL REFERENCES Dim_Object(object_id),
    energy_kwh REAL NOT NULL,
    power_kw REAL,
    temperature_c REAL,
    is_anomaly INTEGER NOT NULL DEFAULT 0,
    UNIQUE (time_id, object_id)
);
CREATE INDEX IF NOT EXISTS ix_fact_consumption_time_obj ON Fact_Consumption(time_id, object_id);
CREATE TABLE IF NOT EXISTS Fact_Forecast (
    forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_id INTEGER NOT NULL REFERENCES Dim_Time(time_id),
    object_id INTEGER NOT NULL REFERENCES Dim_Object(object_id),
    model_id INTEGER NOT NULL REFERENCES Dim_Model(model_id),
    predicted_kwh REAL NOT NULL,
    horizon_h INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS App_User (
    username TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    salt TEXT NOT NULL,
    pwd_hash TEXT NOT NULL
);
"""

TABLES = ["Dim_Time", "Dim_Object", "Dim_Model", "Fact_Consumption", "Fact_Forecast"]


@contextmanager
def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


def _fmt(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def upsert_times(conn, timestamps):
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps).unique())).sort_values()
    hol = is_holiday_index(ts)
    rows = [(_fmt(t), t.hour, t.dayofweek, t.month, int(t.dayofweek >= 5), int(h))
            for t, h in zip(ts, hol)]
    conn.executemany(
        "INSERT OR IGNORE INTO Dim_Time(datetime, hour, day_of_week, month, is_weekend, is_holiday) "
        "VALUES (?,?,?,?,?,?)", rows)
    m = pd.read_sql("SELECT time_id, datetime FROM Dim_Time", conn)
    return dict(zip(m["datetime"], m["time_id"]))


def upsert_objects(conn, names, meta=None):
    meta = meta or {}
    for n in names:
        t, a, loc = meta.get(n, (None, None, None))
        conn.execute("INSERT OR IGNORE INTO Dim_Object(name, object_type, area_m2, location) VALUES (?,?,?,?)",
                     (n, t, _f(a), loc))
    m = pd.read_sql("SELECT object_id, name FROM Dim_Object", conn)
    return dict(zip(m["name"], m["object_id"]))


def clear_all(conn):
    for t in ["Fact_Forecast", "Fact_Consumption", "Dim_Model", "Dim_Time", "Dim_Object"]:
        conn.execute(f"DELETE FROM {t}")
    conn.execute("DELETE FROM sqlite_sequence")


def save_consumption(df, replace=False, path=None):
    """Зберігає довгу таблицю (timestamp, object_name, consumption[, temperature, power_kw,
    object_type, area_m2, location]). Дані з кроком менше години агрегуються до години."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.floor("h")
    for c, default in [("temperature", float("nan")), ("power_kw", float("nan")),
                       ("object_type", None), ("area_m2", float("nan")), ("location", None)]:
        if c not in d.columns:
            d[c] = default
    agg = d.groupby(["timestamp", "object_name"], as_index=False).agg(
        consumption=("consumption", "sum"), temperature=("temperature", "mean"),
        power_kw=("power_kw", "mean"), object_type=("object_type", "first"),
        area_m2=("area_m2", "first"), location=("location", "first"))
    with connect(path) as conn:
        if replace:
            clear_all(conn)
        meta = {r.object_name: (r.object_type, r.area_m2, r.location)
                for r in agg.drop_duplicates("object_name").itertuples()}
        tmap = upsert_times(conn, agg["timestamp"])
        omap = upsert_objects(conn, agg["object_name"].unique(), meta)
        rows = [(int(tmap[_fmt(r.timestamp)]), int(omap[r.object_name]), float(r.consumption),
                 _f(r.power_kw), _f(r.temperature))
                for r in agg.itertuples()]
        conn.executemany(
            "INSERT INTO Fact_Consumption(time_id, object_id, energy_kwh, power_kw, temperature_c) "
            "VALUES (?,?,?,?,?) ON CONFLICT(time_id, object_id) DO UPDATE SET "
            "energy_kwh=excluded.energy_kwh, power_kw=excluded.power_kw, temperature_c=excluded.temperature_c",
            rows)
    return len(rows)


def load_consumption(path=None):
    with connect(path) as conn:
        df = pd.read_sql(
            "SELECT t.datetime AS timestamp, o.name AS object_name, o.object_type, o.area_m2, "
            "f.energy_kwh AS consumption, f.power_kw, f.temperature_c AS temperature, "
            "t.is_weekend, t.is_holiday, f.is_anomaly "
            "FROM Fact_Consumption f JOIN Dim_Time t ON t.time_id = f.time_id "
            "JOIN Dim_Object o ON o.object_id = f.object_id ORDER BY t.datetime", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def data_version(path=None):
    with connect(path) as conn:
        r = conn.execute("SELECT COUNT(*), COALESCE(MAX(consumption_id),0), COALESCE(SUM(is_anomaly),0) "
                         "FROM Fact_Consumption").fetchone()
    return tuple(r)


def table_counts(path=None):
    with connect(path) as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


def save_models(results, trained_at, path=None):
    """results: список dict(model_name, mae, rmse, mape). Повертає {model_name: model_id}."""
    ids = {}
    with connect(path) as conn:
        for r in results:
            cur = conn.execute("INSERT INTO Dim_Model(model_name, mae, rmse, mape, trained_at) VALUES (?,?,?,?,?)",
                               (r["model_name"], _f(r["mae"]), _f(r["rmse"]), _f(r["mape"]), trained_at))
            ids[r["model_name"]] = cur.lastrowid
    return ids


def save_forecast(scope_name, model_id, fc_df, step_hours, path=None):
    """fc_df: timestamp, forecast. scope_name: об'єкт або сума кількох об'єктів."""
    with connect(path) as conn:
        omap = upsert_objects(conn, [scope_name], {scope_name: ("aggregate", None, None)})
        tmap = upsert_times(conn, fc_df["timestamp"])
        rows = [(tmap[_fmt(r.timestamp)], omap[scope_name], model_id, float(r.forecast), int((i + 1) * step_hours))
                for i, r in enumerate(fc_df.itertuples())]
        conn.executemany("INSERT INTO Fact_Forecast(time_id, object_id, model_id, predicted_kwh, horizon_h) "
                         "VALUES (?,?,?,?,?)", rows)
    return len(rows)


def load_forecasts(limit=500, path=None):
    with connect(path) as conn:
        df = pd.read_sql(
            "SELECT t.datetime AS timestamp, o.name AS object_name, m.model_name, "
            "f.predicted_kwh, f.horizon_h, m.trained_at FROM Fact_Forecast f "
            "JOIN Dim_Time t ON t.time_id=f.time_id JOIN Dim_Object o ON o.object_id=f.object_id "
            "JOIN Dim_Model m ON m.model_id=f.model_id ORDER BY m.trained_at DESC, f.forecast_id DESC "
            f"LIMIT {int(limit)}", conn)
    return df


def load_models(path=None):
    with connect(path) as conn:
        return pd.read_sql("SELECT model_id, model_name, mae, rmse, mape, trained_at FROM Dim_Model "
                           "ORDER BY trained_at DESC, rmse ASC", conn)


def load_objects(path=None):
    with connect(path) as conn:
        return pd.read_sql("SELECT object_id, name, object_type, area_m2, location FROM Dim_Object "
                           "WHERE COALESCE(object_type,'') <> 'aggregate' ORDER BY name", conn)


def update_objects(df, path=None):
    with connect(path) as conn:
        for r in df.itertuples():
            conn.execute("UPDATE Dim_Object SET object_type=?, area_m2=?, location=? WHERE object_id=?",
                         (r.object_type, _f(r.area_m2), r.location, int(r.object_id)))


def set_anomaly_flags(object_name, checked_from, checked_to, flagged, gran, path=None):
    """Скидає позначки на перевіреному інтервалі та ставить нові. gran: 'h' або 'D'."""
    with connect(path) as conn:
        r = conn.execute("SELECT object_id FROM Dim_Object WHERE name=?", (object_name,)).fetchone()
        if r is None:
            return 0
        oid = r[0]
        conn.execute("UPDATE Fact_Consumption SET is_anomaly=0 WHERE object_id=? AND time_id IN "
                     "(SELECT time_id FROM Dim_Time WHERE datetime BETWEEN ? AND ?)",
                     (oid, _fmt(checked_from), _fmt(pd.Timestamp(checked_to) + pd.Timedelta(hours=23 if gran == "D" else 0))))
        n = 0
        for ts in flagged:
            if gran == "D":
                cur = conn.execute("UPDATE Fact_Consumption SET is_anomaly=1 WHERE object_id=? AND time_id IN "
                                   "(SELECT time_id FROM Dim_Time WHERE substr(datetime,1,10)=?)",
                                   (oid, pd.Timestamp(ts).strftime("%Y-%m-%d")))
            else:
                cur = conn.execute("UPDATE Fact_Consumption SET is_anomaly=1 WHERE object_id=? AND time_id IN "
                                   "(SELECT time_id FROM Dim_Time WHERE datetime=?)", (oid, _fmt(ts)))
            n += cur.rowcount
    return n
