"""Підготовка рядів та формування ознак (календарні, лагові, ковзні, температурні)."""
import numpy as np
import pandas as pd

from calendar_utils import calendar_frame


def detect_granularity(df):
    first = df["object_name"].iloc[0]
    ts = np.sort(df.loc[df["object_name"] == first, "timestamp"].unique())
    if len(ts) < 2:
        return "D"
    med = pd.Series(ts).diff().median()
    return "h" if med < pd.Timedelta(hours=23) else "D"


def build_series(df, objects, gran):
    """Сума вибраних об'єктів у єдиний регулярний ряд. Повертає (series, info)."""
    d = df[df["object_name"].isin(objects)].copy()
    if d.empty:
        raise ValueError("Немає даних для вибраних об'єктів і періоду.")
    d["ts"] = d["timestamp"].dt.floor(gran)
    g = d.groupby("ts").agg(consumption=("consumption", "sum"), temperature=("temperature", "mean"),
                            n=("consumption", "size"))
    g = g[g["n"] == g["n"].max()]                       # лише повні інтервали
    full = pd.date_range(g.index.min(), g.index.max(), freq=gran)
    g = g.reindex(full)
    limit = 3 if gran == "h" else 2
    g["consumption"] = g["consumption"].interpolate(limit=limit, limit_area="inside")
    info = {"truncated": False}
    ok = g["consumption"].notna().to_numpy()
    if not ok.all():
        start = int(np.where(~ok)[0].max()) + 1           # беремо останній безперервний фрагмент
        g = g.iloc[start:]
        info["truncated"] = True
    g["temperature"] = g["temperature"].interpolate(limit_area="inside").ffill().bfill()
    use_temp = bool(g["temperature"].notna().all())
    out = pd.DataFrame({"timestamp": g.index, "consumption": g["consumption"].to_numpy(float),
                        "temperature": g["temperature"].to_numpy(float) if use_temp else np.nan})
    info.update({"n": len(out), "has_temp": use_temp})
    return out.reset_index(drop=True), info


def feature_columns(gran, use_temp):
    cols = ["hour", "dayofweek", "month", "is_weekend", "is_holiday"] if gran == "h" else \
           ["dayofweek", "month", "dayofyear", "is_weekend", "is_holiday"]
    if use_temp:
        cols.append("temperature")
    cols += ["lag_1", "lag_24", "lag_168", "roll_24"] if gran == "h" else ["lag_1", "lag_7", "roll_7"]
    return cols


def make_features(series, gran, use_temp):
    """Векторизований набір ознак для навчання. Ковзне середнє рахується за попередні значення."""
    y = series["consumption"]
    f = calendar_frame(series["timestamp"])
    if use_temp:
        f["temperature"] = series["temperature"].to_numpy(float)
    f["lag_1"] = y.shift(1).to_numpy()
    if gran == "h":
        f["lag_24"] = y.shift(24).to_numpy()
        f["lag_168"] = y.shift(168).to_numpy()
        f["roll_24"] = y.shift(1).rolling(24).mean().to_numpy()
    else:
        f["lag_7"] = y.shift(7).to_numpy()
        f["roll_7"] = y.shift(1).rolling(7).mean().to_numpy()
    f["y"] = y.to_numpy()
    f["timestamp"] = series["timestamp"].to_numpy()
    return f.dropna().reset_index(drop=True)


def expected_temp(temps, gran):
    """Очікувана температура майбутнього кроку: та сама година попередніх діб / середнє за 7 діб."""
    if gran == "h":
        vals = [temps[-24 * k] for k in range(1, 8) if len(temps) >= 24 * k]
    else:
        vals = temps[-7:]
    return float(np.mean(vals))


def build_row(vals, cal, k, t_exp, gran, use_temp):
    r = {"dayofweek": cal["dayofweek"][k], "month": cal["month"][k],
         "is_weekend": cal["is_weekend"][k], "is_holiday": cal["is_holiday"][k]}
    r["lag_1"] = vals[-1]
    if gran == "h":
        r["hour"] = cal["hour"][k]
        r["lag_24"], r["lag_168"] = vals[-24], vals[-168]
        r["roll_24"] = float(np.mean(vals[-24:]))
    else:
        r["dayofyear"] = cal["dayofyear"][k]
        r["lag_7"] = vals[-7]
        r["roll_7"] = float(np.mean(vals[-7:]))
    if use_temp:
        r["temperature"] = t_exp
    return r


def exog_matrix(cal, temp_scaled, gran, use_temp):
    """Зовнішні ознаки для LSTM (циклічне кодування календаря, вихідні, свята, температура)."""
    two_pi = 2 * np.pi
    cols = [np.sin(two_pi * cal["dayofweek"] / 7), np.cos(two_pi * cal["dayofweek"] / 7),
            np.sin(two_pi * (cal["month"] - 1) / 12), np.cos(two_pi * (cal["month"] - 1) / 12)]
    if gran == "h":
        cols += [np.sin(two_pi * cal["hour"] / 24), np.cos(two_pi * cal["hour"] / 24)]
    cols += [cal["is_weekend"], cal["is_holiday"]]
    if use_temp:
        cols.append(temp_scaled)
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])
